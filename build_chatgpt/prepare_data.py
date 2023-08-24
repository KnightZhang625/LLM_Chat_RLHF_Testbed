import os
import requests
import tiktoken
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import time
import datetime

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open("data/input.txt", 'r') as f:
    data = f.read()
n = len(data)

tokens = list(set(data))
token_to_ids = {t:i for i, t in enumerate(tokens)}
idx_to_token = {i:t for i, t in enumerate(tokens)}

encode_fn = lambda sent: [token_to_ids[t] for t in sent]
decode_fn = lambda ids: "".join([idx_to_token[i] for i in ids])

train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

train_data = torch.tensor(encode_fn(train_data), dtype=torch.long)
val_data = torch.tensor(encode_fn(val_data), dtype=torch.long)

torch.manual_seed(1337)
batch_size = 64
block_size = 256

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y

# xb: [bsz, block_size]
# yb: [bsz, block_size]
xb, yb = get_batch("train")
for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b, t]
        print(f"Context: {context} Target: {target}")

class LayerNorm(object):
    def __init__(self, dim, eps=1e-5, momentum=0.1):
        self.eps = eps
        # self.momentum = momentum
        # self.training = True
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
        # self.running_mean = torch.zeros(dim)
        # self.running_var = torch.ones(dim)
    
    def __call__(self, x):
        # if self.training:
        #     xmean = x.mean(1, keepdim=True)
        #     xvar = x.var(1, keepdim=True)
        # else:
        #     xmean = self.running_mean
        #     xvar = self.running_var

        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - x_mean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        # if self.training:
        #     with torch.no_grad():
        #         self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * xmean
        #         self.running_var = (1 - self.momentum) * self.running_var + self.momentum * xvar
        
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]

class Head(nn.Module):
    def __init__(self, head_size, num_heads):
        super(Head, self).__init__()
        
        self.num_heads = num_heads
        self.key = nn.Linear(384, head_size, bias=False)
        self.query = nn.Linear(384, head_size, bias=False)
        self.value = nn.Linear(384, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones((block_size, block_size))))
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x)     # [B, T, C]
        q = self.query(x)   # [B, T, C]
        
        assert C % self.num_heads == 0
        sub_c = C // self.num_heads
        k_reshaped = torch.reshape(k, (B, T, self.num_heads, sub_c)).transpose(1, 2) # [B, num_heads, T, sub_c]
        q_reshaped = torch.reshape(q, (B, T, self.num_heads, sub_c)).transpose(1, 2) # [B, num_heads, T, sub_c]
        
        # "C**0.5" is the scale, the reason is to keep variance at the initialization,
        # if the one score in "attention_weights" is extremely high, its "attention_score" from softmax
        # will be close to 1, so that "attention_score" is like one-hot-encoding.
        # then the context_vector will only focuses on one particularly token.
        attention_weights = q_reshaped @ k_reshaped.transpose(-2, -1) / C**0.5
        attention_weights = attention_weights.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        attention_scores = F.softmax(attention_weights, dim=-1) # [B, num_heads, T, T]
        attention_scores = self.dropout(attention_scores)
        
        v = self.value(x)   # [B, T, C]
        v_reshape = torch.reshape(v, (B, T, self.num_heads, sub_c)) # [B, T, num_heads, sub_c]
        context_vector = attention_scores @ v_reshape.transpose(1, 2)   # [B, num_heads, T, C]
        context_vector = torch.reshape(context_vector.transpose(1, 2), (B, T, -1))
        return context_vector

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        # self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.heads = Head(head_size, num_heads)
        self.proj = nn.Linear(384, 384)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.size()
        out = self.heads(x)  # [B, T, C]
        
        # out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(0.1),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super(Block, self).__init__()
        self.sa = MultiHeadAttention(n_head, n_embed)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = self.sa(self.ln1(x)) + x
        # residual network here, 
        # at the beginning, the backpropogation will have the highway directly to the x due adding x here,
        # not through the self.sa and self.ffwd. This makes training deep neural network efficiently.
        x = self.ffwd(self.ln2(x)) + x
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed=384):
        super(BigramLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
     
        self.blocks = nn.Sequential(
            Block(n_embed, 6),
            Block(n_embed, 6),
            Block(n_embed, 6),
            Block(n_embed, 6),
            Block(n_embed, 6),
            Block(n_embed, 6),
            nn.LayerNorm(n_embed),
        )
     
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.embeddings(idx)  # [b, T, e]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # [T, e]
        x = tok_emb + pos_emb 
        x = self.blocks(x)
        logits = self.lm_head(x)  # [b, T, t]
        
        if targets == None:
            loss = None
        else:
            bsz, seq, c = logits.size()
            logits = torch.reshape(logits, (bsz * seq, c))
            targets = torch.reshape(targets, (-1, ))
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_context = idx[:, -block_size:]
            logits, _ = self(idx_context)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(len(tokens))
model.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(100)
        for k in range(100):
            X, Y = get_batch(split)
            X = X.to(device)
            Y = Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out

parameters = [
    {"params": [p for n, p in model.named_parameters()],
     "lr": 3e-4}
]
optimizer = torch.optim.AdamW(parameters)
for step in range(10000):
    xb, yb = get_batch("train")
    xb = xb.to(device)
    yb = yb.to(device)
    
    logits, loss = model(xb, yb)
    loss.backward()
    # print(f"Step: [{step}], loss: {loss.item():.2f}")
    optimizer.step()
    optimizer.zero_grad()
    
    if step % 100 == 0:
        print(estimate_loss(model))

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
pred_ids = model.generate(idx, 1000)
print(decode_fn(pred_ids[0].tolist()))
