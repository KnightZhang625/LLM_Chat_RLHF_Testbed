# coding:utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

class LayerNorm(nn.Module):
    
    def __init__(self, ndim, bias):
        super().__init__()
        self.ndim = ndim
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj_dropout = nn.Dropout(config.dropout)
        causal_mask = torch.tril(
            torch.ones(config.block_size, config.block_size).view(1, 1, config.block_size, config.block_size)
        )
        self.register_buffer("causal_mask", causal_mask)
        
        self.n_embed = config.n_embed
        self.n_head = config.n_head
    
    def forward(self, x):
        # B: batch_size, T: block_size, C: embedding dim
        B, T, C = x.size()
        
        # [B, T, 3*C] -> ([B, T, C], [B, T, C], [B, T, C])
        q, k, v = torch.split(self.c_attn(x), self.n_embed, -1)
        # [B, T, C] -> [B, T, n_head, n_embed // n_head] -> [B, n_head, T, n_embed // n_head]
        q_to_head = q.view(B, T, self.n_head, self.n_embed // self.n_head).contiguous().transpose(1, 2)
        k_to_head = k.view(B, T, self.n_head, self.n_embed // self.n_head).contiguous().transpose(1, 2)
        v_to_head = v.view(B, T, self.n_head, self.n_embed // self.n_head).contiguous().transpose(1, 2)
        
        scale = 1 / math.sqrt(k_to_head.size(-1))
        attention_weights = q_to_head @ k_to_head.transpose(-2, -1) * scale
        attention_weights = attention_weights.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        # [B, n_head, T, T]
        attention_scores = F.softmax(attention_weigths, dim=-1)
        attention_scores = self.attn_dropout(attention_scores)

        # [B, n_head, T, n_embed // n_head]
        context_vector = attention_scores @ v_to_head
        context_vector = context_vector.transpose(1, 2).view(B, T, self.n_embed).contiguous()
        context_vector = self.c_proj(context_vector)
        context_vector = self.proj_dropout(context_vector)
        
        return context_vector

class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embed, config.bias)
        self.attn = CausalAttention(config)
        self.ln_2 = LayerNorm(config.n_embed, config.bias)
        self.mlp = MLP(config)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_szie: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768
    dropout: float = 0.0
    bias: bool = True
    
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size != None
        assert config.block_size != None
        self.config = config
    
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embed),
                wpe = nn.Embedding(config.vocab_size, config.n_embed),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embed, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        for pn, p in slef.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
                
        print(f"number of parameters: {self.get_num_params()/1e6:.2f}Millon")
    
    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)    # [b, t, n_embed]
        
        if targets != None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logis.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            # [B, 1, C]
            logits = self.lm_head(x[:, [-1], :])    # note: using list [-1] to preserve the time dim
        
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "causal_mask"):
                block.attn.causal_mask = block.attn.causal_mask[:, :, :block_size, :block_size]
    
    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2=large", "gpt2-xl"}
        override_args = override_args if override_args != None else {}
        assert all(k == "dropout" for k in override_args)
        
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)
        
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.causal_mask")]
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_hf_keys = sd_hf.keys()
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(".attn.masked_bias")]
        sd_hf_keys = [k for k in sd_hf_keys if not k.endswith(".attn.bias")]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]
        assert len(sd_keys) == len(sd_hf_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_hf_keys:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.required_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rage, betas=betas)

        return optimizer
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            # [B, C]
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        
        return idx