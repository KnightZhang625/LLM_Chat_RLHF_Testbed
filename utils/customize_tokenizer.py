# coding:utf-8
# Author: Jiaxin Zhang
# Date: 16th-Aug-2023

from transformers import AutoTokenizer
from collections import defaultdict

class BPETokenizer:
    def __init__(self, tokenizer_name="gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def preprocess(self, str):
        return self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(str)
    
    def build(self, corpus, dict_len):
        # split by whitespace and normalize
        words_with_offsets = [self.preprocess(sent) for sent in corpus]
        
        # count the word frequency
        word_frequency = self.__count_word_frequency(words_with_offsets)
        word_to_chars_mapping = {word: [w for w in word] for word in word_frequency.keys()}
        
        # count the char frequency
        char_frequency = self.__count_char_frequency(word_frequency, word_to_chars_mapping)

        # initialize the vocab
        vocab = ["<|endoftext|>"] + list(set([c for char_list in word_to_chars_mapping.values() for c in char_list]))
        merge_dict = {}
        
        while len(vocab) < dict_len:
            selected_pair = self.find_maximum_pair(char_frequency)
            vocab.append("".join(selected_pair))
            merge_dict[selected_pair] = "".join(selected_pair)
            
            word_to_chars_mapping = self.merge(word_to_chars_mapping, selected_pair)
            char_frequency = self.__count_char_frequency(word_frequency, word_to_chars_mapping)
        return vocab, merge_dict
        
    def __count_word_frequency(self, words_with_offsets):
        word_frequency = defaultdict(int)
        for word_offset_list in words_with_offsets:
            for word_offset in word_offset_list:
                word, _ = word_offset
                word_frequency[word] +=1
        return word_frequency
    
    def __count_char_frequency(self, word_frequency, word_to_chars_mapping):
        char_frequency = defaultdict(int)
        for word, freq in word_frequency.items():
            chars = word_to_chars_mapping[word]
            if len(chars) == 1:
                continue
            i = 0
            while i < (len(chars) - 1):
                pair_a = chars[i]
                pair_b = chars[i+1]
                char_frequency[(pair_a, pair_b)] += freq
                i +=1
        return char_frequency

    def merge(self, word_to_chars_mapping, pair):
        pair_a, pair_b = pair
        for word, chars in word_to_chars_mapping.items():
            i = 0
            while i < (len(chars) - 1):
                a = chars[i]
                b = chars[i+1]
                if (pair_a == a) and (pair_b == b):
                    chars = chars[:i] + [f"{pair_a}{pair_b}"] + chars[i+2:] 
                else:
                    i +=1
            word_to_chars_mapping[word] = chars
        return word_to_chars_mapping

    def find_maximum_pair(self, char_frequency):
        return sorted(char_frequency.items(), key=lambda items: items[1], reverse=True)[0][0]
    
    def tokenize(self, str, merge_dict):
        processed_str = [word_with_offset[0] for word_with_offset in self.preprocess(str)]
        chars_list = [[c for c in word] for word in processed_str]
        
        for pair, replace in merge_dict.items():
            for idx, chars in enumerate(chars_list):
                pair_a, pair_b = pair
                i = 0
                while i < (len(chars) - 1):
                    a = chars[i]
                    b = chars[i+1]
                    if (pair_a == a) and (pair_b == b):
                        chars = chars[:i] + [replace] + chars[i+2:]
                    else:
                        i +=1
                chars_list[idx] = chars
        return sum(chars_list, [])

if __name__ == "__main__":
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    tokenizer = BPETokenizer()
    vocab, merge_dict = tokenizer.build(corpus, 50)
    res = tokenizer.tokenize("This is not a token.", merge_dict)
    print(res)