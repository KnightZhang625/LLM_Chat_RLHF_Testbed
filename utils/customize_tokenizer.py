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

class WordPieceTokenizer:
    def __init__(self, tokenizer_name="bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    def build(self, corpus, dict_len):
        words_with_offset = [self.preprocess(sent) for sent in corpus]

        words_frequency = self.__count_word_frequency(words_with_offset)
        
        word_to_chars_mapping = self.split_word(list(words_frequency.keys()))

        single_char_frequency = self.__count_single_char_frequence(word_to_chars_mapping, words_frequency)
 
        pair_frequency = self.__count_pair_frequency(words_frequency, word_to_chars_mapping)

        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + sorted(list(single_char_frequency.keys()))
      
        while len(vocab) < dict_len:
            selected_pair = self.select_pair(pair_frequency, single_char_frequency)
          
            word_to_chars_mapping = self.merge(selected_pair, word_to_chars_mapping)

            single_char_frequency = self.__count_single_char_frequence(word_to_chars_mapping, words_frequency)
        
            pair_frequency = self.__count_pair_frequency(words_frequency, word_to_chars_mapping)
            
            vocab.append(f"{selected_pair[0]}{selected_pair[1].strip('##')}")

        return vocab
        
    def preprocess(self, sent):
        return self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sent)
    
    def merge(self, selected_pair, word_to_chars_mapping):
        pair_a, pair_b = selected_pair
        for word, chars in word_to_chars_mapping.items():
            if len(chars) == 1:
                continue
            i = 0
            while i < (len(chars) - 1):
                a = chars[i]
                b = chars[i+1]
                if (pair_a == a) and (pair_b == b):
                    b = b.strip("#")
                    chars = chars[:i] + [f"{a}{b.strip('##')}"] + chars[i+2:]
                else:
                    i +=1
            word_to_chars_mapping[word] = chars
        return word_to_chars_mapping
        
    def __count_single_char_frequence(self, word_to_chars_mapping, words_frequency):
        single_char_frequency = defaultdict(int)
        for word, chars in word_to_chars_mapping.items():
            freq = words_frequency[word]
            for c in chars:
                single_char_frequency[c] += freq
        return single_char_frequency

    def __count_word_frequency(self, words_with_offset):
        words_frequency = defaultdict(int)
        for sent in words_with_offset:
            for word_offset in sent:
                word, _ = word_offset
                words_frequency[word] +=1
        return words_frequency

    def __count_pair_frequency(self, words_frequency, word_to_chars_mapping):
        pair_frequency = defaultdict(int)
        for word, chars in word_to_chars_mapping.items():
            freq = words_frequency[word]
            if len(chars) == 1:
                continue
            i = 0
            while i < (len(chars) - 1):
                pair_a = chars[i]
                pair_b = chars[i+1]
                pair_frequency[(pair_a, pair_b)] += freq
                i +=1
        return pair_frequency

    def select_pair(self, pair_frequency, single_char_frequency):
        max_score = -1
        selected_pair = None
        for pair, freq in pair_frequency.items():
            single_a = pair[0]
            single_b = pair[1]
            score = self.calculate_score(freq, single_char_frequency[single_a], single_char_frequency[single_b])
            if score > max_score:
                max_score = score
                selected_pair = pair
        return selected_pair
    
    def calculate_score(self, pair_freq, single_a_freq, single_b_freq):
        return pair_freq / (single_a_freq * single_b_freq)
                                          
    def split_word(self, words):
        word_to_chars_mapping = {}
        for word in words:
            chars = []
            for idx, w in enumerate(word):
                if idx == 0:
                    chars.append(w)
                else:
                    chars.append(f"##{w}")
            word_to_chars_mapping[word] = chars
        return word_to_chars_mapping

    def tokenize_word(self, word, vocab):
        tokens = []
        while len(word) > 0:
            i = len(word)
            while i > 0 and word[:i] not in vocab:
                i -=1
            if i == 0:
                return ["[UNK]"]
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = f"##{word}"
        return tokens

    def tokenize(self, sent, vocab):
        words_with_offset = self.preprocess(sent)
        words = [word_offset[0] for word_offset in words_with_offset]
        results = [self.tokenize_word(word, vocab) for word in words]
        
        return sum(results, [])

if __name__ == "__main__":
    corpus = [
        "This is the Hugging Face Course.",
        "This chapter is about tokenization.",
        "This section shows several tokenizer algorithms.",
        "Hopefully, you will be able to understand how they are trained and generate tokens.",
    ]
    # tokenizer = BPETokenizer()
    # vocab, merge_dict = tokenizer.build(corpus, 50)
    # res = tokenizer.tokenize("This is not a token.", merge_dict)
    # print(res)
    
    tokenizer = WordPieceTokenizer()
    vocab = tokenizer.build(corpus, 70)
    # print(vocab)
    
    
    # gold = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '##a', '##b', '##c', '##d', '##e', '##f', '##g', '##h', '##i', '##k',
    # '##l', '##m', '##n', '##o', '##p', '##r', '##s', '##t', '##u', '##v', '##w', '##y', '##z', ',', '.', 'C', 'F', 'H',
    # 'T', 'a', 'b', 'c', 'g', 'h', 'i', 's', 't', 'u', 'w', 'y', 'ab', '##fu', 'Fa', 'Fac', '##ct', '##ful', '##full', '##fully',
    # 'Th', 'ch', '##hm', 'cha', 'chap', 'chapt', '##thm', 'Hu', 'Hug', 'Hugg', 'sh', 'th', 'is', '##thms', '##za', '##zat',
    # '##ut']
    
    # print(len(gold))
    # print(len(vocab))
    # not_in = []
    # for v in vocab:
    #     if v not in gold:
    #         not_in.append(v)
    # print(not_in)
    
    # not_in = []
    # for v in gold:
    #     if v not in vocab:
    #         not_in.append(v)
    # print(not_in)
    
    # def encode_word(word):
    #     tokens = []
    #     while len(word) > 0:
    #         i = len(word)
    #         while i > 0 and word[:i] not in vocab:
    #             i -= 1
    #         if i == 0:
    #             return ["[UNK]"]
    #         tokens.append(word[:i])
    #         word = word[i:]
    #         if len(word) > 0:
    #             word = f"##{word}"
    #     return tokens
    # print(encode_word("Hugging"))
    # print(encode_word("HOgging"))
    
    print(tokenizer.tokenize("This is the Hugging Face course!", vocab))
    
    # ['##a', '##b', '##c', '##d', '##e', '##f', '##g', '##h', '##i', '##k', '##l', '##m', '##n', '##o', '##p', '##r', '##s',
    # '##t', '##u', '##v', '##w', '##y', '##z', ',', '.', 'C', 'F', 'H', 'T', 'a', 'b', 'c', 'g', 'h', 'i', 's', 't', 'u',
    # 'w', 'y']
    
    # ['Th', '##i', '##s', 'is', 'th', '##e', 'Hugg', '##i', '##n', '##g', 'Fac', '##e', 'c', '##o', '##u', '##r', '##s',
    # '##e', '[UNK]']