# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 22:55
# @Author  : yxq
# @File    : vocab.py

from collections import defaultdict


class Vocab:
    def __init__(self, tokens=None):
        self.id_to_token = list()
        self.token_to_id = dict()

        for token in tokens:
            if '<unk>' not in tokens:
                tokens = ['<unk>'] + tokens  # list 间的相加
            if type(token) == list:
                for t in token:
                    self.id_to_token.append(t)
                    self.token_to_id[t] = len(self.id_to_token) - 1
            else:
                self.id_to_token.append(token)
                self.token_to_id[token] = len(self.id_to_token) - 1
            self.unk = self.token_to_id['<unk>']

    def __len__(self):
        return len(self.id_to_token)

    def __getitem__(self, token):
        return self.token_to_id.get(token, self.unk)

    @classmethod
    def build(cls, text, min_freq=1, reversed_token=None):
        token_freqs = defaultdict(int)
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_token = ['<unk>'] + (reversed_token if reversed_token else [])
        uniq_token += [token for token, freq in token_freqs.items() if token != '<unk>' and freq >= min_freq]
        return cls(uniq_token)

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token[id] for id in ids]

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]


def save_vocab(vocab, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(vocab.id_to_token))
        f.close()


def read_vocab(path):
    # f.read() 返回一个字符串， f.readlines() 返回一个逐行分开的列表  f.readline() 返回第一行的字符串
    with open(path, 'r', encoding='utf-8') as f:
        tokens = f.read().split('\n')
        f.close()
    return Vocab(tokens)
