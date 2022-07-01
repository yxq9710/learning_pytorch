# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 17:24
# @Author  : yxq
# @File    : vocab.py

from collections import defaultdict


class Vocab:
    def __init__(self, tokens=None):
        self.token_to_id = dict()
        self.id_to_token = list()

        if tokens is not None:
            if '<unk>' not in tokens:
                tokens = tokens + ['<unk>']
            for token in tokens:
                if type(token) == list:   # 新增判断是否为嵌套list
                    for t in token:
                        self.id_to_token.append(t)
                        self.token_to_id[t] = len(self.id_to_token) - 1
                else:
                    self.id_to_token.append(token)
                    self.token_to_id[token] = len(self.id_to_token) - 1
            self.unk = self.token_to_id['<unk>']

    @classmethod
    def build(cls, text, minfreq=1, reversed_tokens=None):
        token_freqs = defaultdict(int)  # 不存在则默认为0， 类似于java中的getOrDefault()函数
        for sentence in text:
            for token in sentence:
                token_freqs[token] += 1
        uniq_token = ['<unk>'] + (reversed_tokens if reversed_tokens else [])
        uniq_token += [token for token, freq in token_freqs.items() if freq >= minfreq and token != '<unk>']
        return cls(uniq_token)

    def __len__(self):
        return len(self.id_to_token)

    def __getitem__(self, token):
        return self.token_to_id.get(token, self.unk)
        # return self.id_to_token[token]

    def convert_tokens_to_ids(self, tokens):
        return [self[token] for token in tokens]  # self[token]调用的是getitem方法

    def convert_ids_to_tokens(self, ids):
        return [self.id_to_token[id] for id in ids]


def main():
    vocab = Vocab(['test', 'a', 'as', 'dassd', 'v'])
    print(vocab.convert_tokens_to_ids('a'))
    print("a")


if __name__ == '__main__':
    main()
