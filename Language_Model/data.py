# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 22:15
# @Author  : yxq
# @File    : data.py

import torch
from torch.utils.data import Dataset
from vocab import Vocab
from config import Config
from tqdm.auto import tqdm
from torch.nn.utils.rnn import pad_sequence

cfg = Config()


def load_reuters():
    from nltk.corpus import reuters
    # 原始数据获取
    text = reuters.sents()
    text = [[word.lower() for word in sentence] for sentence in text]

    # 构建词表和数据集
    vocab = Vocab.build(text, min_freq=cfg.min_freq, reversed_token=[cfg.BOS_TOKEN, cfg.EOS_TOKEN, cfg.PAD_TOKEN])
    corpus = [vocab.convert_tokens_to_ids(sentence) for sentence in text]

    return corpus, vocab


class NGramDataSet(Dataset):
    """
    实现N-Gram预训练语言模型, 其中N为指定的超参数
    """

    def __init__(self, corpus, vocab, n_gram=2):
        self.data = []
        self.bos = vocab[cfg.BOS_TOKEN]
        self.eos = vocab[cfg.EOS_TOKEN]

        for sentence in tqdm(corpus, desc='Dataset Construction'):
            # sentence 是一个 list
            if len(sentence) < n_gram:
                continue
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(len(sentence) - n_gram):
                input = sentence[i: i + n_gram]
                target = sentence[i + n_gram]
                self.data.append((input, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, examples):
        inputs = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        return inputs, targets


class RNNDataSet(Dataset):
    """
    实现循环神经网络预训练模型, 其中N为指定的超参数
    """

    def __init__(self, corpus, vocab):
        self.data = []
        self.bos = vocab[cfg.BOS_TOKEN]
        self.eos = vocab[cfg.EOS_TOKEN]
        self.pad = vocab[cfg.PAD_TOKEN]

        for sentence in tqdm(corpus, desc='Dataset Construction'):
            input = [self.bos] + sentence
            target = sentence + [self.eos]
            self.data.append((input, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, examples):
        inputs = [torch.tensor(ex[0]) for ex in examples]
        targets = [torch.tensor(ex[1]) for ex in examples]

        inputs = pad_sequence(inputs, batch_first=True, padding_value=self.pad)
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad)
        return inputs, targets
