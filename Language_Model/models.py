# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 23:02
# @Author  : yxq
# @File    : models.py

import torch
from torch import nn
from torch.nn import functional as F


class NGramLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, window_size):
        super(NGramLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim * window_size, hidden_size)
        self.activation = F.relu
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        flatten = embedding.view(embedding.shape[0], -1)
        hidden_states = self.linear(flatten)
        hidden_states = self.activation(hidden_states)
        out = self.output(hidden_states)
        probs = F.log_softmax(out, dim=1)
        return probs


class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.activation = F.tanh
        self.output = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        hidden, (hn, cn) = self.lstm(embedding)
        # hidden = self.activation(hidden)   # 用了1个batch的结果，包含n个句子，每个句子的每个词都有一个输出，即: 使用了前面的所有词来预测下一个词
        out = self.output(hidden)
        probs = F.log_softmax(out, dim=2)  # 也可以用 dim=-1
        return probs


class CBOW(nn.Module):
    """
    连续词袋模型，通过上下文预测中心词
    """
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        hidden = embedding.mean(dim=1)
        out = self.output(hidden)
        probs = F.log_softmax(out, dim=1)
        return probs


class Skip_Gram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Skip_Gram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embedding = self.embedding(inputs)
        out = self.output(embedding)
        probs = F.log_softmax(out, dim=1)
        return probs
