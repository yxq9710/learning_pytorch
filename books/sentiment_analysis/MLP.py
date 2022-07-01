# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 18:12
# @Author  : yxq
# @File    : MLP.py

import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_class):
        super(MLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, hidden_size)
        self.activation = F.relu
        self.linear2 = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        out = self.embedding(x)
        out = out.mean(dim=1)
        out = self.linear1(out)
        out = self.activation(out)
        out = self.linear2(out)
        probs = F.log_softmax(out, dim=1)
        return probs


def main():
    inputs = torch.tensor([[0, 2, 1, 3], [2, 0, 2, 1]])
    inputs = inputs.to(torch.int64)
    mlp = MLP(4, 100, 50, 2)
    y = mlp(inputs)
    print(y)


if __name__ == '__main__':
    main()
