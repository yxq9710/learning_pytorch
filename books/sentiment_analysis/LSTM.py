# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 10:31
# @Author  : yxq
# @File    : LSTM.py

from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch
from torch import nn
from torch.nn import functional as F


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_class):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_class)

    def forward(self, x, lengths=None):
        embedding = self.embedding(x)
        # 通过pack_padded_sequence可以消除padding 0带来的影响
        if lengths is not None:
            embedding = pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(embedding)
        out = self.linear(hn[-1])
        probs = F.log_softmax(out, dim=-1)
        return probs


def collect_fn(examples):
    train_data = [torch.tensor(ex[0]) for ex in examples]
    test_data = torch.tensor([ex[1] for ex in examples], dtype=torch.long)  # type()是函数，返回的是数据结构类型， dtype返回的是元素的数据类型
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    train_data = pad_sequence(train_data, batch_first=True)
    return train_data, test_data, lengths


def main():
    inputs = torch.tensor([[0, 2, 1, 3], [2, 0, 2, 1]])
    inputs = inputs.to(torch.int64)
    lstm = LSTM(4, 100, 50, 2)
    print('yes')
    y = lstm(inputs)
    print(y)


if __name__ == '__main__':
    main()
