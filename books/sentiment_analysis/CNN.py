# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 22:03
# @Author  : yxq
# @File    : CNN.py

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filter_size, num_filter, num_class):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1d = nn.Conv1d(embedding_dim, num_filter, filter_size, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, num_filter, 4, padding=2)
        self.conv3 = nn.Conv1d(embedding_dim, num_filter, 5, padding=2)
        self.activate = F.relu
        self.linear = nn.Linear(num_filter * 3, num_class)

    def forward(self, x):
        embedding = self.embedding(x)  # [2, 4, 100]
        pooled = []
        for conv in [self.conv1d, self.conv2, self.conv3]:
            convolution = self.activate(conv(embedding.permute(0, 2, 1)))
            pooling = F.max_pool1d(convolution, kernel_size=convolution.shape[2])
            pooled.append(pooling)
        pooling = torch.cat(pooled, dim=-2)
        outputs = self.linear(pooling.squeeze(dim=2))
        probs = F.log_softmax(outputs, dim=1)
        return probs


def collect_fn(examples):
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, targets



def main():
    inputs = torch.tensor([[0, 2, 1, 3], [2,0,2,1]])
    inputs = inputs.to(torch.int64)
    cnn = CNN(4, 100, 3, 50, 2)
    print('yes')
    y = cnn(inputs)
    print(y)


if __name__ == '__main__':
    main()
