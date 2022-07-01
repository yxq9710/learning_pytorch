# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 18:03
# @Author  : yxq
# @File    : Create_Dataset.py

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class BowDataset(Dataset):
    def __init__(self, data):
        super(BowDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


def collect_fn(examples):
    # 从数据集中构建各样本的输入输出
    inputs = [torch.tensor(ex[0]) for ex in examples]
    targets = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    # offsets = [0] + [i.shape[0] for i in inputs]
    # offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)  # 每个序列起始位置的偏移量
    inputs = pad_sequence(inputs, batch_first=True)
    return inputs, targets
