# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 13:54
# @Author  : yxq
# @File    : Tree_DataSet.py

from torch.utils.data import Dataset


class Tree_DataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
