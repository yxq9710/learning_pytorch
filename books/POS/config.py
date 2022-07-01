# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 18:26
# @Author  : yxq
# @File    : config.py

class Config:
    def __init__(self):
        self.embedding_dim = 128
        self.num_class = 2
        self.batch_size = 32
        self.num_epoch = 5
        self.learning_rate = 0.001

        # MLP
        self.hidden_dim = 256

        # CNN
        self.num_filter = 100
        self.filter_size = 3

        # Transformer
        self.deep_forward = 512
        self.num_head = 2
        self.num_layer = 2
        self.dropout = 0.1
        self.max_len = 512
        self.activation: str = 'relu'
        self.hidden_size = 128
