# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 23:11
# @Author  : yxq
# @File    : config.py

import os


class Config:
    def __init__(self):
        # common setting
        self.batch_size = 1024
        self.embedding_dim = 128
        self.hidden_size = 256
        self.num_epoch = 10
        self.learning_rate = 1e-3

        # N_Gram LM setting
        self.window_size = 3

        # path
        self.vocab_path = './vocab'
        self.pretrained_embedding_path = './pretrained_embedding'

        for path in [self.vocab_path, self.pretrained_embedding_path]:
            if not os.path.exists(path):
                os.mkdir(path)

        # common Token
        self.BOS_TOKEN = '<bos>'
        self.EOS_TOKEN = '<eos>'
        self.PAD_TOKEN = '<pad>'

        # vocab min_freq
        self.min_freq = 2
