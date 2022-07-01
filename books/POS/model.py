# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 13:33
# @Author  : yxq
# @File    : model.py

import torch
from torch import nn
from torch.nn import functional as F
from config import Config
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from data_process import *
import math

cfg = Config()


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_class):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, num_class)

    def forward(self, inputs, lengths=None, device=None):
        embedding = self.embedding(inputs)
        if lengths is not None:
            embedding = pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        hidden, (hn, cn) = self.lstm(embedding)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True)

        out = self.output(hidden)
        probs = F.log_softmax(out, dim=-1)
        return probs


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_class,
                 deep_forward=cfg.deep_forward, num_head=cfg.num_head, num_layer=cfg.num_layer,
                 dropout=cfg.dropout, max_len=cfg.max_len, actication=cfg.activation):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_head, deep_forward, dropout, actication)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layer)
        self.output = nn.Linear(embedding_dim, num_class)

    def forward(self, inputs, lengths, device):
        inputs = torch.transpose(inputs, 0, 1)
        embedding = self.embedding(inputs)
        hidden_states = self.position_embedding(embedding)

        attention_mask = length_to_mask(lengths, device) == False
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask).transpose(0, 1)
        out = self.output(hidden_states)
        probs = F.log_softmax(out, dim=-1)
        return probs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model))
        pe[:, 0:: 2] = torch.sin(position * div_term)
        pe[:, 1:: 2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # 不对位置编码求梯度

    def forward(self, inputs):
        x = inputs + self.pe[:inputs.size(0), :]   # 将word embedding与position embedding进行拼接
        return x
