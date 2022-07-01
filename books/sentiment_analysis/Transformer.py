# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 11:04
# @Author  : yxq
# @File    : Transformer.py

import torch, math
from torch import nn
from torch.nn import functional as F
from config import Config

cfg = Config()


class Transformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_class,
                 deep_forward=cfg.deep_forward, num_head=cfg.num_head, num_layer=cfg.num_layer,
                 dropout=cfg.dropout, max_len=cfg.max_len, activation=cfg.activation):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEncoding(embedding_dim, dropout, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(hidden_size, num_head, deep_forward, dropout, activation)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layer)
        self.output = nn.Linear(hidden_size, num_class)

    def forward(self, inputs, lengths, device=None):
        inputs = torch.transpose(inputs, 0, 1)
        embedding = self.embedding(inputs)
        hidden_states = self.position_embedding(embedding)
        attention_mask = length_to_mask(lengths, device) == False
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        hidden_states = hidden_states[0, :, :]
        output = self.output(hidden_states)
        probs = F.log_softmax(output, dim=1)
        return probs


def length_to_mask(lengths, device=None):
    max_len = torch.max(lengths)
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len) < lengths.unsqueeze(1)
    if device is not None:
        mask = mask.to(device)
    return mask


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


def main():
    lengths = torch.tensor([3, 5, 4])
    print(length_to_mask(lengths))


if __name__ == '__main__':
    main()
