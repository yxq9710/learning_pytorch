# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 23:04
# @Author  : yxq
# @File    : utils.py

import torch
from torch.utils.data import DataLoader
from vocab import Vocab


def get_loader(dataset, batch_size, shuffle=False):
    data_loader = DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=dataset.collate_fn)
    return data_loader


def save_pretrained(vocab, embedding, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"{embedding.shape[0]}  {embedding.shape[1]}\n")
        for index, token in enumerate(vocab.id_to_token):
            vec = " ".join([f"{x}" for x in embedding[index]])  # 将tensor转化为字符串列表，然后逐个输出
            f.write(f"{token} {vec}\n")
        f.close()


def load_pretrained(save_path):
    tokens, embeddings = [], []
    with open(save_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        vocab_size, embedding_dim = lines[0].strip('\n').split(' ')
        print(f"词表大小为{vocab_size} 词向量维度为{embedding_dim}\n")
        for line in lines[1:]:
            token, embedding = line[0], list(map(float, line[1:]))  # 使用内置的map函数将字符串映射为float类型，然后转化为list
            tokens.append(token)
            embeddings.append(embedding)
        vocab = Vocab(tokens)
        embeddings = torch.tensor(embeddings, dtype=torch.float)
        return vocab, embeddings


def plot_loss(loss):
    from matplotlib import pyplot as plt
    import numpy as np
    epochs = len(loss)
    x = np.arange(1, epochs + 1)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.grid()
    plt.title("Loss 变化图")
    plt.plot(x, loss)
    plt.show()
    print("Ending! ")
