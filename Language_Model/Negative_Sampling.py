# -*- coding: utf-8 -*-
# @Time    : 2022/7/2 19:29
# @Author  : yxq
# @File    : Negative_Sampling.py

import torch
from torch import nn, optim
from config import Config
from data import load_reuters
from utils import get_loader, save_pretrained, plot_loss
from tqdm.auto import tqdm
from torch.nn import functional as F
import os
from torch.utils.data import Dataset

cfg = Config()


class Skip_Gram_NSDataSet(Dataset):
    def __init__(self, corpus, vocab, window_size=2, n_negatives=5, ns_dist=None):
        self.data = []
        self.bos = vocab[cfg.BOS_TOKEN]  # 一定要加上vocab，将字符串转化为int
        self.eos = vocab[cfg.EOS_TOKEN]
        self.pad = vocab[cfg.PAD_TOKEN]
        self.n_negatives = n_negatives
        # if ns_dist is None:
        #     self.ns_dist = torch.ones(len(vocab))
        # else:
        #     self.ns_dist = ns_dist
        self.ns_dist = ns_dist if ns_dist is not None else torch.ones(len(vocab))
        for sentence in tqdm(corpus, desc='Dataset Construction'):
            sentence = [self.bos] + sentence + [self.eos]
            for i in range(1, len(sentence) - 1):
                input = sentence[i]
                contexts = sentence[max(0, i-window_size): i] + sentence[i+1: min(len(sentence), i+1+window_size)]
                contexts += [self.pad] * (2 * window_size - len(contexts))
                self.data.append((input, contexts))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def collate_fn(self, examples):
        words = torch.tensor([ex[0] for ex in examples], dtype=torch.long)
        contexts = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
        batch_size, window_size = contexts.shape
        neg_contexts = []
        for i in range(batch_size):
            ns_dist = self.ns_dist.index_fill(0, contexts[i], .0)  # 将第0维的索引为contexts[i]的数据修改为0.
            # 从ns_dist中取self.n_negatives * window_size 个样本， 且是有放回的取， 返回ns_dist的行下标，ns_dist可看做每一行被取到的概率
            neg_contexts.append(torch.multinomial(ns_dist, self.n_negatives * window_size, replacement=True))
        neg_contexts = torch.stack(neg_contexts, dim=0)
        return words, contexts, neg_contexts


class Skip_Gram_NSLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Skip_Gram_NSLM, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward_w(self, words):
        return self.word_embedding(words)

    def forward_c(self, contexts):
        return self.context_embedding(contexts)


def get_unigram_distribution(corpus, vocab_size):
    token_counts = torch.tensor([0] * vocab_size)
    total_count = 0
    for sentence in corpus:
        total_count += len(sentence)
        for token in sentence:
            token_counts[token] += 1
    unigram_dist = torch.div(token_counts.float(), total_count)
    return unigram_dist


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    corpus, vocab = load_reuters()
    unigram_dist = get_unigram_distribution(corpus, len(vocab))
    negative_sampling_dist = unigram_dist ** 0.75
    negative_sampling_dist /= negative_sampling_dist.sum()

    datas = Skip_Gram_NSDataSet(corpus, vocab, cfg.window_size, cfg.n_negatives, negative_sampling_dist)
    data_loader = get_loader(datas, cfg.batch_size, shuffle=True)
    model = Skip_Gram_NSLM(len(vocab), cfg.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    model.train()
    for epoch in range(cfg.num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f'Training Epoch {epoch}'):
            words, contexts, neg_contexts = [x.to(device) for x in batch]

            optimizer.zero_grad()

            batch_size = words.shape[0]
            word_embedding = model.forward_w(words).unsqueeze(dim=2)
            context_embedding = model.forward_c(contexts)
            neg_contexts_embedding = model.forward_c(neg_contexts)

            # c=torch.bmm(a, b) ==> a.size()=(b, h1, w), b.size()=(b, w, h2)  ==>  c.size()=(b, h1, h2)
            context_loss = F.logsigmoid(torch.bmm(context_embedding, word_embedding).squeeze(dim=2))
            context_loss = context_loss.mean(dim=1)

            neg_context_loss = F.logsigmoid(torch.bmm(neg_contexts_embedding, word_embedding).squeeze(dim=2).neg())
            neg_context_loss = neg_context_loss.view(batch_size, -1, cfg.n_negatives).sum(dim=2)
            neg_context_loss = neg_context_loss.mean(dim=1)

            loss = -(context_loss + neg_context_loss).mean()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Loss: {total_loss :.2f}")
    combined_embeddings = model.word_embedding.weight + model.context_embedding.weight
    save_pretrained(vocab, combined_embeddings.data, os.path.join(cfg.pretrained_embedding_path, 'Skip_Gram_NS_embeddings.txt'))


if __name__ == '__main__':
    main()
