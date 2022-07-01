# -*- coding: utf-8 -*-
# @Time    : 2022/6/28 13:51
# @Author  : yxq
# @File    : main.py

import torch
from torch import nn, optim
from config import Config
from model import *
from vocab import Vocab
from data_process import *
from Tree_DataSet import Tree_DataSet
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


cfg = Config()


def main():
    # 数据准备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, test_data, vocab, tag_vocab = load_treebank()
    train_data, test_data = Tree_DataSet(train_data), Tree_DataSet(test_data)
    train_data_loader = DataLoader(train_data, cfg.batch_size, shuffle=True, collate_fn=collect_fn)
    test_data_loader = DataLoader(test_data, cfg.batch_size, shuffle=True, collate_fn=collect_fn)

    # 模型配置
    # model = LSTM(len(vocab), cfg.embedding_dim, cfg.hidden_size, len(tag_vocab)).to(device)
    model = Transformer(len(vocab), cfg.embedding_dim, cfg.hidden_size, len(tag_vocab)).to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # 模型训练
    model.train()
    for epoch in range(cfg.num_epoch):
        total_loss = 0
        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
            inputs, targets, lengths, mask = [x.to(device) for x in batch]
            lengths = lengths.to('cpu')
            preds = model(inputs, lengths, device)
            loss = criterion(preds[mask], targets[mask])

            # 计算损失 + 梯度更新
            optimizer.zero_grad()
            loss.backward()   # 梯度的反向传播
            optimizer.step()

            total_loss += loss.item()
        print(f"Loss : {total_loss:.2f}")

    model.eval()
    total_correct = 0
    total_num = 0
    for batch in tqdm(test_data_loader, desc=f"Testing"):
        inputs, targets, lengths, mask = [x.to(device) for x in batch]
        lengths = lengths.to('cpu')
        preds = model(inputs, lengths, device)
        prediction = (preds.argmax(dim=-1) == targets)[mask]
        total_correct += prediction.sum().item()
        total_num += mask.sum().item()
    print(f"Acc : {total_correct / total_num :.2f}")


if __name__ == '__main__':
    main()
