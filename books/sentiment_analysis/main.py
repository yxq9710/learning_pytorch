# -*- coding: utf-8 -*-
# @Time    : 2022/6/27 18:26
# @Author  : yxq
# @File    : main.py

from config import Config
from data_process import load_sentence_polarity
from Create_Dataset import BowDataset, collect_fn
from MLP import MLP
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch

cfg = Config()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_data, test_data, vocab = load_sentence_polarity()
    train_data, test_data = BowDataset(train_data), BowDataset(test_data)

    from CNN import CNN, collect_fn
    from LSTM import LSTM, collect_fn
    from Transformer import Transformer

    train_data_loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True, collate_fn=collect_fn)
    test_data_loader = DataLoader(test_data, batch_size=cfg.batch_size, shuffle=True, collate_fn=collect_fn)

    # model = MLP(len(vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.num_class).to(device)
    # model = CNN(len(vocab), cfg.embedding_dim, cfg.filter_size, cfg.num_filter, cfg.num_class).to(device)
    # model = LSTM(len(vocab), cfg.embedding_dim, cfg.hidden_dim, cfg.num_class).to(device)
    model = Transformer(len(vocab), cfg.embedding_dim, cfg.hidden_size, cfg.num_class).to(device)
    criterion = torch.nn.NLLLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    model.train()
    for epoch in range(cfg.num_epoch):
        total_loss = 0.
        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch}"):
            # inputs, targets = [x.to(device) for x in batch]
            inputs, targets, lengths = [x.to(device) for x in batch]
            lengths = lengths.to('cpu')
            preds = model(inputs, lengths, device)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch} Loss : {total_loss: .2f}")

    model.eval()
    total_correct = 0
    for batch in tqdm(test_data_loader, desc="Testing"):
        # inputs, targets = [x.to(device) for x in batch]
        inputs, targets, lengths = [x.to(device) for x in batch]
        lengths = lengths.to('cpu')
        preds = model(inputs, lengths, device)
        predictions = preds.argmax(dim=1)
        total_correct += (predictions == targets).sum().item()
        # total_correct += torch.eq(predictions, targets).float().sum().item()
    print(f"Acc: {total_correct / len(test_data) : .2f}")


if __name__ == '__main__':
    main()
