# -*- coding: utf-8 -*-
# @Time    : 2022/7/1 23:10
# @Author  : yxq
# @File    : main.py

from vocab import *
from models import *
from data import *
from utils import *
from config import Config
from torch import optim
import os

cfg = Config()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据准备
    corpus, vocab = load_reuters()
    # datas = NGramDataSet(corpus, vocab, cfg.window_size)
    # data_loader = get_loader(datas, cfg.batch_size, shuffle=True)
    datas = RNNDataSet(corpus, vocab)
    data_loader = get_loader(datas, 16, shuffle=True)

    # 模型配置
    # model = NGramLM(len(vocab), cfg.embedding_dim, cfg.hidden_size, cfg.window_size).to(device)
    model = RNNLM(len(vocab), 32, 64).to(device)
    # criterion = nn.NLLLoss().to(device)
    criterion = nn.NLLLoss(ignore_index=datas.pad).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    # 模型训练
    model.train()
    losses = []
    for epoch in range(cfg.num_epoch):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch}"):
            inputs, targets = [x.to(device) for x in batch]
            preds = model(inputs)
            # loss = criterion(preds, targets)
            loss = criterion(preds.view(-1, preds.shape[-1]), targets.view(-1))  # RNN时，需要将target变为1维， preds变为2维

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Total loss: {total_loss: .2f}")
        losses.append(total_loss)
    # save_pretrained(vocab, model.embedding.weight.data, os.path.join(cfg.pretrained_embedding_path, 'embeddings.txt'))
    save_pretrained(vocab, model.embedding.weight.data, os.path.join(cfg.pretrained_embedding_path, 'rnn_embeddings.txt'))
    plot_loss(losses)


if __name__ == '__main__':
    main()
