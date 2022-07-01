# -*- coding: utf-8 -*-
# @Time    : 2022/6/26 22:30
# @Author  : yxq
# @File    : 使用梯度下降解决异或问题(线性不可分问题).py

import torch
from torch import nn, optim
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden):
        super(MLP, self).__init__()
        self.ln1 = nn.Linear(input_dim, hidden)
        self.ln2 = nn.Linear(hidden, output_dim)

    def forward(self, x):
        out = self.ln1(x)
        out = F.relu(out)
        out = self.ln2(out)
        logits = F.log_softmax(out)  # 防止直接softmax发生数值溢出
        return logits


def main():
    # NLLLoss + log_softmax 等价于交叉熵损失函数

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_train = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]).to(device)
    y_train = torch.tensor([0, 1, 1, 0]).to(device)
    model = MLP(input_dim=2, output_dim=2, hidden=5).to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(1000):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss. backward()
        optimizer.step()

    print("Parameters: ")
    for name, param in model.named_parameters():
        print(name, param)

    y_pred = model(x_train)
    print("Predicted results: ", y_pred.argmax(dim=1))


if __name__ == '__main__':
    main()
