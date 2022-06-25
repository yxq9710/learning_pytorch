from torch import nn as nn
import torch


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class MyLinear(nn.Module):
    def __init__(self, input, output):
        super(MyLinear, self).__init__()

        # requires_grad = True
        self.w = nn.Parameter(torch.randn(output, input))   # 使用nn.Parameter封装w和b,才能将这些参数放到优化器中
        self.b = nn.Parameter(torch.randn(output))

    def forward(self, x):
        x = x @ self.w.t() + self.b
        return x
