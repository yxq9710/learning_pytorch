import torch
from torch import nn as nn
from torch.nn import functional as F
from torchsummary import summary


class ResBlk(nn.Module):
    """
    中间有三个卷积的ResNet结构
    """
    def __init__(self, ch_in, ch_out):
        super(ResBlk, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out // 4, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(ch_out // 4)
        self.conv2 = nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out // 4)
        self.conv3 = nn.Conv2d(ch_out // 4, ch_out, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_in != ch_out:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.extra(x) + out
        return out


# 要加上设备迁移的这行代码
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
res = ResBlk(64, 256).to(device)

summary(res, (64, 28, 28), batch_size=16)
print(dict(res.named_parameters()).keys())
print(res)

res.train()  # 只是设置一个模式，并不运行模型代码
