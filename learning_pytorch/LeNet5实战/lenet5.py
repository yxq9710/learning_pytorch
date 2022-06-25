import torch
from torch import nn
from torchsummary import summary


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )
        self.fc_unit = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        # 可以中途测试模型的输出
        # tmp = torch.randn(2, 3, 32, 32)
        # out = self.conv_unit(tmp)
        # print('out : ', out.shape)

    def forward(self, input):
        out = self.conv_unit(input)
        out = out.view(out.size(0), -1)
        logits = self.fc_unit(out)
        return logits


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tmp = torch.randn(2, 3, 32, 32).to(device)
    lenet = LeNet5().to(device)
    out = lenet(tmp)
    print('lenet out : ', out.shape)
    # print(lenet)
    summary(lenet, (3, 32, 32), batch_size=16)


if __name__ == '__main__':
    main()
