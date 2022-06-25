import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
import torchvision


def one_hot(label, depth=10):
    out = torch.zeros(label.size(0), depth)
    idx = torch.LongTensor(label).view(-1, 1)
    out.scatter_(dim=1, index=idx, value=1)
    return out


batch_size = 512
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])), batch_size=batch_size, shuffle=False
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,)
                                   )
                               ])), batch_size=batch_size, shuffle=False
)

x, y = next(iter(train_loader))
print(x.shape, y.shape, x.min(), y.max())


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        # x: [b, 1, 28, 28]
        # h1 = relu(wx+b)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('---- device: {}'.format(device))
print('---- pytorch version: {}'.format(torch.__version__))

net = Net().to(device)
# [w1, b1, w2, b2, w3, b3]
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
mse_loss = nn.MSELoss().to(device)

train_loss = []

# 添加 L1-regularization
regularization_loss = 0
for param in net.parameters():
    regularization_loss += torch.sum(torch.abs(param))

for epoch in range(3):
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.view(x.size(0), 28 * 28).to(device)  # [b, 1, 28, 28] => [b, 784]
        out = net(x)
        y_onehot = one_hot(y).to(device)
        # loss = F.mse_loss(out, y_onehot)  # loss = mse(out, y_onehot)

        loss = mse_loss(out, y_onehot)
        # 添加 L1-regularization
        # loss = mse_loss(out, y_onehot) + 0.01 * regularization_loss

        optimizer.zero_grad()  # 梯度清零
        loss.backward()
        optimizer.step()  # w' = w - lr * grad 梯度更新过程

        train_loss.append(loss.item())

        if batch_idx % 10 == 0:
            print(epoch, batch_idx, loss.item())

total_correct = 0
for x, y in test_loader:
    x_ = x.view(x.size(0), 28 * 28).to(device)
    out = net(x_)  # [b, 10]

    pred = out.argmax(dim=1)
    correct = pred.eq(y.to(device)).sum().float().item()
    total_correct += correct
total_num = len(test_loader.dataset)
acc = total_correct / total_num
print("acc : ", acc)
