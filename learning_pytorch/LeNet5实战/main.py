import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import Config
from lenet5 import LeNet5
from resnet19 import ResNet19

cfg = Config()


def main():
    cifar_train = datasets.CIFAR10('../cifar_10/', True,
                                   transform=transforms.Compose([
                                       transforms.Resize((32, 32)),
                                       transforms.ToTensor()
                                   ]), download=True)

    cifar_train = DataLoader(cifar_train, batch_size=cfg.batchsz, shuffle=True)
    cifar_test = datasets.CIFAR10('../cifar_10', False,
                                  transform=transforms.Compose([
                                      transforms.Resize((32, 32)),
                                      transforms.ToTensor()
                                  ]), download=True)
    cifar_test = DataLoader(cifar_test, batch_size=cfg.batchsz, shuffle=True)

    # 测试输出的形状
    # x_train, label_train = next(iter(cifar_train))
    # print('cifar_train! data.shape : {}, label.shape : {}'.format(x_train.shape, label_train.shape))
    # x_test, label_test = next(iter(cifar_test))
    # print('cifar_test! data.shape : {}, label.shape : {}'.format(x_test.shape, label_test.shape))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5().to(device)
    # model = ResNet19().to(device)
    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(cfg.epochs):
        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits, label)

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            for p in model.parameters():
                # print(p.grad.norm())  # 打印训练参数的梯度
                torch.nn.utils.clip_grad_norm(p, 10)  # 最大不能超过10

            optimizer.step()
        print(epoch, loss.item())

        model.eval()
        with torch.no_grad():
            total_num, total_correct = 0, 0
            for x, label in cifar_test:
                x, label = x.to(device), label.to(device)
                logits = model(x)
                predict = logits.argmax(dim=1)
                total_num += x.size(0)
                total_correct += torch.eq(predict, label).float().sum().item()

            print('test acc : {:.2f}'.format(total_correct / total_num))


if __name__ == '__main__':
    main()
