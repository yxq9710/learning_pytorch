import torch

# 直接用torch创建tensor
# a = torch.randn(2, 3)
# print(a)
# print(a.type)  # 返回的是基本数据类型， 不如使用type(a)
# print(type(a))
# print(isinstance(a, torch.FloatTensor))
#
# # a = a.cuda()
# # print(isinstance(a, torch.cuda.FloatTensor))
# print(a.shape)
# print(a.size(0))
# print(a.shape[1])
# print(list(a.shape))
# print(len(a.shape))
# print(a.dim())

# # 创建tensor
# a = torch.tensor([1, 2])
# print(a)
# b = torch.FloatTensor(1, 2)
# print(b)

# 从 numpy 转化
# import numpy as np
# a = np.random.rand(1, 2)
# print(a)

# a = torch.rand([4, 1, 28, 28])
# print(a.shape)
# b = a.view(4, 28*28)
# print(b.size())

# a = torch.rand(4, 32, 1, 1)
# b = a.expand(4, 32, 2, 2)
# print(a.shape)
# print(b.shape)


# 乘法
a = torch.rand(3, 2)
b = torch.rand(2, 3)
c = a @ b
print(c.shape)
print(torch.matmul(a, b).shape)  # 只有最后两维进行相乘
