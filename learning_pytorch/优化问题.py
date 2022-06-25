import torch
from torch import optim


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


x = torch.tensor([0., 0.], requires_grad=True)
optimizer = optim.Adam([x], lr=1e-3)

for step in range(20000):
    pred = himmelblau(x)

    optimizer.zero_grad()
    pred.backward()
    optimizer.step()

    if step % 2000 == 0:
        print('step {}: x = {}, f(x) = {:.5f}'. format(step, x.tolist(), pred.item()))
