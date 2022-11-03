import torch

a = torch.randn(2, 3, 4)

a[0, :1, 2:] = 0
print(a)