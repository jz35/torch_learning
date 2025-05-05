import numpy as np 
import matplotlib.pyplot as plt
import torch 

x_data = [1, 2, 3]
y_data = [4, 9, 16]

a = torch.Tensor([1.0])
a.requires_grad = True

b = torch.Tensor([1.0])
b.requires_grad = True

c = torch.Tensor([1.0])
c.requires_grad = True

def forward(x):
    return a * x * x + b * x + c 
'''Tensor运算'''

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict before training : ", 4, forward(4).item())
# print(forward(4))
# print(forward(4).item())

for epoch in range(1000):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad : ', x, y, a.grad.item(), b.grad.item(), c.grad.item())
        a.data = a.data - 0.01 * a.grad.data
        b.data = b.data - 0.01 * b.grad.data
        c.data = c.data - 0.01 * c.grad.data

        a.grad.data.zero_()
        b.grad.data.zero_()
        c.grad.data.zero_()

    print("progress : ", epoch, l.item())

print("predict after training : ", 4, forward(4).item())