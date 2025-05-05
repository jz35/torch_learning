import torch
import numpy as np 
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
w = 1.0

def forward(x):
    return x * w 

def cost(xs, ys):
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost / len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

print("predict before training : ", 4, forward(4))

# 用来保存每一轮的损失
cost_list = []

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val
    cost_list.append(cost_val)   # 保存当前损失
    print(f"epoch: {epoch} | w: {w:.4f} | loss: {cost_val:.4f}")

print("predict after training : ", 4, forward(4))

# 绘制损失曲线
def draw():
    plt.plot(range(len(cost_list)), cost_list)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.show()

draw()
