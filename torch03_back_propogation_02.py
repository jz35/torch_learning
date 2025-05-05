import torch
import matplotlib.pyplot as plt

# 数据
x_data = torch.tensor([1.0, 2.0, 3.0])
y_data = torch.tensor([4.0, 9.0, 16.0])

# 参数，开启自动求导
a = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)
c = torch.tensor([1.0], requires_grad=True)

# 定义模型
def model(x):
    return a * x**2 + b * x + c

# 优化器
optimizer = torch.optim.SGD([a, b, c], lr=0.01)

# 用于保存每次的loss
loss_list = []

# 训练
for epoch in range(1000):
    y_pred = model(x_data)
    loss = torch.mean((y_pred - y_data) ** 2)

    optimizer.zero_grad()  # 梯度清零
    loss.backward()        # 自动计算梯度
    optimizer.step()       # 更新参数

    loss_list.append(loss.item())

# 绘制损失曲线
plt.figure(figsize=(8, 4))
plt.plot(loss_list, label='Loss over epochs', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)
plt.show()

# 绘制拟合曲线
a_val, b_val, c_val = a.item(), b.item(), c.item()
x_plot = torch.linspace(0, 5, 100)
y_plot = a_val * x_plot**2 + b_val * x_plot + c_val

plt.figure(figsize=(8, 5))
plt.scatter(x_data.numpy(), y_data.numpy(), color='red', label='Data Points')
plt.plot(x_plot.numpy(), y_plot.detach().numpy(), color='green',
         label=f'Fitted Curve: {a_val:.2f}x² + {b_val:.2f}x + {c_val:.2f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Quadratic Regression Fit')
plt.legend()
plt.grid(True)
plt.show()
