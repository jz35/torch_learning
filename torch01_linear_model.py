import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 训练数据
x_data = [1.0, 2.0, 3.0]
y_data = [3.0, 5.0, 7.0]

# 预测模型
def forward(x):
    return x * w + b 

# 均方误差损失
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

'''存储历史记录'''
w_list = [] 
b_list = []
mse_list = []

# 遍历w和b
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(0.0, 2.0, 0.1):
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
        mse = l_sum / len(x_data)
        w_list.append(w)
        b_list.append(b)
        mse_list.append(mse)

# 绘制3D曲面图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(w_list, b_list, mse_list, c='r', marker='o')

ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('MSE Loss')

plt.show()
