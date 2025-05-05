import torch

x_data = torch.randn(10, 8)
y_data = torch.randint(0, 2, (10, 1)).float()

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
    
model = Model()

critersion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = critersion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

import matplotlib.pyplot as plt

# 保存每个epoch的loss
loss_list = []

for epoch in range(1000):
    y_pred = model(x_data)
    loss = critersion(y_pred, y_data)
    loss_list.append(loss.item())  # 保存loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# 训练完之后画图
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.show()
