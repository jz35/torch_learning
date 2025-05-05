import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class DiabetesDataset(Dataset):
    def __init__(self):
        # 假设数据集有1000个样本，每个样本8个特征
        self.x_data = torch.randn(1000, 8)  # 输入特征，1000行，8列
        self.y_data = torch.randint(0, 2, (1000, 1)).float()  # 输出标签，1000个标签（0或1）

    def __getitem__(self, index):
        # 返回一个数据样本和它的标签
        x = self.x_data[index]
        y = self.y_data[index]
        return x, y

    def __len__(self):
        # 返回数据集的大小
        return len(self.x_data)

# 创建数据集对象
dataset = DiabetesDataset()

# 创建数据加载器
train_loader = DataLoader(dataset=dataset, 
                          batch_size=32,  # 每次加载32个样本
                          shuffle=True,   # 每个epoch打乱数据
                          num_workers=2)  # 使用2个子进程加载数据

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

# 实例化模型
model = Model()

# 定义损失函数：二元交叉熵损失
criterion = torch.nn.BCELoss()

# 定义优化器：SGD优化器，学习率0.01
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

if __name__ == '__main__':
    for epoch in range(100):  # 100个epoch
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # 获取数据和标签
            inputs, labels = data
            
            # 前向传播：计算模型的输出
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 清空梯度
            optimizer.zero_grad()
            
            # 反向传播：计算梯度
            loss.backward()
            
            # 优化器更新参数
            optimizer.step()
            
            # 打印每10个batch的进度
            running_loss += loss.item()
            if i % 10 == 9:  # 每10个batch输出一次
                print(f"Epoch [{epoch+1}/100], Step [{i+1}/{len(train_loader)}], Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        # 每个epoch后可以保存模型
        # torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")

    print("Training finished!")
