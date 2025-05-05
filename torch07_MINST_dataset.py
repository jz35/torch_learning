import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim

# 加载训练数据集和测试数据集
train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=32,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=32,
                         shuffle=False)

# 定义简单的神经网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 输入1通道，输出32通道，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 输入32通道，输出64通道，卷积核3x3
        self.fc1 = nn.Linear(7*7*64, 128)  # 全连接层
        self.fc2 = nn.Linear(128, 10)      # 输出10类（MNIST有10个数字）

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)  # 2x2最大池化
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)  # 2x2最大池化
        x = x.view(-1, 7*7*64)  # 展平卷积后的结果，变为一维
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = SimpleCNN()
model = model.to('cuda')

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多分类交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练过程
for epoch in range(10):  # 训练10个epoch
    model.train()  # 设置为训练模式
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # inputs: 输入的图片，targets: 输入的标签
        
        # 将输入和目标转换到相应的设备（如果有GPU的话）
        inputs, targets = inputs.cuda(), targets.cuda()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 清空梯度
        optimizer.zero_grad()
        
        # 反向传播
        loss.backward()
        
        # 优化器更新参数
        optimizer.step()
        
        # 打印每个batch的损失
        if batch_idx % 100 == 0:  # 每100个batch打印一次
            print(f"Epoch [{epoch+1}/10], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# 测试过程
model.eval()  # 设置为评估模式
correct = 0
total = 0
with torch.no_grad():  # 评估时不计算梯度
    for inputs, targets in test_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f'Accuracy of the model on the test dataset: {100 * correct / total:.2f}%')
