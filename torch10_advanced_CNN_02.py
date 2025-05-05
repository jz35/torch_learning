import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

'''
1. 理论《深度学习》
2. 阅读pytorch文档
3. 复现经典工作（读代码） 写代码，自己去写
4. 扩充视野
'''

# 设置批处理大小
batch_size = 64

# 定义数据预处理的转换
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.1307, ), (0.3081, ))  # 归一化
])

# 下载并加载训练数据集
train_dataset = datasets.MNIST(root='../dataset/mnist',
                               train=True,
                               download=True,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,  # 打乱数据
                          batch_size=batch_size)

# 下载并加载测试数据集
test_dataset = datasets.MNIST(root='../dataset/mnist',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,  # 不打乱数据
                         batch_size=batch_size)

class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)
    
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = torch.nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)

        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)

        x = x.view(in_size, -1)
        x = self.fc(x)

        return x
    
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 随机梯度下降

def train(epoch):
    running_loss = 0.0
    # 遍历训练数据
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data  # 获取输入和目标
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()  # 清零梯度

        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, target)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        running_loss += loss.item()  # 累加损失
        if batch_idx % 300 == 299:  # 每300个batch打印一次损失
            print('[%d, %5d] loss : %.3f' % (epoch + 1, batch_idx + 1,
                                             running_loss / 300))
            running_loss = 0.0  # 重置损失

def test():
    correct = 0
    total = 0
    with torch.no_grad():  # 不计算梯度
        # 遍历测试数据
        for data in test_loader:
            images, labels = data  # 获取图像和标签
            inputs, target = images.to(device), labels.to(device)  # 将图像和标签移动到设备上
            outputs = model(inputs)  # 前向传播
            _, predicted = torch.max(outputs.data, dim=1)  # 获取预测结果
            total += target.size(0)  # 累加总数
            correct += (predicted == target).sum().item()  # 计算正确预测的数量
        print('Accuracy on test set : %d %%' % (100 * correct / total))  # 打印准确率

if __name__ == '__main__':
    for epoch in range(10):  # 训练10个epoch
        train(epoch)  # 训练
        test()  # 测试