import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

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

# 定义神经网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义网络层
        self.l1 = torch.nn.Linear(784, 512)  # 输入层到隐藏层1
        self.l2 = torch.nn.Linear(512, 256)  # 隐藏层1到隐藏层2
        self.l3 = torch.nn.Linear(256, 128)  # 隐藏层2到隐藏层3
        self.l4 = torch.nn.Linear(128, 64)   # 隐藏层3到隐藏层4
        self.l5 = torch.nn.Linear(64, 10)    # 隐藏层4到输出层

    def forward(self, x):
        # 前向传播
        x = x.view(-1, 784)  # 将输入展平
        x = F.relu(self.l1(x))  # 激活函数ReLU
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  # 输出层

# 实例化模型
model = Net()

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 随机梯度下降

def train(epoch):
    running_loss = 0.0
    # 遍历训练数据
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data  # 获取输入和目标
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
            outputs = model(images)  # 前向传播
            _, predicted = torch.max(outputs.data, dim=1)  # 获取预测结果
            total += labels.size(0)  # 累加总数
            correct += (predicted == labels).sum().item()  # 计算正确预测的数量
        print('Accuracy on test set : %d %%' % (100 * correct / total))  # 打印准确率

if __name__ == '__main__':
    for epoch in range(10):  # 训练10个epoch
        train(epoch)  # 训练
        test()  # 测试