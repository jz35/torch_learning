import torch

in_channels, out_channels = 5, 10
width, height = 100, 100
kernel_size = 3
batch_size = 1

input = torch.randn(batch_size,
                    in_channels,
                    width,
                    height)
''' m * n * w' * h' '''

conv_layer = torch.nn.Conv2d(in_channels,
                             out_channels,
                             kernel_size=kernel_size)
'''输入通道数量、输出通道数量、卷积核大小'''

output = conv_layer(input)

print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)

input = [3, 4, 6, 5, 7,
         2, 4, 6, 8, 2,
         1, 6, 7, 8, 4,
         9, 7, 4, 6, 2,
         3, 7, 5, 4, 1]
input = torch.Tensor(input).view(1, 1, 5, 5)

conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3,
                             padding=1, bias=False)
'''bias : 偏置量'''

kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
'''view : 输入通道数， 输出通道数， 宽度， 高度'''
conv_layer.weight.data = kernel.data

output = conv_layer(input)
print(output)

input = [3, 4, 6, 5,
         2, 4, 6, 8,
         1, 6, 7, 8,
         9, 7, 4, 6]
input = torch.Tensor(input).view(1, 1, 4, 4)

maxpolling_layer = torch.nn.MaxPool2d(kernel_size=2)

output = maxpolling_layer(input)
print(output)