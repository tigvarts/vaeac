import torch
from torch import nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super(type(self), self).__init__()
        self.dims = args

    def forward(self, input):
        return input.view(input.size(0), *self.dims)

class MemData(nn.Module):
    def __init__(self):
        super(type(self), self).__init__()

    def forward(self, input):
        self.data = input
        return input

    def get_memory(self):
        return self.data

class AddSkipConnection(nn.Module):
    def __init__(self, memory):
        super(type(self), self).__init__()
        self.memory = memory

    def forward(self, input):
        data = self.memory.get_memory()
        n_data = data.size(0)
        n_input = input.size(0)
        shape = data.shape[1:]
        if n_data == n_input:
            return data + input
        return (input.view(n_data, -1, *shape) + data.view(n_data, 1, *shape)).view(-1, *shape)

class MergeSkipConnection(nn.Module):
    def __init__(self, memory):
        super(type(self), self).__init__()
        self.memory = memory

    def forward(self, input):
        data = self.memory.get_memory()
        n_data = data.size(0)
        n_input = input.size(0)
        l_data = data.size(1)
        l_input = input.size(1)
        K = n_input // n_data
        shape = data.shape[2:]
        if n_data == n_input:
            return torch.cat([input, data], 1)
        return torch.cat(
            [
                input.view(n_data, K, l_input, *shape),
                data.view(n_data, 1, l_data, *shape).repeat(1, K, 1, *([1] * len(shape)))
            ],
            2).view(n_input, l_input + l_data, *shape)

class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, batchnorm=True, bias=False):
        super(type(self), self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, stride=stride, bias=bias)
        if batchnorm:
            self.bn1 = nn.BatchNorm2d(planes)
        else:
            self.bn1 = None
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=bias)
        if batchnorm:
            self.bn2 = nn.BatchNorm2d(planes)
        else:
            self.bn2 = None
        self.stride = stride
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=bias),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = None

    def forward(self, input):
        residual = input

        out = self.conv1(input)
        if self.bn1 is not None:
            out = self.bn1(out)
        out = self.leaky_relu(out)

        out = self.conv2(out)
        if self.bn2 is not None:
            out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(input)

        out += residual
        out = self.leaky_relu(out)

        return out