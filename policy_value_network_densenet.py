import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *


class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.bias


class DenseLayer(nn.Module):
    def __init__(self, channels, growth_rate):
        super(DenseLayer, self).__init__()

        # 3x3の畳み込みカーネル及び5x5の畳み込みカーネルのチャンネル数を計算する
        self.in_channels = channels
        self.in_kernel3 = int(0.8 * channels)
        self.in_kernel5 = self.in_channels - self.in_kernel3
        self.growth_rate_kernel3 = int(0.8 * growth_rate)
        self.growth_rate_kernel5 = growth_rate - self.growth_rate_kernel3
        self.growth_rate4x = 4 * growth_rate
        self.growth_rate4x_kernel3 = 4 * self.growth_rate_kernel3
        self.growth_rate4x_kernel5 = self.growth_rate4x - self.growth_rate4x_kernel3

        # 畳み込みカーネルの作成
        self.norm1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1_3x3 = nn.Conv2d(
            self.in_kernel3, self.growth_rate4x_kernel3, kernel_size=3, padding=1, bias=False)
        self.conv1_5x5 = nn.Conv2d(
            self.in_kernel5, self.growth_rate4x_kernel5, kernel_size=5, padding=2, bias=False)
        self.norm2 = nn.BatchNorm2d(growth_rate*4)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2_3x3 = nn.Conv2d(
            self.growth_rate4x_kernel3, self.growth_rate_kernel3, kernel_size=3, padding=1, bias=False)
        self.conv2_5x5 = nn.Conv2d(
            self.growth_rate4x_kernel5, self.growth_rate_kernel5, kernel_size=5, padding=2, bias=False)

    def forward(self, x):
        out = torch.cat(x, 1)
        out = self.norm1(out)
        out = self.relu1(out)
        out_3x3 = self.conv1_3x3(out[:, :self.in_kernel3])
        out_5x5 = self.conv1_5x5(out[:, self.in_kernel3:self.in_channels])
        out = torch.cat([out_3x3, out_5x5], 1)

        out = self.norm2(out)
        out = self.relu2(out)
        out_3x3 = self.conv2_3x3(out[:, :self.growth_rate4x_kernel3])
        out_5x5 = self.conv2_5x5(
            out[:, self.growth_rate4x_kernel3:self.growth_rate4x])
        out = torch.cat([out_3x3, out_5x5], 1)
        return out


class DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, channels, growth_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = DenseLayer(channels=channels+i*growth_rate,
                               growth_rate=growth_rate)
            self.add_module(f"denselayer{i+1}", layer)

    def forward(self, x0):
        x = [x0]
        for name, layer in self.items():
            out = layer(x)
            x.append(out)
        return torch.cat(x, 1)


class TransitionLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.add_module("norm", nn.BatchNorm2d(in_channels))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False))


class PolicyValueNetwork(nn.Module):
    def __init__(self, growth_rate=32, blocks=(10,), channels=192, fcl=256):
        super(PolicyValueNetwork, self).__init__()
        self.conv1_1_1 = nn.Conv2d(
            in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=1, padding=0, bias=False)
        self.conv1_1_2 = nn.Conv2d(
            in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.conv1_1_3 = nn.Conv2d(
            in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=5, padding=2, bias=False)
        self.conv1_2 = nn.Conv2d(
            in_channels=FEATURES2_NUM, out_channels=channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)

        # Dense Block及びTransition Layerを作成
        self.blocks = nn.Sequential()
        for i, num_layers in enumerate(blocks):
            block = DenseBlock(
                num_layers=num_layers,
                channels=channels,
                growth_rate=growth_rate
            )
            self.blocks.add_module(f"denseblock{i+1}", block)

            channels = channels+num_layers*growth_rate
            if i != len(blocks)-1:
                # 最後のDense Block出ない場合はTransition Layerを追加
                trans = TransitionLayer(
                    in_channels=channels, out_channels=channels//2)
                self.blocks.add_module(f"transition{i+1}", trans)
                channels //= 2

        # policy head
        self.policy_conv = nn.Conv2d(
            in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = Bias(9*9*MAX_MOVE_LABEL_NUM)

        # value head
        self.value_conv1 = nn.Conv2d(
            in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.value_norm1 = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x1, x2):
        x_1_1 = self.conv1_1_1(x1)
        x_1_2 = self.conv1_1_2(x1)
        x_1_3 = self.conv1_1_3(x1)
        x_2 = self.conv1_2(x2)
        x = F.relu(self.norm1(x_1_1 + x_1_2 + x_1_3 + x_2))

        # dense blocks
        x = self.blocks(x)

        # policy head
        policy = self.policy_conv(x)
        policy = self.policy_bias(torch.flatten(policy, 1))

        # value head
        value = F.relu(self.value_norm1(self.value_conv1(x)))
        value = F.relu(self.value_fc1(torch.flatten(value, 1)))
        value = self.value_fc2(value)

        return policy, value
