import torch
import torch.nn as nn
import torch.nn.functional as F

from dlshogi.common import *

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.bias

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.norm1=nn.BatchNorm2d(channels)
        self.relu1=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(channels,channels,kernel_size=3,padding=1,bias=False)
        self.norm2=nn.BatchNorm2d(channels)
        self.relu2=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,padding=1,bias=False)
        self.layer_norm = nn.LayerNorm([channels, 9, 9])

    def forward(self, x):
        out=self.layer_norm(x)
        out=self.conv1(out)
        out=self.norm1(out)
        out=self.relu1(out)
        
        out=self.conv2(out)
        out=self.norm2(out)
        return self.relu2(out + x)

class PolicyValueNetwork(nn.Module):
    def __init__(self, blocks=10, channels=154, fcl=154):
        super(PolicyValueNetwork, self).__init__()
        self.conv1_1_1 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=3, padding=1, bias=False)
        self.conv1_1_2 = nn.Conv2d(in_channels=FEATURES1_NUM, out_channels=channels, kernel_size=1, padding=0, bias=False)
        self.conv1_2 = nn.Conv2d(in_channels=FEATURES2_NUM, out_channels=channels, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(channels)

        # ResNet Blockを作成
        self.blocks = nn.Sequential(*[ResNetBlock(channels) for _ in range(blocks)])
        

        # policy head
        self.policy_conv = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.policy_bias = Bias(9*9*MAX_MOVE_LABEL_NUM)

        # value head
        self.value_conv1 = nn.Conv2d(in_channels=channels, out_channels=MAX_MOVE_LABEL_NUM, kernel_size=1, bias=False)
        self.value_norm1 = nn.BatchNorm2d(MAX_MOVE_LABEL_NUM)
        self.value_fc1 = nn.Linear(9*9*MAX_MOVE_LABEL_NUM, fcl)
        self.value_fc2 = nn.Linear(fcl, 1)

    def forward(self, x1, x2):
        x_1_1 = self.conv1_1_1(x1)
        x_1_2 = self.conv1_1_2(x1)
        x_2 = self.conv1_2(x2)
        x = F.relu(self.norm1(x_1_1 + x_1_2 + x_2))

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
