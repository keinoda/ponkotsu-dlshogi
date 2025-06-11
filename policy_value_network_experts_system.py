import torch
import torch.nn as nn
import torch.nn.functional as F
from mup import MuReadout

from dlshogi.common import *

class Bias(nn.Module):
    def __init__(self, shape):
        super(Bias, self).__init__()
        self.bias=nn.Parameter(torch.zeros(shape))

    def forward(self, input):
        return input + self.bias

class Router(nn.Module):
    def __init__(self, channels):
        super(Router, self).__init__()
        self.fc = nn.Linear(9*9*channels, 3)

    def forward(self, x):
        return self.fc(torch.flatten(x, 1))

class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.router = Router(channels)
        self.norm1 = nn.ModuleList([
            nn.BatchNorm2d(channels),
            nn.BatchNorm2d(channels),
            nn.BatchNorm2d(channels)
        ])
        self.relu1 = nn.ModuleList([
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True)
        ])
        self.conv1 = nn.ModuleList([
            nn.Conv2d(channels,channels,kernel_size=3,padding=1,bias=False),
            nn.Conv2d(channels,channels,kernel_size=(1,9),padding=(0,4),bias=False),
            nn.Conv2d(channels,channels,kernel_size=1,padding=0,bias=False)
        ])
        self.norm2 = nn.ModuleList([
            nn.BatchNorm2d(channels),
            nn.BatchNorm2d(channels),
            nn.BatchNorm2d(channels)
        ])
        self.relu2 = nn.ModuleList([
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True),
            nn.ReLU(inplace=True)
        ])
        self.conv2 = nn.ModuleList([
            nn.Conv2d(channels,channels,kernel_size=3,padding=1,bias=False),
            nn.Conv2d(channels,channels,kernel_size=(9,1),padding=(4,0),bias=False),
            nn.Conv2d(channels,channels,kernel_size=1,padding=0,bias=False)
        ])

    def forward(self, x):
        route_probs = self.router(x)

        if self.training:
            # 訓練時：ソフトルーティング（勾配の流れを良くする）
            route_probs_soft = F.softmax(route_probs, dim=1)

            # 全てのエキスパートの出力を計算
            expert_outputs = []
            identity = x

            for i in range(3):
                out = self.conv1[i](x)
                out = self.norm1[i](out)
                out = self.relu1[i](out)

                out = self.conv2[i](out)
                out = self.norm2[i](out)
                out = self.relu2[i](out + identity)

                expert_outputs.append(out)

            # 重み付き平均で結合
            expert_outputs = torch.stack(expert_outputs, dim=1)
            route_probs_soft = route_probs_soft.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            output = torch.sum(expert_outputs * route_probs_soft, dim=1)

        else:
            # 推論時：ハードルーティング（効率重視）
            route = torch.argmax(route_probs, 1)
            outputs = torch.zeros_like(x)
            identity = x

            for expert_id in range(3):
                mask = (route == expert_id)
                if mask.any():
                    masked_x = x[mask]
                    out = self.conv1[expert_id](masked_x)
                    out = self.norm1[expert_id](out)
                    out = self.relu1[expert_id](out)

                    out = self.conv2[expert_id](out)
                    out = self.norm2[expert_id](out)
                    out = self.relu2[expert_id](out + masked_x)

                    outputs[mask] = out

            output = outputs

        return output

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
        self.value_fc2 = MuReadout(fcl, 1, readout_zero_init=True)

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
