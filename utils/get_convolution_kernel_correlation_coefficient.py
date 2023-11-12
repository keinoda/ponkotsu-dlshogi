import sys
sys.path.append('../')
from policy_value_network_densenet import *
import torch
from dlshogi import serializers
import numpy as np

path = "/home/vmlab/jj1guj/densenet10/model/model_densenet10-288.pth"

# model定義
model = PolicyValueNetwork()
device = torch.device(f"cuda:0")
model.to(device)

# パラメータの読み込み
serializers.load_npz(path, model)
model.eval()

# パラメータの取り出し
params = list(model.parameters())

# 3x3のカーネルだけ取り出して3x3のtorch.tensor()のリストを作成する
params_3x3 = []
for p in params:
    if p.shape[2:] == torch.Size([3, 3]):
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                params_3x3.append(p[i][j])

# 念のためオブジェクトを削除
del params

# パラメータを各インデックスごとにまとめる
params_3x3_by_index=[[] for i in range(9)]
for p in params_3x3:
    for i in range(3):
        for j in range(3):
            params_3x3_by_index[3 * i + j].append(p[i][j].item())

# 念のためオブジェクトを削除
del params_3x3

# 各インデックスごとにまとめたパラメータの配列をnumpyに変換
params_3x3_by_index = np.array(params_3x3_by_index)

# 相関係数を求める
correlation_coefficient = np.round(np.corrcoef(params_3x3_by_index), 3)

# CSVに出力
for c in correlation_coefficient:
    print(",".join(map(str, c)))