# external/ — 取り込み済み外部ソース

## external/dlshogi

dlshogi (DeepLearningShogi) のソース一式です。**このリポジトリ内のコピーが正であり、自由に改造してよい**前提で取り込んでいます。変更は keinoda/ponkotsu-dlshogi にのみ push してください(本家 TadaoYamaoka / jj1guj には push しない)。

### 出所(2026-07-11 取り込み)

以下 2 系列をマージしたスナップショットです:

| 系列 | リポジトリ / ブランチ | コミット |
| --- | --- | --- |
| ponkotsu 改造版(ベース) | `jj1guj/DeepLearningShogi` @ `wcsc35` | `27d0f30` (2026-03-23) |
| 本家最新 | `TadaoYamaoka/DeepLearningShogi` @ `master` | `ff520f0` (2026-06-25) |

マージベースは `687bac0` (2025-06-20)。ponkotsu 側 23 コミット、本家側 61 コミットを統合しています。

### 手動で解決した衝突

| ファイル | 解決内容 |
| --- | --- |
| `usi/nn_tensorrt.cpp` | 本家の複数最適化プロファイル対応(`InferenceSlot` 構造)を、ponkotsu 側の TensorRT 10 API(`setInputShape` / `setTensorAddress` / `enqueueV3` / `setMemoryPoolLimit`)に移植。TRT10 の name-based API では binding index のオフセット計算が不要なため `bindings` / `binding_offset` は削除 |
| `dlshogi/train.py` | autocast は ponkotsu 側の新 API `torch.autocast(device_type_str, ...)` を採用し、train モード切替は本家の `compiled_model.train()` を採用 |
| `setup.py` | 本家の `__builtins__` try/except 修正を採用(ponkotsu 側の `build_extensions` 内 numpy include 追加も維持) |
| `pyproject.toml` | ビルド要件を統合: `setuptools>=57.4.0, wheel, numpy>=1.20.0, cython>=0.29.0` |

ponkotsu 側の主な独自変更(マージ後も有効): TensorRT 10 対応、SWA 適用/再開のバグ修正、BN 再推定スキップオプション、ReduceLROnPlateau 対応、external data を使わない保存、VRAM OOM ワークアラウンド。

本家側の主な新規変更: 学習の PyTorch Lightning 化 (`dlshogi/ptl.py`, DataModule)、hcpe3 デコードのマルチスレッド化、USI エンジンの複数最適化プロファイル対応、各種ユーティリティ (`dlshogi/utils/`) の拡充。

### 次回以降の本家取り込み手順

```bash
# 作業用ディレクトリで
git clone https://github.com/jj1guj/DeepLearningShogi.git dlshogi-work
cd dlshogi-work
git remote add upstream https://github.com/TadaoYamaoka/DeepLearningShogi.git
git fetch upstream master

# このリポジトリの external/dlshogi を「現状」としてブランチ化
git checkout -b vendored 27d0f30   # ← 前回取り込み時の wcsc35 コミット
rsync -a --delete --exclude=.git <ponkotsu-dlshogiのパス>/external/dlshogi/ ./
git add -A && git commit -m "sync with vendored copy"

# 本家最新をマージ(衝突があれば解決)
git merge upstream/master

# 結果を external/dlshogi に書き戻す
rsync -a --delete --exclude=.git ./ <ponkotsu-dlshogiのパス>/external/dlshogi/
```

マージ後は `docker build --target engine-test` (docker/Dockerfile.vastai) で USI エンジンがコンパイルできることを確認してください。
