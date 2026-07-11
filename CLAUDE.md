# ponkotsu-dlshogi 開発ガイド (Claude 向け)

将棋 AI「ponkotsu」(dlshogi 系) の開発リポジトリ。学習スクリプト・パラメータ調整・定跡生成に加え、
dlshogi 本体のソースを `external/dlshogi` に取り込み済みで、このリポジトリ単体でエンジン開発が完結する。

## 絶対的なルール

- **push 先は `keinoda/ponkotsu-dlshogi` のみ**。本家 (TadaoYamaoka/DeepLearningShogi, jj1guj/DeepLearningShogi, jj1guj のスクリプトリポジトリ) には絶対に push しない。
- dlshogi 本体の改造は `external/dlshogi` に対して行う (取り込みの経緯・本家追従手順は `external/README.md`)。

## 構成

| パス | 内容 |
| --- | --- |
| `external/dlshogi/` | dlshogi 本体 (jj1guj/wcsc35 + 本家 master 2026-06 をマージ済み)。学習コード (`dlshogi/`)・USI エンジン (`usi/`)・自己対局 (`selfplay/`) |
| `train_prior.sh` / `train_cosine_annealing.sh` | 学習実行スクリプト (`python -m dlshogi.train` を呼ぶ。Misskey 通知込み) |
| `docker/Dockerfile.vastai` | vast.ai 用オールインワンイメージ (`ghcr.io/keinoda/ponkotsu:vastai`) |
| `docker/Dockerfile` / `Dockerfile.develop` | WCSC36 当時の大会用 / 開発用イメージ |
| `docs/vastai.md` | vast.ai での実行手順書 |
| `utils/` | SPSA / Optuna パラメータ調整、強さ計測、ログ可視化 |
| `books/makebook/` | 定跡生成 |

## よく使うコマンド

```bash
# dlshogi (Python) のインストール / 改造の反映
pip install -e ./external/dlshogi

# USI エンジンのビルド (要 CUDA + TensorRT。Dockerfile.vastai のイメージ内で可能)
cd external/dlshogi/usi && make -j$(nproc)

# 学習 (ネットワークは resnet35x512_fcl512 が現行)
python -m dlshogi.train <train.hcpe...> <test.hcpe> --network resnet35x512_fcl512 ...

# Docker イメージのコンパイル検証 (PyTorch 層なしで軽い)
docker build -f docker/Dockerfile.vastai --target engine-test .
```

## CI

- `.github/workflows/docker-vastai.yml`: `docker/Dockerfile.vastai` / `external/**` の変更で `ghcr.io/keinoda/ponkotsu:vastai` を自動ビルド (workflow_dispatch でも実行可)
- `.github/workflows/docker-image.yml`: 旧 (wcsc36 ブランチ、self-hosted ランナー前提)

## 注意点

- ルート `.gitignore` は `*.txt` `*.png` `*.log` `*.so` などを広く無視する。external/ 配下にこれらの拡張子のファイルを追加する際は `git status` で追跡されているか確認すること。
- `requirements.txt` の dlshogi はローカルパス (`./external/dlshogi`) を指す。リポジトリルートから実行。
- 学習スクリプトはデータセット名・Misskey 通知先がハードコード。環境に合わせて書き換えて使う。
