# ponkotsu

ponkotsu の学習・検証・パラメータ調整・定跡生成をまとめた、将棋 AI 開発用リポジトリです。  
第36回世界コンピュータ将棋選手権(WCSC36)で5位入賞しました。  
学習ログ可視化、SPSA / Optuna による探索パラメータ調整、対局結果の整理、補助検証コードなどを含みます。

## 主な内容

| 領域 | 内容 |
| --- | --- |
| 学習 | `train_prelearn.sh`(WCSC36 事前学習の再現データ用)、`train_prior.sh`, `train_cosine_annealing.sh` による学習実行 |
| 可視化 | `log_plot.py`, `utils/plot_optuna_log.py` による学習・パラメータ調整ログの確認 |
| パラメータ調整 | `utils/param_optimize.sh`, `utils/param_optimize_spsa.sh` |
| 強さ検証 | `utils/measure_strength.sh` |
| 定跡生成処理 | `books/makebook/` |
| 補助検証 | `verify_swa_resume.py` などの検証スクリプト |
| 詳細アピール文書補足資料 | `docs/README.md`|
| 定期実行 | `systemd/` 配下の user service / timer 設定 |
| dlshogi 本体(取り込み済み) | `external/dlshogi`(jj1guj/wcsc35 + 本家 master マージ。詳細は `external/README.md`) |
| vast.ai での実行 | `docs/vastai.md`(手順書)、`docker/Dockerfile.vastai`(専用イメージ) |

## 前提

利用にあたっては以下を確認してください。

- `dlshogi` が import / 実行可能であること(本体ソースは `external/dlshogi` に取り込み済み。`pip install ./external/dlshogi` でインストール可能)
- 学習データ一式 `.hcpe` や評価用データが別途配置されていること
- GPU を使う場合は PyTorch と CUDA 環境が整っていること
- 一部スクリプトでは `curl`, `jq`, `systemd --user` などの外部コマンドが必要なこと
- 一部スクリプトにはモデルパスや通知先などがハードコードされているため、利用前に各スクリプトの設定箇所を確認・書き換えること

Python の最小限の依存パッケージは `requirements.txt` にあります。

```bash
pip install -r requirements.txt
```

開発用コンテナを使う場合は `docker/Dockerfile.develop` をベースにしています。PyTorch nightly、`dlshogi`、`cshogi` などをまとめて導入しています。

## 主要ファイル

### 学習

- `train_prior.sh`  
  事前学習用のスクリプトです。チェックポイントからの再開、ログ更新、画像生成、Misskey への通知まで含みます。
- `train_cosine_annealing.sh`  
  追加学習用のスクリプトです。cosine annealing ベースの学習と、SWA 利用の切り替えや resume 制御を含みます。

### 解析・可視化

- `log_plot.py`  
  学習ログから loss / accuracy の推移画像を生成します。
- `utils/plot_optuna_log.py`  
  Optuna の探索ログを可視化するスクリプトです。
- `utils/analyze_search_history_shapes.py`  
  探索履歴の形状や分布を確認するための解析スクリプトです。

### 調整・評価

- `utils/param_optimize.sh`  
  USI パラメータを Optuna で調整します。
- `utils/param_optimize_spsa.sh`  
  SPSA ベースの調整を実行します。
- `utils/measure_strength.sh`  
  複数設定の対戦成績を測定します。

### 補助ツール

- `verify_swa_resume.py`  
  SWA 再開時の `update_bn` 経路や device mismatch を検証するためのスクリプトです。
- `books/makebook/scripts/download_wcsc_csa.py`  
  WCSC の棋譜リストから CSA 棋譜を一括ダウンロードするスクリプトです。
- `systemd/README.md`  
  SPSA ログの定期通知設定手順です。

## 典型的な使い方

### 1. 学習を回す

このリポジトリの学習スクリプトは、保存先・実験名・データディレクトリなどを引数で受け取る前提です。

```bash
bash train_prior.sh <save_dir> <run_name> <data_dir> <test_dir> <cache_dir> <compare_log_dir> <misskey_base_url> <token_file> <visible_user_id>

bash train_cosine_annealing.sh <save_dir> <run_name> <data_dir> <test_dir> <cache_dir> <compare_log_dir> <misskey_base_url> <token_file> <visible_user_id>
```

どちらも内部で `python -m dlshogi.train` を呼び出し、エポックごとにログ更新と可視化を行います。

### 2. SWA 再開時の動作を検証する

```bash
python verify_swa_resume.py --resume <checkpoint_path> --network resnet35x512_fcl512 --gpu 0 --apply_swa_device_fix
```

### 3. 学習ログを可視化する

```bash
python log_plot.py <比較元ログ> <対象ログ>
```

実行後、リポジトリルートに loss / accuracy の画像が生成されます。

### 4. パラメータを調整する

```bash
bash utils/param_optimize.sh
bash utils/param_optimize_spsa.sh
```

これらのスクリプトは、対局実行環境やモデル配置場所に強く依存します。実運用前に保存先パスやモデルパスを書き換えて使う前提です。

## 詳細アピール文書補足資料

[詳細アピール文書](https://www.apply.computer-shogi.org/wcsc36/appeal/ponkotsu/ponkotsu_WCSC36_detail.pdf)の補足資料として [docs/README.md](docs/README.md) を配置しています。紙面の都合上掲載できなかったグラフ・対局検証データをまとめています。

## ディレクトリ構成

```text
.
├── books/                 # 棋譜取得・定跡化補助
├── docker/                # 開発・実行コンテナ定義 (Dockerfile.vastai は vast.ai 用)
├── docs/                  # 詳細アピール文書補足資料
│   ├── README.md             # 補足資料本体
│   ├── vastai.md             # vast.ai 手順書
│   └── plots/             # 補足資料用プロット画像
├── external/              # 取り込み済み外部ソース
│   └── dlshogi/              # dlshogi 本体 (jj1guj/wcsc35 + 本家 master マージ済み)
├── systemd/               # 定期通知設定
├── utils/                 # 補助スクリプト群
├── log_plot.py            # 学習ログ可視化
├── train_prior.sh         # 事前学習スクリプト
├── train_cosine_annealing.sh  # 追加学習スクリプト
└── verify_swa_resume.py   # SWA 検証スクリプト
```

## 補足

- `requirements.txt` は最小限です。実際には `dlshogi` 側の依存と CUDA 環境が前提になります。
- `systemd/README.md` には SPSA ログの定期通知のセットアップ手順があります。
