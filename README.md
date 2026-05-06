# ponkotsu

ponkotsu の学習・検証・パラメータ調整・棋譜処理をまとめた、将棋 AI 開発用リポジトリです。  
WCSC 向けの運用で使ったスクリプト群を中心に、学習ログ可視化、SPSA / Optuna 調整、対局結果の整理、補助検証コードを収めています。

## 何が入っているか

このリポジトリは、単一のアプリケーションというより、ponkotsu の学習運用を支える実験・自動化資産の集合です。

| 領域 | 内容 |
| --- | --- |
| 学習 | `train_prior.sh`, `train_cosine_annealing.sh` による学習実行 |
| 可視化 | `log_plot.py`, `utils/plot_optuna_log.py` による学習・探索ログの確認 |
| パラメータ調整 | `utils/param_optimize.sh`, `utils/param_optimize_spsa.sh` |
| 強さ検証 | `utils/measure_strength.sh` |
| 棋譜処理 | `books/makebook/` |
| 補助検証 | `verify_swa_resume.py` などの検証スクリプト |
| 詳細アピール文書補足資料 | `docs/appeal_supplement.md`|
| 定期実行 | `systemd/` 配下の user service / timer 設定 |

## 前提

このリポジトリ単体では完結しません。実運用には少なくとも次が必要です。

- `dlshogi` が import / 実行可能であること
- 学習データ一式 `.hcpe` や評価用データが別途配置されていること
- GPU を使う場合は PyTorch と CUDA 環境が整っていること
- 一部スクリプトでは `curl`, `jq`, `systemd --user` などの外部コマンドを使用すること

Python 依存の最小セットは `requirements.txt` にあります。

```bash
pip install -r requirements.txt
```

開発用コンテナを使う場合は `docker/Dockerfile.develop` が土台になります。ここでは PyTorch nightly、`dlshogi`、`cshogi` などをまとめて導入しています。

## 主要ファイル

### 学習

- `train_prior.sh`  
  事前学習用のスクリプトです。チェックポイント継続、ログ更新、画像生成、Misskey への通知まで含みます。
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
- `books/makebook/get_wcsc_csa.sh`  
  WCSC 棋譜の取得処理をまとめて実行するスクリプトです。
- `books/makebook/download_wcsc_csa.py`  
  WCSC の棋譜リストから CSA 棋譜を一括ダウンロードするスクリプトです。
- `systemd/README.md`  
  SPSA ログの定期通知設定手順です。

## 典型的な使い方

### 1. 学習を回す

このリポジトリの学習スクリプトは、保存先・実験名・データディレクトリなどを引数で受け取る前提です。

```bash
bash train_prior.sh <save_dir> <run_name> <data_dir> <test_dir> <cache_dir> <compare_log_dir> <misskey_base_url> <token_file>

bash train_cosine_annealing.sh <save_dir> <run_name> <data_dir> <test_dir> <cache_dir> <compare_log_dir> <misskey_base_url> <token_file>
```

どちらも内部で `python -m dlshogi.train` を呼び出し、エポックごとにログ更新と可視化を行います。

### 2. 学習ログを可視化する

```bash
python log_plot.py <比較元ログ> <対象ログ>
```

実行後、ルート配下に loss / accuracy の画像が生成されます。

### 3. パラメータを調整する

```bash
bash utils/param_optimize.sh
bash utils/param_optimize_spsa.sh
```

これらのスクリプトは、対局実行環境やモデル配置場所に強く依存します。実運用前に保存先パスやモデルパスを書き換えて使う前提です。

### 4. SWA 再開まわりを検証する

```bash
python verify_swa_resume.py --resume <checkpoint_path> --network resnet35x512_fcl512 --gpu 0 --apply_swa_device_fix
```

## 詳細アピール文書補足資料

詳細アピール文書の補足資料として [docs/appeal_supplement.md](docs/appeal_supplement.md) を配置しています。紙面の都合上掲載できなかったグラフ・対局検証データをまとめています。

## ディレクトリ構成

```text
.
├── books/                 # 棋譜取得・定跡化補助
├── docker/                # 開発・実行コンテナ定義
├── docs/                  # 詳細アピール文書補足資料
│   ├── appeal_supplement.md  # 補足資料本体
│   └── plots/             # 補足資料用プロット画像
├── systemd/               # 定期通知設定
├── utils/                 # 補助スクリプト群
├── log_plot.py            # 学習ログ可視化
├── train_prior.sh         # 事前学習スクリプト
├── train_cosine_annealing.sh  # 追加学習スクリプト
└── verify_swa_resume.py   # SWA 検証スクリプト
```

## 公開にあたって

このリポジトリは、汎用ライブラリではなく実験運用に寄った作業リポジトリです。  
そのため、スクリプト内には環境依存のモデル配置、外部通知先、対局実行基盤などが含まれます。

公開用に扱う際は、次を切り分ける前提で読むのが現実的です。

- 環境依存パスの外出し
- データセット配置ルールの明文化
- 通知や対局実行部分の設定ファイル化
- 実験ログと成果物の整理
- 機密情報や個人向け通知設定が含まれないことの確認

## 補足

- `requirements.txt` は最小限です。実際には `dlshogi` 側の依存と CUDA 環境が前提になります。
- `systemd/README.md` には定期実行のセットアップ手順があります。
