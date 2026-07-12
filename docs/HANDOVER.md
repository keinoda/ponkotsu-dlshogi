# 作業引き継ぎメモ (2026-07-12 時点)

新しいセッション(または vast.ai インスタンス上の Claude)が作業を再開するためのスナップショット。
まず `CLAUDE.md` → `docs/vastai.md` → `external/README.md` の順に読むこと。

## 絶対的なルール

- **push 先は `keinoda/ponkotsu-dlshogi` のみ**。本家 (TadaoYamaoka / jj1guj) には絶対 push しない
- 作業ブランチ: `claude/shogi-dl-vast-ai-setup-9mwh3n`

## 完了済みの状態

| 項目 | 状態 |
| --- | --- |
| dlshogi 本体 | `external/dlshogi` に取り込み済み。jj1guj/wcsc35 (27d0f30) + 本家 master (ff520f0, 2026-06-25) のマージ。TensorRT 10 対応、`ptl.py` の pin_memory 修正済み (詳細: `external/README.md`) |
| Docker イメージ | `keinoda/ponkotsu:vastai` (Docker Hub) / `ghcr.io/keinoda/ponkotsu:vastai`。公開済み・匿名 pull 可。PyTorch cu128 / TensorRT / lightning / hf CLI / pandas 入り。USI エンジンビルド済み (`dlshogi-usi`)。リポジトリ一式は `/opt/ponkotsu-dlshogi` に同梱 (.git 込み) |
| CI | `.github/workflows/docker-vastai.yml`。`docker/**` `external/**` の push で自動ビルド → 両レジストリへ自動配布 (Docker Hub secrets 設定済み) |
| 学習データ | [HF: penguinkumimanu/generic_ponkostu_wcsc36_Pre-learning](https://huggingface.co/datasets/penguinkumimanu/generic_ponkostu_wcsc36_Pre-learning)。hcpe 58 ファイル・約 544GB・1 ファイル ≈ 2.5 億局面。本家の「27 億局面」相当 ≈ 11 ファイル |
| 学習スクリプト | `train_prelearn.sh` (自動再開、`NETWORK` / `EXTRA_TRAIN_ARGS` 環境変数対応)。テストデータは floodgate から作成 (手順書 5-2) |
| スモークテスト | `tests/ptl_transformer_smoke.sh` (ResNet+Transformer × Lightning、1 ファイルのみ DL)。実データスライスで検証済み |

## 2026-07-12 実測セッションの結果 (vast.ai 2×RTX 5090, RAM 440GB)

1. ✅ 環境確認 OK (torch 2.11.0+cu128、CUDA 認識、`/workspace` 2TB)
2. ✅ スモークテスト PASSED (`jsonargparse[signatures]` 不足を発見 → Dockerfile.vastai に追加済み)
3. ✅ `/workspace/test/floodgate.hcpe` 配置・サイズ検証 OK
4. ✅ 入玉特徴量ビルド済み (`FEATURES2_NUM=119`)。60b768 の実測完了 (手順書 6-6 に詳細表):
   - 1×5090 + compile: **645 局面/秒** (1 ファイル 4.5 日)
   - 2×5090 DDP (batch 256×累積8) + compile: **1,149 局面/秒** (1 ファイル 2.5 日、11 ファイル 27.7 日)
   - 参考 resnet35x512_fcl512: 1,770 局面/秒 (1 ファイル 39 時間)。steps=5000 で test acc policy 37.2% / value 58.4%
5. ⏳ **規模の決定待ち (ユーザー判断: 一旦テストのみ)**。1 ファイル学習の DDP run
   (`/workspace/models/yamaoka60x768`、cosine t_initial=60000 = 1 ファイル正規構成) を走行中。
   完走で「1 ファイル 60b768 モデル + floodgate 精度カーブ」が得られる → スケール判断材料

## 未実施 (次のタスク)

1. **蒸留 A/B 比較** — 教師 = ponkotsu-wcsc36 公開評価関数 (model.onnx 667,162,409 bytes。
   再配布禁止のためリポジトリには含めない。**ユーザーから Drive リンクを会話で受け取り** /workspace/teacher/ に展開)。
   `hcpe_re_eval` で data_000 を「α=1 (全置換)」「α=0.5 (ブレンド)」の 2 種類作り (手順書 5-3)、
   60b768 を (a) 素の再現データ (b) α=1 (c) α=0.5 の 3 構成で同一ステップ数だけ学習して
   floodgate.hcpe の val loss / accuracy で比較する
2. 走行中の 1 ファイル DDP run (= A/B の (a) 素データ構成に相当) の扱いを決める:
   完走 (~2.5 日) させて精度カーブを取るか、A/B 比較を短ステップで揃えるため途中打ち切りにするか
3. 実測値と A/B 結果から学習計画 (ファイル数・構成・期間・費用) を確定する
   → 11 ファイルなら DDP で分割継ぎ (RAM 制約: rank あたり全量ロード × 2)
4. 必要なら ONNX 変換 → USI エンジンでの動作確認 (手順書 7 章)

## 既知の注意点

- 学習は 1 ファイルあたり RAM 約 30GB を消費 (全局面をメモリ展開)
- `train_prior.sh` (旧スクリプト) は Misskey 通知がハードコード。使わず `train_prelearn.sh` を使う
- `requirements.txt` の dlshogi は `./external/dlshogi` を指す (PyPI 版 0.0.1 はスタブ)
- 本家 dlshogi の追従マージ手順は `external/README.md`
