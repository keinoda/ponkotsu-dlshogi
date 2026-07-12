# 作業引き継ぎメモ (2026-07-11 時点)

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

## 未実施 (次のタスク)

vast.ai インスタンス上で:

1. `nvidia-smi` / `python3 -c "import torch; print(torch.cuda.is_available())"` で環境確認 (手順書 3 章)
2. `bash tests/ptl_transformer_smoke.sh` で Lightning 経路の動作確認 (GPU での初実行)
3. テストデータ取得: 公式評価データセット [takaoyamaoka/floodgate.hcpe](https://huggingface.co/datasets/takaoyamaoka/floodgate.hcpe) を `/workspace/test/floodgate.hcpe` へ (手順書 5-2、サイズ検証付き)
4. **本命: 60b768 モデル (exp___i60x768_fcl256, 739.9M params) の学習** —
   `NYUGYOKU_FEATURES=1 pip3 install ./external/dlshogi` してから
   `python3 -m dlshogi.ptl fit --config configs/ptl_yamaoka_60x768.yaml` (手順書 6-6)。
   ユーザーの主目的は policy/value の精度 (対局時の NPS 低下は許容)。
   **まず 1 ファイルで実測**し、スループット (局面/秒) を報告してからファイル数を増やす
5. 実測値から学習計画 (ファイル数・期間・費用) を確定する
   (計算量は resnet35x512 比で約 4 倍。全量 1 周は非現実的なので 11 ファイル ≈ 27 億局面規模から)

## 既知の注意点

- 学習は 1 ファイルあたり RAM 約 30GB を消費 (全局面をメモリ展開)
- `train_prior.sh` (旧スクリプト) は Misskey 通知がハードコード。使わず `train_prelearn.sh` を使う
- `requirements.txt` の dlshogi は `./external/dlshogi` を指す (PyPI 版 0.0.1 はスタブ)
- 本家 dlshogi の追従マージ手順は `external/README.md`
