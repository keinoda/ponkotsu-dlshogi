# makebook_think_like.py README

この README は、`tools/makebook_think_like.py` の使い方を日本語で説明するものです。

## 1. 概要

`makebook_think_like.py` は、`EVAL_LEARN` を使わずに `makebook think` 相当の処理を行う補助スクリプトです。

主な機能:

- USI エンジンを呼び出して候補手を探索
- `makebook from_sfen` を使って book DB を生成
- MultiPV 対応
- `nodes` / `depth` 指定対応
- 複数ワーカーによる並列探索
- 既存 book DB へのマージ
- 定期スナップショット保存
- 進捗表示、速度表示、SIGINT(Ctrl+C)対応

## 2. 前提条件

- Python 3.8 以上
- USI エンジン実行ファイル
- 評価関数ディレクトリ (必要なエンジン設定に応じて)
- 入力 SFEN ファイル

例:

- エンジン: `./YaneuraOu-by-gcc`
- EvalDir: `./eval_suisho11b/`
- 入力: `test.sfens`

## 3. 基本的な使い方

### 3.1 単一 SFEN ファイル

```bash
python3 tools/makebook_think_like.py \
  --engine ./YaneuraOu-by-gcc \
  --eval-dir ./eval_suisho11b/ \
  --threads 2 \
  think test.sfens test_book.db \
  moves 320 nodes 100000 multipv 4
```

### 3.2 先手/後手ファイル分離 (bw)

```bash
python3 tools/makebook_think_like.py \
  --engine ./YaneuraOu-by-gcc \
  --eval-dir ./eval_suisho11b/ \
  --threads 2 \
  think bw black.sfens white.sfens test_book.db \
  moves 320 depth 18 multipv 4
```

## 4. コマンド構文

スクリプト全体:

```text
python3 tools/makebook_think_like.py [グローバル引数] think [think引数]
```

### 4.1 グローバル引数

- `--engine PATH` (必須)
  - USI エンジンの実行ファイル
- `--eval-dir PATH`
  - `setoption name EvalDir value ...` に渡す値
- `--cwd PATH`
  - エンジン起動時の作業ディレクトリ
- `--threads N`
  - 並列ワーカー数 (デフォルト: 1)
- `--usi-hash N`
  - ワーカーごとの `USI_Hash` を明示指定
  - 未指定時は **エンジンデフォルト値** を使用
- `--timeout SEC`
  - 1 回の探索タイムアウト秒 (デフォルト: 120)
- `--quiet`
  - 進捗表示を抑制

### 4.2 think 引数

基本:

```text
think <sfen_file> <book_name> [オプション...]
```

bw モード:

```text
think bw <black_file> <white_file> <book_name> [オプション...]
```

指定可能オプション:

- `moves N`
  - 各入力局面から展開する最大手数
- `depth D`
  - 探索深さ
- `nodes N`
  - ノード数指定 (`depth` の代わりに利用可)
- `startmoves N`
  - 何手目以降を候補化するか (1 始まり)
- `cluster ID NUM`
  - 局面分割実行
- `book_save_interval SEC`
  - 定期スナップショット保存間隔 (秒)
- `multipv N`
  - MultiPV 本数

## 5. 出力 DB の仕様

### 5.1 depth の扱い

- `depth` を明示指定した場合:
  - book の depth 列は指定値で固定
- `depth` 未指定 (`nodes` など) の場合:
  - 実探索で得られた深さを手ごとに反映

### 5.2 move_count の扱い

- 新規生成手の `move_count` は固定 `800`
  - 本家 `makebook think` の実装意図に合わせた動作

### 5.3 2手目がPVにない場合

- 2手目を補完する追加探索は行いません
- 1手目のみを書き出します (応手は `None` 相当)

### 5.4 既存 DB へのマージ

- 出力先 `book_name` が既に存在する場合:
  - 既存 DB と新規結果をマージ
  - 同一 `sfen + move` の `count` は加算
  - depth は深い方を採用
  - value/ponder は depth 優先、同深さなら value 優先

### 5.5 ソート

- `makebook sort` ではなく Python 側で同等ソート
- 並び順:
  - SFEN 昇順
  - 同一 SFEN 内は `move_count` 降順、次に `value` 降順

## 6. スキップ判定

既存 book DB がある場合のスキップは以下:

- `depth` 明示指定時のみ有効
- 条件:
  - 既存の最大 depth が指定 depth より深い → スキップ
  - 既存 depth が同一で、その depth の手数が MultiPV 以上 → スキップ

注意:

- `depth` 未指定 (`nodes` 運用など) では、事前スキップ判定を行いません。

## 7. 進捗表示と速度表示

通常モードでは次を表示します:

- 開始時の探索対象局面数 (`actual search positions`)
- 1 局面完了ごとの進捗ログ
- 1 分ごとの平均速度
  - 速い場合: `positions/min`
  - 遅い場合: `min/position`

## 8. 定期保存

`book_save_interval` 秒ごとに、進捗が増えていればスナップショット保存します。

ファイル名規則:

- 出力先が `foo.db` のとき: `foo.db-1.db`, `foo.db-2.db`, ...
- 出力先が `foo` のとき: `foo.db-1.db`, `foo.db-2.db`, ...

## 9. 中断 (Ctrl+C)

- Ctrl+C 受信時:
  - ワーカーへ `stop` を送信
  - 進行中処理の停止を試行
- その時点での途中結果は、最終 book には確定保存しません
  - 定期保存タイミングで作られたスナップショットがあれば残ります

## 10. トラブルシューティング

### Q1. `error: no candidate positions`

主な原因:

- 入力 SFEN が空、またはフォーマット不正
- `startmoves` / `moves` の条件で候補がゼロ
- (`depth` 明示時のみ) 既存 DB スキップで全件除外

確認ポイント:

- 入力ファイル内容
- `moves`, `startmoves`, `depth`, `multipv`
- 出力先 DB の既存状態

### Q2. メモリ使用量が大きい

対策例:

- `--threads` を減らす
- `--usi-hash` を明示的に小さめ設定
  - 例: `--usi-hash 256`

### Q3. 探索が遅い

対策例:

- `--threads` を増やす (CPU/メモリに余裕がある場合)
- `nodes` を下げる、または `depth` を下げる
- `multipv` を減らす

## 11. 補足

このスクリプトは運用補助を目的としたツールです。
本家実装との差異が必要な場合は、要件に応じて拡張してください。
