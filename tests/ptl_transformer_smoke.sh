#!/bin/bash
# ResNet+Transformer モデル (policy_value_network_exp___i) を Lightning 経路 (dlshogi.ptl) で
# 1 ファイル学習するスモークテスト。
#
# やること:
#   1. WCSC36 再現データセットから hcpe を「1 ファイルだけ」ダウンロード (既にあればスキップ)
#   2. 同じファイルの先頭から検証用スライスを切り出す (追加ダウンロードなし)
#   3. dlshogi.ptl fit を少ステップ実行し、checkpoint が生成されることを確認
#
# 使い方 (vast.ai インスタンス上):
#   bash tests/ptl_transformer_smoke.sh
#
# 環境変数で調整可能:
#   NETWORK     : ネットワーク (デフォルト exp___i15x224_fcl256 = ResNet+Transformer 小型構成)
#   MAX_STEPS   : 学習ステップ数 (デフォルト 200)
#   BATCH       : バッチサイズ (デフォルト 256)
#   VAL_POS     : 検証用に切り出す局面数 (デフォルト 500000)
#   ACCELERATOR : gpu / cpu (デフォルト gpu)
#   PRECISION   : 16-mixed / 32-true など (デフォルト 16-mixed。cpu なら 32-true 推奨)
#   DATA_DIR    : データ置き場 (デフォルト /workspace/data/prelearn)
#   WORK_DIR    : テスト作業ディレクトリ (デフォルト /workspace/smoke_ptl)
#
# 注意: 学習ファイル (9.5GB ≈ 2.5億局面) は全局面をメモリに載せるため RAM 約 30GB を要する。

set -eu

NETWORK=${NETWORK:-exp___i15x224_fcl256}
MAX_STEPS=${MAX_STEPS:-200}
BATCH=${BATCH:-256}
VAL_POS=${VAL_POS:-500000}
ACCELERATOR=${ACCELERATOR:-gpu}
PRECISION=${PRECISION:-16-mixed}
DATA_DIR=${DATA_DIR:-/workspace/data/prelearn}
WORK_DIR=${WORK_DIR:-/workspace/smoke_ptl}

# NUM_WORKERS > 0 にする場合は cache が必須 (上流仕様。下の [2.5/3] で自動構築する)
NUM_WORKERS=${NUM_WORKERS:-null}
if [ "${NUM_WORKERS}" != "null" ] && [ "${NUM_WORKERS}" -gt 0 ]; then
    CACHE=${CACHE:-${WORK_DIR}/hcpe3_cache}
else
    CACHE=${CACHE:-null}
fi

DATASET=penguinkumimanu/generic_ponkostu_wcsc36_Pre-learning
FILE=generic_ponkostu_wcsc36_Prelearning_data_000.hcpe
HCPE_RECORD_SIZE=38  # sizeof(HuffmanCodedPosAndEval)

train_file="${DATA_DIR}/${FILE}"
val_file="${WORK_DIR}/val_head_${VAL_POS}.hcpe"
mkdir -p "${DATA_DIR}" "${WORK_DIR}"

echo "=== [1/3] 学習データ (1 ファイルのみ) の用意 ==="
if [ -f "${train_file}" ]; then
    echo "既に存在: ${train_file} (ダウンロードをスキップ)"
else
    hf download "${DATASET}" --repo-type dataset \
        --include "${FILE}" --local-dir "${DATA_DIR}"
fi

echo "=== [2/3] 検証用スライスの切り出し (先頭 ${VAL_POS} 局面) ==="
head -c "$(( VAL_POS * HCPE_RECORD_SIZE ))" "${train_file}" > "${val_file}"
echo "$(du -h "${val_file}" | cut -f1) -> ${val_file}"

if [ "${NUM_WORKERS}" != "null" ] && [ ! -f "${CACHE}" ]; then
    echo "=== [2.5/3] hcpe3 cache の事前構築 (num_workers > 0 に必須) ==="
    python3 -m dlshogi.utils.make_hcpe3_cache "${train_file}" --cache "${CACHE}" --use_average --use_evalfix
fi

echo "=== [3/3] dlshogi.ptl (Lightning) で ${NETWORK} を ${MAX_STEPS} ステップ学習 ==="
config="${WORK_DIR}/smoke_config.yaml"
cat > "${config}" <<EOF
seed_everything: 0
trainer:
  max_epochs: 1
  max_steps: ${MAX_STEPS}
  val_check_interval: $(( MAX_STEPS / 2 ))
  limit_val_batches: 10
  precision: ${PRECISION}
  accelerator: ${ACCELERATOR}
  devices: 1
  gradient_clip_val: 10.0
  default_root_dir: ${WORK_DIR}
  enable_progress_bar: true
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        save_top_k: 1
        monitor: val/loss
model:
  network: ${NETWORK}
  val_lambda: 0.333
  model_filename: smoke-{epoch:03}
data:
  train_files:
    - ${train_file}
  val_files:
    - ${val_file}
  batch_size: ${BATCH}
  val_batch_size: ${BATCH}
  num_workers: ${NUM_WORKERS}
  cache: ${CACHE}
  use_average: true
  use_evalfix: true
  temperature: 1.0
optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1e-4
    weight_decay: 1e-2
lr_scheduler:
  class_path: dlshogi.lr_scheduler.CosineLRScheduler
  init_args:
    t_initial: 100
    lr_min: 1e-8
    warmup_t: 10
    warmup_lr_init: 1e-7
    warmup_prefix: true
EOF

python3 -m dlshogi.ptl fit --config "${config}"

echo "=== 結果確認 ==="
ckpt=$(find "${WORK_DIR}/lightning_logs" -name '*.ckpt' | tail -1)
if [ -n "${ckpt}" ]; then
    echo "checkpoint 生成を確認: ${ckpt}"
    echo "SMOKE TEST PASSED"
else
    echo "checkpoint が見つかりません" >&2
    echo "SMOKE TEST FAILED" >&2
    exit 1
fi
