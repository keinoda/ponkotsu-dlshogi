#!/bin/bash
# ponkotsu WCSC36 事前学習の再現データで学習を回すスクリプト
#
# データセット: https://huggingface.co/datasets/penguinkumimanu/generic_ponkostu_wcsc36_Pre-learning
#   (generic_ponkostu_wcsc36_Prelearning_data_NNN.hcpe を data_dir に配置しておく)
#
# 使い方:
#   bash train_prelearn.sh <save_dir> <name> <data_dir> <test_hcpe> [files_per_epoch] [cache_dir]
#
#   save_dir        : チェックポイント・モデル・ログの保存先 (例: /workspace/models)
#   name            : 実験名 (例: prelearn01)
#   data_dir        : 再現データの .hcpe を置いたディレクトリ (例: /workspace/data/prelearn)
#   test_hcpe       : テストデータ (floodgate 等から作成した .hcpe)
#   files_per_epoch : 1エポックに使うファイル数 (省略時 1。1ファイルあたり RAM 約30GB を目安に増やす)
#   cache_dir       : 指定すると --cache を使用 (2周目以降のロード高速化。データと同規模のディスクを消費)
#
# 環境変数 EXTRA_TRAIN_ARGS で dlshogi.train への追加引数を渡せる。例:
#   EXTRA_TRAIN_ARGS="--use_compile" bash train_prelearn.sh ...   # torch.compile による高速化 (本家 2026-05 の対応)
#
# チェックポイントが存在する場合は最新の続きから自動再開する。
# 学習設定は train_prior.sh (WCSC36 当時) と同一。

set -eu

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

save_dir=$1
name=$2
data_dir=$3
test_file=$4
files_per_epoch=${5:-1}
cache_dir=${6:-}

checkpoint_dir="${save_dir}/${name}"
model_dir="${checkpoint_dir}/model"
log="${checkpoint_dir}/train_log.txt"
mkdir -p "${model_dir}"

# データファイル一覧 (番号順)。欠番があっても問題ない
mapfile -t files < <(ls -v "${data_dir}"/generic_ponkostu_wcsc36_Prelearning_data_*.hcpe 2>/dev/null)
total=${#files[@]}
if [ "${total}" -eq 0 ]; then
    echo "エラー: ${data_dir} に generic_ponkostu_wcsc36_Prelearning_data_*.hcpe が見つかりません" >&2
    exit 1
fi
epochs=$(( (total + files_per_epoch - 1) / files_per_epoch ))
echo "データ ${total} ファイル / ${files_per_epoch} ファイルずつ → 全 ${epochs} エポック"

# 最新のチェックポイント+1から再開する
start=1
for f in $(ls -v "${checkpoint_dir}"/checkpoint_${name}-???.pth 2>/dev/null); do chkp=$f; done
if [ -n "${chkp:-}" ]; then
    start=$(( 10#${chkp: -7:3} + 1 ))
    echo "チェックポイント ${chkp} から再開 (epoch ${start})"
fi

for (( i=start; i<=epochs; i++ )); do
    off=$(( (i - 1) * files_per_epoch ))
    src=("${files[@]:off:files_per_epoch}")

    resume=""
    if [ "$i" -gt 1 ]; then
        rrr=$(printf "%03d" $((i - 1)))
        resume="-r ${checkpoint_dir}/checkpoint_${name}-${rrr}.pth"
    fi

    cache_opt=""
    if [ -n "${cache_dir}" ]; then
        mkdir -p "${cache_dir}"
        cache_opt="--cache ${cache_dir}/cache_${name}_$(printf "%03d" "$i")"
    fi

    echo "epoch ${i}/${epochs} start: ${src[*]}"

    python3 -m dlshogi.train "${src[@]}" "${test_file}" \
        ${resume} \
        --checkpoint "${checkpoint_dir}/checkpoint_${name}-{epoch:03}.pth" \
        --model "${model_dir}/model_${name}-{epoch:03}.pth" \
        --network resnet35x512_fcl512 -e 1 \
        --use_average --use_evalfix --use_amp --amp_dtype bfloat16 --temperature 0 --lr 0.2 \
        --lr_scheduler ReduceLROnPlateau'('eps=1e-20,factor=0.5')' --scheduler_step_mode epoch \
        ${cache_opt} ${EXTRA_TRAIN_ARGS:-} --log "${log}"
done
