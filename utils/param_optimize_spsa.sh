#!/bin/sh

# SPSA によるパラメータ最適化 (dev-vs-dev 有効: Fishtest方式)
# Phase 1+3 の top-5 median を初期値として fine-tuning する
#
# dev-vs-dev有効: 30 iter × 150局 (θ+50 + θ-50 + θ+vsθ-50) = 4,500局

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_DIR=$(dirname "$SCRIPT_DIR")

PYTHONPATH="$REPO_DIR" python -u -m utils.usi_params_spsa \
/root/YaneuraOu/source/FukauraOu-by-gcc /root/yaneuraou-V921-dev-mac-all/source/YaneuraOu-by-gcc \
--options1 PV_Interval:0,PV_Mate_Search_Threads:3,DNN_Model:/root/models/model_resnet35x512-348.onnx,NetworkDelay2:0,MaxMovesToDraw:324 \
--options2 PV_Interval:0,NetworkDelay2:0,Threads:2,USI_Hash:256,EvalDir:/volume/book/suisho11b,FV_SCALE:28,MaxMovesToDraw:324 \
--opening /volume/measure_strength/yaneuraou_start/start_sfens.txt \
--games 50 \
--byoyomi 2000 \
--csa /volume/param_optimize/kifu/resnet35x512_spsa \
--init_params "C_init:117,C_base:45670,C_fpu_reduction:17,C_init_root:132,C_base_root:26661,Softmax_Temperature:173" \
--suggest_params "C_init:100~200,C_base:20000~50000,C_fpu_reduction:0~40,C_init_root:100~200,C_base_root:20000~50000,Softmax_Temperature:100~200" \
--optimize_side total \
--spsa_iterations 30 \
--spsa_c_end 0.05 \
--spsa_r_end 0.01 \
--spsa_dev_vs_dev \
--spsa_verify_games 200 \
--spsa_log /volume/param_optimize/log_spsa_total.jsonl \
--spsa_checkpoint /volume/param_optimize/spsa_total_checkpoint.json \
 | tee -a /volume/param_optimize/log_spsa_total.txt
