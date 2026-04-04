#!/bin/sh

# SPSA によるパラメータ最適化 (後手用: 後手勝率で最適化)
# SPSA共通の結果を初期値として使用する
# ※ init_params は SPSA共通の最終パラメータで更新すること
#
# 30 iter × 100局 (θ+50局 + θ-50局) = 3,000局

python -u -m utils.usi_params_spsa \
/root/YaneuraOu/source/FukauraOu-by-gcc /root/yaneuraou-V921-dev-mac-all/source/YaneuraOu-by-gcc \
--options1 PV_Interval:0,PV_Mate_Search_Threads:3,DNN_Model:/root/models/model_resnet35x512-348.onnx,NetworkDelay2:0,MaxMovesToDraw:324 \
--options2 PV_Interval:0,NetworkDelay2:0,Threads:2,USI_Hash:256,EvalDir:/volume/book/suisho11b,FV_SCALE:28,MaxMovesToDraw:324 \
--opening /volume/measure_strength/yaneuraou_start/start_sfens.txt \
--games 50 \
--byoyomi 2000 \
--csa /volume/param_optimize/kifu/resnet35x512_spsa_white \
--init_params "REPLACE_WITH_SPSA_TOTAL_RESULT" \
--suggest_params "C_init:100~200,C_base:20000~50000,C_fpu_reduction:0~40,C_init_root:100~200,C_base_root:20000~50000,Softmax_Temperature:100~200" \
--optimize_side white \
--spsa_iterations 30 \
--spsa_c_end 0.05 \
--spsa_r_end 0.01 \
--spsa_verify_games 0 \
--spsa_log /volume/param_optimize/log_spsa_white.jsonl \
--spsa_checkpoint /volume/param_optimize/spsa_white_checkpoint.json \
 | tee -a /volume/param_optimize/log_spsa_white.txt
