#!/bin/sh

# Phase 3: 範囲縮小TPE (Phase 1+2の知見に基づき探索範囲を大幅に絞り込み)
# Phase 2の結果: Temp 200超は悪化を確認 → 170~200に絞る
# 変更点 (Phase 1の範囲との比較):
#   Softmax_Temperature: 100~200 → 170~200 (200超は悪化、170未満も低調)
#   C_fpu_reduction: 0~40 → 0~20 (20超は低調)
#   C_base_root: 20000~50000 → 18000~30000 (30k超は低調)
#   C_init: 100~200 → 100~160 (100-129帯が最良)
#   C_init_root: 100~200 → 100~170 (Top10 maxが170)
#   C_base: 20000~50000 → 変更なし
# 初期値: Phase 1 Top-5 median

python -u -m dlshogi.utils.usi_params_optimizer --study_name model_resnet35x512_phase3 --storage "sqlite:////volume/param_optimize/model_resnet35x512_phase3.db" \
/root/YaneuraOu/source/FukauraOu-by-gcc /root/yaneuraou-V921-dev-mac-all/source/YaneuraOu-by-gcc \
--options1 PV_Interval:0,PV_Mate_Search_Threads:3,DNN_Model:/root/models/model_resnet35x512-348.onnx,NetworkDelay2:0,MaxMovesToDraw:324 \
--options2 PV_Interval:0,NetworkDelay2:0,Threads:2,USI_Hash:256,EvalDir:/volume/book/suisho11b,FV_SCALE:28,MaxMovesToDraw:324 \
--init_params "C_init:123,C_base:32303,C_fpu_reduction:14,C_init_root:155,C_base_root:26661,Softmax_Temperature:173" \
--suggest_params "C_init:100~160,C_base:20000~50000,C_fpu_reduction:0~20,C_init_root:100~170,C_base_root:18000~30000,Softmax_Temperature:170~200" \
--opening /volume/measure_strength/yaneuraou_start/start_sfens.txt --games 100 --trials 50 --byoyomi 2000 --csa /volume/param_optimize/kifu/resnet35x512_phase3 \
 | tee -a /volume/param_optimize/log_optimize_model_resnet35x512_phase3.txt
