#!/bin/sh

# Phase 2: 範囲調整TPE (Phase 1の結果に基づき探索範囲を再設計)
# 変更点:
#   Softmax_Temperature: 100~200 → 160~300 (上限拡張、最も勝率に寄与)
#   C_fpu_reduction: 0~40 → 0~25 (20以上は低調)
#   C_base_root: 20000~50000 → 18000~35000 (40k以上は低調)
#   C_init: 100~200 → 100~170 (160以上は低調)
#   C_init_root: 100~200 → 100~175 (高い方ほど低調)
#   C_base: 20000~50000 → 変更なし

python -u -m dlshogi.utils.usi_params_optimizer --study_name model_resnet35x512_phase2 --storage "sqlite:////volume/param_optimize/model_resnet35x512_phase2.db" \
/root/YaneuraOu/source/FukauraOu-by-gcc /root/yaneuraou-V921-dev-mac-all/source/YaneuraOu-by-gcc \
--options1 PV_Interval:0,PV_Mate_Search_Threads:3,DNN_Model:/root/models/model_resnet35x512-348.onnx,NetworkDelay2:0,MaxMovesToDraw:324 \
--options2 PV_Interval:0,NetworkDelay2:0,Threads:2,USI_Hash:256,EvalDir:/volume/book/suisho11b,FV_SCALE:28,MaxMovesToDraw:324 \
--suggest_params "C_init:100~170,C_base:20000~50000,C_fpu_reduction:0~25,C_init_root:100~175,C_base_root:18000~35000,Softmax_Temperature:160~300" \
--opening /volume/measure_strength/yaneuraou_start/start_sfens.txt --games 100 --trials 50 --byoyomi 2000 --csa /volume/param_optimize/kifu/resnet35x512_phase2 \
 | tee -a /volume/param_optimize/log_optimize_model_resnet35x512_phase2.txt
