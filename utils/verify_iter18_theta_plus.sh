#!/bin/sh

# Iter 18 θ+ パラメータの検証対局 (200局 vs 水匠11β, 2秒秒読み)

python -u -m cshogi.cli \
/root/YaneuraOu/source/FukauraOu-by-gcc \
/root/yaneuraou-V921-dev-mac-all/source/YaneuraOu-by-gcc \
--name1 ponkotsu_iter18_theta_plus \
--name2 suisho11b \
--options1 PV_Interval:0,PV_Mate_Search_Threads:3,DNN_Model:/root/models/model_resnet35x512-348.onnx,NetworkDelay2:0,MaxMovesToDraw:324,C_init:108,C_base:46029,C_fpu_reduction:24,C_init_root:134,C_base_root:28340,Softmax_Temperature:169 \
--options2 PV_Interval:0,NetworkDelay2:0,Threads:2,USI_Hash:256,EvalDir:/volume/book/suisho11b,FV_SCALE:28,MaxMovesToDraw:324 \
--opening /volume/measure_strength/yaneuraou_start/start_sfens.txt \
--byoyomi 2000 \
--draw 320 \
--games 200 \
--csa /volume/param_optimize/kifu/verify_iter18_theta_plus \
| tee log_verify_iter18_theta_plus.txt
