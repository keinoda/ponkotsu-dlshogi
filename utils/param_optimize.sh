#!/bin/sh

python -u -m dlshogi.utils.usi_params_optimizer --study_name model_resnet35x512 --storage "sqlite:////volume/param_optimize/model_resnet35x512.db" \
/root/YaneuraOu/source/FukauraOu-by-gcc /root/yaneuraou-V921-dev-mac-all/source/YaneuraOu-by-gcc \
--options1 PV_Interval:0,PV_Mate_Search_Threads:3,DNN_Model:/root/models/model_resnet35x512-348.onnx,NetworkDelay2:0,MaxMovesToDraw:324 \
--options2 PV_Interval:0,NetworkDelay2:0,Threads:2,USI_Hash:256,EvalDir:/volume/book/suisho11b,FV_SCALE:28,MaxMovesToDraw:324 \
--opening /volume/measure_strength/yaneuraou_start/start_sfens.txt --games 100 --trials 150 --byoyomi 2000 --csa /volume/param_optimize/kifu/resnet35x512 \
 | tee -a /volume/param_optimize/log_optimize_model_resnet35x512.txt
