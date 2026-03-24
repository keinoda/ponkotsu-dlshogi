python -u -m cshogi.cli /root/YaneuraOu/source/FukauraOu-by-gcc \
/root/yaneuraou-V921-dev-mac-all/source/YaneuraOu-by-gcc --name1 ponkotsu_wcsc36 --name2 suisho11b_1t \
--options1 PV_Interval:0,PV_Mate_Search_Threads:3,DNN_Model:/root/models/model_resnet35x512-348.onnx,NetworkDelay2:0,MaxMovesToDraw:324 \
--options2 PV_Interval:0,NetworkDelay2:0,Threads:2,USI_Hash:256,EvalDir:/volume/book/suisho11b,FV_SCALE:28,MaxMovesToDraw:324 \
--opening /volume/measure_strength/yaneuraou_start/start_sfens.txt --draw 320 --byoyomi 2000 --games 200 \
--csa /volume/param_optimize/kifu/wcsc36default_vs_suisho11b --pgn /volume/measure_strength/wcsc36_vs_suisho11b \
| tee log_wcsc36_vs_suisho11b.txt
