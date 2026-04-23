python -u -m cshogi.cli \
/root/YaneuraOu/source/FukauraOu-by-gcc \
/root/YaneuraOu/source/FukauraOu-by-gcc \
/root/YaneuraOu/source/FukauraOu-by-gcc \
--name1 ponkotsu_wcsc36_optuna \
--name2 ponkotsu_wcsc36_spsa \
--name3 ponkotsu_wcsc35 \
--options1 PV_Interval:0,PV_Mate_Search_Threads:3,DNN_Model:/root/models/model_resnet35x512-348.onnx,NetworkDelay2:0,UCT_Threads:2,MaxMovesToDraw:324,C_init:101,C_base:21657,C_fpu_reduction:17,C_init_root:100,C_base_root:20163,Softmax_Temperature:195 \
--options2 PV_Interval:0,PV_Mate_Search_Threads:3,DNN_Model:/root/models/model_resnet35x512-348.onnx,NetworkDelay2:0,UCT_Threads:2,MaxMovesToDraw:324,C_init:110,C_base:26556,C_fpu_reduction:12,C_init_root:150,C_base_root:47039,Softmax_Temperature:172 \
--options3 PV_Interval:0,PV_Mate_Search_Threads:3,DNN_Model:/root/models/model_resnet30x256_pre_ln_cos_anealing_1e-4-407.onnx,NetworkDelay2:0,UCT_Threads:2,MaxMovesToDraw:324,C_init:145,C_base:36415,C_fpu_reduction:15,C_init_root:120,C_base_root:28756,Softmax_Temperature:178 \
--opening /volume/measure_strength/yaneuraou_start/start_sfens.txt --draw 320 --time 120000 --inc 2000 --games 200 \
--csa /volume/measure_strength/kifu/league_optuna_spsa_wcsc35 --pgn /volume/measure_strength/league_optuna_spsa_wcsc35 \
| tee log_league_optuna_spsa_wcsc35.txt
