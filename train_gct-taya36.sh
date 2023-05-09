last=380

# 変数設定
save_dir=$1
name="densenet20"
checkpoint_dir="${save_dir}/${name}"
model_dir="${save_dir}/${name}/model"
log_dir="${save_dir}/${name}"
data_dir=$2

# 最新のチェックポイント+1から学習を再開する
for i in $(ls -v ${checkpoint_dir}/checkpoint_${name}-???.pth 2>/dev/null); do chkp=$i;
    done

if [ -v chkp ];then
    start=$(expr ${chkp: -7:3} + 1)
else
    start=1
fi

for ((i=$start; i<=$last; i++)); do
    iii=$(((i-375)/2))
    rrr=$(printf "%03d" $((i-1)))

    if [ $iii -eq 0 ];then
        src="${data_dir}/selfplay_model-0000221_taya36.hcpe3"
    elif [ $iii -eq 1 ];then
        src="${data_dir}/selfplay_model-0000224_taya36.hcpe3"
    else
        src="${data_dir}/selfplay_gct070_taya36.hcpe3"
    fi

    # チェックポイントが存在する場合、最新のチェックポイントから学習を継続
    if [ $i -eq 375 ]; then
        resume="-r /home/vmlab/jj1guj/densenet10_g32_c192_add_options_kernel3_adding_gct_model/checkpoint_densenet10_g32_c192_add_options_kernel3_adding_gct_model-374.pth"
    else
        resume="-r ${checkpoint_dir}/checkpoint_${name}-${rrr}.pth"
    fi

    # モデルのファイル名
    model="${model_dir}/model_${name}-{epoch:03}.pth"

    # チェックポイントのファイル名
    checkpoint="${checkpoint_dir}/checkpoint_${name}-{epoch:03}.pth"

    echo epoch $i start

    # 学習
    python -m dlshogi.train ${src} ${test_dir}/floodgate_test_2017-2018_r3500_eval5000.hcpe\
     ${resume} --checkpoint ${checkpoint} --network policy_value_network_densenet.PolicyValueNetwork --model ${model} -e 1\
     --use_average --use_evalfix --use_swa --use_amp --temperature 1 --lr 0.2\
     --lr_scheduler ReduceLROnPlateau --log ${log_dir}/train_log.txt

    
    if [ $? -ne 0 ]; then
        break
    fi
done
