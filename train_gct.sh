last=364

# 変数設定
save_dir=$1
name="densenet10_g32_c192_add_options_kernel3_adding_gct"
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
    start=251
fi

for ((i=$start; i<=$last; i++)); do
    iii=$(printf "%03d" $(((i-251)/3*2+1)))
    jjj=$(printf "%03d" $(((i-251)/3*2+2)))
    kkk=$(printf "%03d" $(((i-251)/3*3+3)))
    rrr=$(printf "%03d" $((i-1)))
    # src="${data_dir}/selfplay_gct-${iii}.hcpe3 ${data_dir}/selfplay_gct-${jjj}.hcpe3 ${data_dir}/selfplay_gct-${kkk}.hcpe3"
    src="${data_dir}/selfplay_gct-${iii}.hcpe3"

    if [ $i -ge $((last-2)) ];then
        src="${src} ${data_dir}/nyugyoku"
    else
        src="${src} ${data_dir}/selfplay_gct-${jjj}.hcpe3"
    fi

    # lr="--lr 0.0002 --lr_scheduler MultiStepLR(milestones=[10,30],gamma=0.1)"

    # 応急処置で一旦
    # lr="--lr 0.0002 --lr_scheduler MultiStepLR(milestones=[0,20],gamma=0.1)"
    if [ $i -ge 280 ]; then
        lr="--lr 0.000002 --reset_scheduler"
    else
        lr="--lr 0.00002 --reset_scheduler"
    fi
    # if [ $i -eq 261 ]; then
    #     lr="${lr} --reset_scheduler"
    # fi

    # チェックポイントが存在する場合、最新のチェックポイントから学習を継続
    if [ $i -eq 251 ]; then
        resume="-r ${checkpoint_dir}/checkpoint_densenet10_g32_c192_add_options_kernel3_adding_no-reset-250.pth"
        lr="${lr} --reset_scheduler"
    else
        resume="-r ${checkpoint_dir}/checkpoint_${name}-${rrr}.pth"
    fi

    # モデルのファイル名
    model="${model_dir}/model_${name}-{epoch:03}.pth"

    # チェックポイントのファイル名
    checkpoint="${checkpoint_dir}/checkpoint_${name}-{epoch:03}.pth"

    echo epoch $i start

    # 学習
    python -m dlshogi.train ${src} ${data_dir}/floodgate_test_2017-2018_r3500_eval5000.hcpe\
     ${resume} --checkpoint ${checkpoint} --network policy_value_network_densenet.PolicyValueNetwork --model ${model} -e 1\
    --use_average --use_evalfix --use_swa --use_amp --temperature 1 ${lr} --log ${log_dir}/train_log.txt

    
    if [ $? -ne 0 ]; then
        break
    fi
done
