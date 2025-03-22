last=600

# 変数設定
save_dir=$1
name=$2
checkpoint_dir="${save_dir}/${name}"
model_dir="${save_dir}/${name}/model"
log_dir="${save_dir}/${name}"
data_dir=$3
test_dir=$4
cache_dir=$5

# 最新のチェックポイント+1から学習を再開する
for i in $(ls -v ${checkpoint_dir}/checkpoint_${name}-???.pth 2>/dev/null); do chkp=$i;
    done

if [ -v chkp ];then
    start=$(expr ${chkp: -7:3} + 1)
else
    start=344
fi

for ((i=$start; i<=$last; i++)); do
    iii=$(printf "%03d" $i)
    jjj=$(printf "%03d" $((i-1)))
    kkk=$(printf "%03d" $(((i-1) % 24 +1)))
    src="${data_dir}/floodgate_2019-20250322-${kkk} ${data_dir}/Suisho10Mn-${kkk} ${data_dir}/dlshogi_with_gct-${kkk}.hcpe"

    # チェックポイントが存在する場合、最新のチェックポイントから学習を継続
    if [ $i -eq 344 ]; then
        resume="-r ${checkpoint_dir}/checkpoint_resnet30x256_pre_ln_prior-343.pth"
    else
        resume="-r ${checkpoint_dir}/checkpoint_${name}-${jjj}.pth"
    fi

    # モデルのファイル名
    model="${model_dir}/model_${name}-{epoch:03}.pth"

    # チェックポイントのファイル名
    checkpoint="${checkpoint_dir}/checkpoint_${name}-{epoch:03}.pth"

    echo epoch ${i} start

    if [ $i -eq 344 ]; then
        reset="--reset_scheduler --reset_optimizer"
    else
        reset=""
    fi

    if [ $i -eq 344 ]; then
        use_swa=""
    else
        LATEST_LR=$(grep "lr_scheduler lr=" "${log_dir}/train_log.txt" | tail -n 1 | awk -F'=' '{print $2}')
        LATEST_LR=$(printf "%.10f" "$LATEST_LR")
        if (( $(echo "$LATEST_LR <= 0.00004" | bc -l) )); then
            use_swa="--use_swa"
        else
            use_swa=""
        fi
    fi

    # 学習
    python -m dlshogi.train ${src} ${data_dir}/floodgate_test_2017-2018_r3500_eval5000.hcpe\
     ${resume} --checkpoint ${checkpoint} --network policy_value_network_pre_ln.PolicyValueNetwork --model ${model} -e 1\
    --optimizer mup.MuSGD'('momentum=0.9,nesterov=True')' --use_average --use_evalfix ${use_swa} --use_amp --temperature 0 --lr 0.001\
    --lr_scheduler ReduceLROnPlateau'('eps=1e-20,factor=0.5')' ${reset} --scheduler_step_mode epoch --cache ${cache_dir}/train_cache_${kkk} --log ${log_dir}/train_log.txt

    # 学習結果をノート
    text="$name\n"$(cat ${log_dir}/train_log.txt | grep "epoch = $i,"| tail -n 1 | cut -f 3)
    curl -XPOST -H 'Content-Type:application/json' -d "{\"i\":\"$(cat ../jiskey_access_token)\",\"localOnly\":true,\"visibility\":\"specified\",\"visibleUserIds\":[\"9gptzj80qf\"],\"text\":\"$text\"}" https://jiskey.dev/api/notes/create
    echo \n
    
    if [ $? -ne 0 ]; then
        break
    fi
done
