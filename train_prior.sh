last=378

# 変数設定
save_dir=$1
name=$2
checkpoint_dir="${save_dir}/${name}"
model_dir="${save_dir}/${name}/model"
log_dir="${save_dir}/${name}"
data_dir=$3
test_dir=$4
cache_dir=$5

if [ ! -d ${model_dir} ]; then
    mkdir -p ${model_dir}
fi

# 最新のチェックポイント+1から学習を再開する
for i in $(ls -v ${checkpoint_dir}/checkpoint_${name}-???.pth 2>/dev/null); do chkp=$i;
    done

if [ -v chkp ];then
    start=$(expr ${chkp: -7:3} + 1)
else
    start=1
fi

for ((i=$start; i<=$last; i++)); do
    iii=$(printf "%03d" $(((i-1) % 63 + 1)))
    jjj=$(printf "%03d" $(((i-1) % 53 + 300)))
    kkk=$(printf "%07d" $(((i-1) % 53 +115)))
    rrr=$(printf "%03d" $((i-1)))
    src="${data_dir}/aoba_p1600-${iii} ${data_dir}/aoba_p3200-${iii} ${data_dir}/hao-${iii} ${data_dir}/tanuki_20240730-${iii} ${data_dir}/suisho5_nyugyoku-${iii}"

    # チェックポイントが存在する場合、最新のチェックポイントから学習を継続
    if [ $i -eq 1 ]; then
        resume=""
    else
        resume="-r ${checkpoint_dir}/checkpoint_${name}-${rrr}.pth"
    fi

    # モデルのファイル名
    model="${model_dir}/model_${name}-{epoch:03}.pth"

    # チェックポイントのファイル名
    checkpoint="${checkpoint_dir}/checkpoint_${name}-{epoch:03}.pth"

    echo epoch ${i} start

    # 学習
    python -m dlshogi.train ${src} ${test_dir}/floodgate_test_2017-2018_r3500_eval5000.hcpe\
     ${resume} --checkpoint ${checkpoint} --network policy_value_network_pre_ln.PolicyValueNetwork --model ${model} -e 1\
    --optimizer mup.MuSGD'('momentum=0.9,nesterov=True')' --use_average --use_evalfix --use_amp --temperature 0 --lr 0.2\
    --lr_scheduler ReduceLROnPlateau'('eps=1e-20,factor=0.5')' --scheduler_step_mode epoch --cache ${cache_dir}/train_cache_prior_${iii} --log ${log_dir}/train_log.txt

    # 学習結果をノート
    text="$name\n"$(cat ${log_dir}/train_log.txt | grep "epoch = $i,"| tail -n 1 | cut -f 3)
    curl -XPOST -H 'Content-Type:application/json' -d "{\"i\":\"$(cat ../jiskey_access_token)\",\"localOnly\":true,\"visibility\":\"specified\",\"visibleUserIds\":[\"9gptzj80qf\"],\"text\":\"$text\"}" https://jiskey.dev/api/notes/create
    echo \n

    
    if [ $? -ne 0 ]; then
        break
    fi
done
