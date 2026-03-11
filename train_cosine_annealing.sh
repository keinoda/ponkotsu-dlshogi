#!/bin/bash
last=370

# 変数設定
save_dir=$1
name=$2
checkpoint_dir="${save_dir}/${name}"
model_dir="${save_dir}/${name}/model"
log_dir="${save_dir}/${name}"
data_dir=$3
test_dir=$4
cache_dir=$5
compare_log_dir=$6

# 最新のチェックポイント+1から学習を再開する
for i in $(ls -v ${checkpoint_dir}/checkpoint_${name}-???.pth 2>/dev/null); do chkp=$i;
    done

if [ -v chkp ];then
    start=$(expr ${chkp: -7:3} + 1)
else
    start=300
fi

for ((i=$start; i<=$last; i++)); do
    iii=$(printf "%03d" $i)
    jjj=$(printf "%03d" $((i-1)))
    kkk=$(printf "%03d" $(((i-300) % 24 +1)))
    src="${data_dir}/floodgate_2019-20260304-${kkk} ${data_dir}/Suisho10Mn-${kkk} ${data_dir}/dlshogi_with_gct-${kkk}.hcpe ${data_dir}/suisho11alpha-20251006-${kkk}"

    # チェックポイントが存在する場合、最新のチェックポイントから学習を継続
    if [ $i -eq 300 ]; then
        resume="-r ${checkpoint_dir}/checkpoint_resnet35x512_prior-299.pth"
    else
        resume="-r ${checkpoint_dir}/checkpoint_${name}-${jjj}.pth"
    fi

    # モデルのファイル名
    model="${model_dir}/model_${name}-{epoch:03}.pth"

    # チェックポイントのファイル名
    checkpoint="${checkpoint_dir}/checkpoint_${name}-{epoch:03}.pth"

    echo epoch ${i} start

    if [ $i -eq 300 ]; then
        reset="--reset_scheduler --reset_optimizer"
    else
        reset=""
    fi

    if [ $i -ge 350 ]; then
        use_swa="--use_swa"
    else
        use_swa=""
    fi

    # 学習
    python -m dlshogi.train ${src} ${data_dir}/floodgate_test_2017-2018_r3500_eval5000.hcpe\
     ${resume} --checkpoint ${checkpoint} --network resnet35x512_fcl512 --model ${model} -e 1\
    --use_average --use_evalfix ${use_swa} --use_amp --amp_dtype bfloat16 --temperature 0 --lr 1e-4\
    --lr_scheduler dlshogi.lr_scheduler.CosineLRScheduler'('t_initial=271220,lr_min=1e-6,cycle_mul=2,cycle_limit=3,cycle_decay=0.8,warmup_t=67805,warmup_lr_init=1e-6,warmup_prefix=True')'\
     ${reset} --scheduler_step_mode step --cache ${cache_dir}/train_cache_${kkk} --log ${log_dir}/train_log.txt

    # ログのプロット
    python log_plot.py ${compare_log_dir}/train_log.txt ${log_dir}/train_log.txt

    # プロット結果をアップロード
    response=$(curl -s https://jiskey.dev/api/drive/files/create \
        --request POST \
        --header 'Content-Type: multipart/form-data' \
        --header "Authorization: Bearer $(cat ../jiskey_access_token)" \
        --form 'isSensitive=false' \
        --form 'force=false' \
        -F "file=@./loss_per_epoch.png")

    id_loss_per_epoch=$(echo $response | jq -r '.id')

    response=$(curl -s https://jiskey.dev/api/drive/files/create \
        --request POST \
        --header 'Content-Type: multipart/form-data' \
        --header "Authorization: Bearer $(cat ../jiskey_access_token)" \
        --form 'isSensitive=false' \
        --form 'force=false' \
        -F "file=@./accuracy_per_epoch.png")

    id_accuracy_per_epoch=$(echo $response | jq -r '.id')

    # 学習結果をノート
    text="$name\n"$(cat ${log_dir}/train_log.txt | grep "epoch = $i,"| tail -n 1 | cut -f 3)
    curl -s -o /dev/null https://jiskey.dev/api/notes/create \
        --request POST \
        --header 'Content-Type: application/json' \
        --header "Authorization: Bearer $(cat ../jiskey_access_token)" \
        --data '{
            "localOnly": true,
            "visibility": "specified",
            "visibleUserIds": ["9gptzj80qf"],
            "text": "'"$text"'",
            "fileIds": ["'"$id_loss_per_epoch"'", "'"$id_accuracy_per_epoch"'"]
        }'

    if [ $? -ne 0 ]; then
        break
    fi
done
