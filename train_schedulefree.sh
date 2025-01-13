last=144

# 変数設定
save_dir=$1
name=$2
checkpoint_dir="${save_dir}/${name}"
model_dir="${save_dir}/${name}/model"
log_dir="${save_dir}/${name}"
data_dir=$3
network=$4
optimizer=$5
lr=$6

# 保存先のディレクトリが存在しなかったら作成する
if [ ! -d ${checkpoint_dir} ]; then
    mkdir -p ${checkpoint_dir}
fi

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
    iii=$(printf "%03d" $i)
    jjj=$(printf "%03d" $((i-1)))
    kkk=$(printf "%03d" $(((i-1) % 24 +1)))
    src="${data_dir}/floodgate_2019-2021_r3500-${kkk}.hcpe ${data_dir}/suisho3kai-${kkk}.hcpe ${data_dir}/dlshogi_with_gct-${kkk}.hcpe"

    # チェックポイントが存在する場合、最新のチェックポイントから学習を継続
    if [ $i -eq 1 ]; then
        resume=""
    else
        resume="-r ${checkpoint_dir}/checkpoint_${name}-${jjj}.pth"
    fi

    # モデルのファイル名
    model="${model_dir}/model_${name}-{epoch:03}.pth"

    # チェックポイントのファイル名
    checkpoint="${checkpoint_dir}/checkpoint_${name}-{epoch:03}.pth"

    echo epoch ${i} start

    if [ $(($i % 144)) -eq 0 ]; then
        use_swa="--use_swa"
    else
        use_swa=""
    fi

    # 学習
    python -m dlshogi.train ${src} ${data_dir}/floodgate_test_2017-2018_r3500_eval5000.hcpe\
     ${resume} --checkpoint ${checkpoint} --network $network --model ${model} -e 1\
    --optimizer ${optimizer} \
    --use_average --use_evalfix ${use_swa} --use_amp --temperature 0 --lr ${lr}\
    --cache ../train_cache/train_cache_${kkk} --log ${log_dir}/train_log.txt

    # 6エポックごとに学習結果をノート
    if [ $(($i % 6)) -eq 0 ]; then
        text="$name\n"$(cat ${log_dir}/train_log.txt | grep "epoch = $i,"| tail -n 1 | cut -f 3)
        curl -XPOST -H 'Content-Type:application/json' -d "{\"i\":\"$(cat ../jiskey_access_token)\",\"localOnly\":true,\"visibility\":\"specified\",\"visibleUserIds\":[\"9gptzj80qf\"],\"text\":\"$text\"}" https://jiskey.dev/api/notes/create
        echo \n
    fi

    if [ $? -ne 0 ]; then
        break
    fi
done
