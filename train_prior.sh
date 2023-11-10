last=106

# 変数設定
save_dir=$1
name="resnet50x192"
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
    iii=$(printf "%03d" $(((i-1) % 53 + 1)))
    jjj=$(printf "%03d" $(((i-1) % 53 + 300)))
    kkk=$(printf "%07d" $(((i-1) % 53 +115)))
    rrr=$(printf "%03d" $((i-1)))
    src="${data_dir}/hcpe/elmo_teacher_shuffle-${iii} ${data_dir}/hcpe/selfplay-${jjj} ${data_dir}/hcpe/aobazero_teacher-arch${kkk}"
    if [ $((i % 53)) -eq 0 ];then
        src="${src} ${data_dir}/suisho/hcpe/nyugyoku"
    fi
    echo ${src}

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
    python -m dlshogi.train ${src} ${data_dir}/floodgate_test_2017-2018_r3500_eval5000.hcpe\
     ${resume} --checkpoint ${checkpoint} --network resnet50x192 --model ${model} -e 1\
    --use_average --use_evalfix --use_swa --use_amp --temperature 0 --lr 0.2\
    --lr_scheduler ReduceLROnPlateau'('eps=1e-20,factor=0.5')' --log ${log_dir}/train_log.txt

    
    if [ $? -ne 0 ]; then
        break
    fi
done
