# vast.ai で ponkotsu (dlshogi) を動かす手順書

このリポジトリを vast.ai の GPU インスタンス上で動かし、学習・エンジン開発を行うための手順書です。
専用 Docker イメージ **`ghcr.io/keinoda/ponkotsu:vastai`** に必要なものは全て入っています。

> **重要**: 変更の push 先は常に `keinoda/ponkotsu-dlshogi` です。本家 (TadaoYamaoka / jj1guj) には絶対に push しないでください。dlshogi 本体のソースは `external/dlshogi` に取り込み済みで、このリポジトリ内だけで完結して改造できます。

---

## 0. クイックスタート(全体像)

```text
1. イメージを public にする(初回のみ)    → 1章
2. vast.ai でインスタンスを借りる          → 2章
3. ssh で入る                              → 3章
4. リポジトリを最新化                       → 4章
5. 学習データを転送                         → 5章
6. 学習を回す                              → 6章
7. エンジンを動かす                         → 7章
8. 成果物を回収して destroy                 → 8章
```

イメージに入っているもの:

| 種別 | 内容 |
| --- | --- |
| ベース | `nvcr.io/nvidia/tensorrt:25.06-py3` (CUDA 12.9 / TensorRT 10.12 / Python 3.12) |
| Python | PyTorch (stable, cu128), lightning, cshogi, optuna, scipy, matplotlib, japanize-matplotlib, onnx, jupyterlab |
| dlshogi | `external/dlshogi`(jj1guj/wcsc35 + 本家 master マージ済み)を `pip install` 済み |
| USI エンジン | ビルド済み: `/opt/ponkotsu-dlshogi/external/dlshogi/usi/bin/usi`(`dlshogi-usi` で PATH に登録済み) |
| リポジトリ | `/opt/ponkotsu-dlshogi`(.git 込み。`git pull` 可能) |
| その他 | YaneuraOu ソース (`/opt/YaneuraOu`)、tmux, fish, rsync, jq, jemalloc |

CUDA 12.9 ベースですが minor version compatibility により **ホストの driver が 525 以上(≒ CUDA 12.x 世代)なら動作**します。

---

## 1. 事前準備(初回のみ)

### 1-1. イメージの取得確認

イメージは GitHub Actions(`.github/workflows/docker-vastai.yml`)が自動ビルドして GHCR に push します。
`workflow_dispatch` でも手動実行できます(GitHub → Actions → "Build and Publish vast.ai Docker Image" → Run workflow)。

vast.ai が pull できるように、パッケージを **public** にします:

1. https://github.com/keinoda?tab=packages → `ponkotsu`
2. Package settings → Danger Zone → Change visibility → **Public**

public にしたくない場合は、vast.ai のテンプレート設定 **Docker Repository Authentication** に
GHCR の認証(ユーザー名 = GitHub ユーザー名、パスワード = `read:packages` 権限の PAT)を設定してください。

Docker Hub の Secrets(`DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN`)がリポジトリに設定されていれば、
`docker.io/<DOCKERHUB_USERNAME>/ponkotsu:vastai` にも push されます。こちらを使う場合は以降のイメージ名を読み替えてください。

### 1-2. vast.ai アカウント

1. https://vast.ai/ でアカウント作成、クレジットをチャージ
2. Account → SSH Keys に手元マシンの公開鍵(`~/.ssh/id_ed25519.pub` など)を登録

---

## 2. インスタンスの作成

### 2-1. テンプレート設定

Search 画面 → 左上の Template を編集(または New Template):

| 項目 | 設定値 |
| --- | --- |
| Image Path/Tag | `ghcr.io/keinoda/ponkotsu:vastai` |
| Version Tag | (再現性が必要なら `vastai-<コミットSHA>` を指定) |
| Launch Mode | **SSH**(`Run interactive shell server, SSH`) |
| On-start Script | (空でよい) |
| Docker Options | (空でよい) |

### 2-2. マシンの選び方(フィルタ)

| 条件 | 推奨値 | 理由 |
| --- | --- | --- |
| GPU | RTX 4090 / RTX 5090(お試し・エンジン検証)、A100/H100(本格学習) | resnet35x512 の学習は VRAM 24GB あれば動作(バッチサイズ調整) |
| CUDA(Max CUDA) | **12.4 以上** | イメージが CUDA 12.9 ベースのため(driver 525+ で動くが余裕を持たせる) |
| Disk Space | **100 GB 以上** | 学習データ + cache + チェックポイント(1 epoch 分で数 GB〜) |
| Internet Speed | 500 Mbps 以上 | データ転送・イメージ pull 時間の短縮 |
| Reliability | 99% 以上 | 長時間学習の中断リスク低減 |

料金体系: On-demand(確実)/ Interruptible(安いが横取りされうる)。長時間学習は On-demand 推奨。

> **注意**: インスタンスを **stop してもストレージ課金は続き**、GPU は他人に取られることがあります。作業単位で destroy まで行うのが基本です(→ 8章)。

### 2-3. 起動

Rent を押すと Instances 画面に現れ、イメージ pull(10〜20分程度。イメージは大きい)後に起動します。

---

## 3. 接続

Instances 画面の「>_」ボタンに表示される ssh コマンドで接続します:

```bash
ssh -p <PORT> root@<sshN.vast.ai>        # プロキシ経由
# または直接続 (Direct ssh connect と表示されるもの)
```

**最初に tmux を起動**してから作業してください(切断・再接続対策):

```bash
tmux new -s work    # 再接続時: tmux attach -t work
```

動作確認:

```bash
nvidia-smi                                   # GPU が見えるか
python3 -c "import torch; print(torch.cuda.is_available())"   # True
python3 -c "import dlshogi; print(dlshogi.__file__)"          # vendored 版が入っている
dlshogi-usi < /dev/null                      # USI エンジン(モデル未指定なので即終了で OK)
```

---

## 4. リポジトリの最新化・改造の反映

イメージ内の `/opt/ponkotsu-dlshogi` はビルド時点のスナップショットです。最新化:

```bash
cd /opt/ponkotsu-dlshogi
git pull origin <作業ブランチ>
```

(リポジトリが private の場合は `git remote set-url origin https://<GitHubユーザー名>:<PAT>@github.com/keinoda/ponkotsu-dlshogi` を先に実行)

**dlshogi 本体(`external/dlshogi`)を改造したら**、反映のために:

```bash
# Python 側(学習など)を反映
pip3 install ./external/dlshogi

# 以後も頻繁にいじるなら editable インストールにしておくと再インストール不要
pip3 install -e ./external/dlshogi

# USI エンジン(C++)を再ビルド
cd external/dlshogi/usi && make clean && make -j"$(nproc)"
```

インスタンス上で行った変更のコミット・push も可能です(push 先は必ず keinoda リポジトリ):

```bash
git add -A && git commit -m "..." && git push origin <作業ブランチ>
```

---

## 5. 学習データの転送

データ置き場は `/workspace` を推奨(慣例的に vast.ai のデータ領域):

```bash
mkdir -p /workspace/data /workspace/test /workspace/models /workspace/cache
```

### 5-1. 手元 → インスタンス(rsync 推奨)

手元マシンから:

```bash
rsync -avP -e "ssh -p <PORT>" ./hcpe_data/ root@<sshN.vast.ai>:/workspace/data/
```

hcpe は圧縮が効くので、事前に `pbzip2` で固めて送り、インスタンス側で展開すると速いことが多いです
(イメージに `pbzip2` / `pv` 導入済み)。

### 5-2. クラウドストレージ経由

大容量・繰り返し使う場合は vast.ai の **Cloud Sync**(Backblaze B2 / S3 / Google Drive)が便利です。
Instances 画面のクラウドアイコンから設定できます。次回以降のインスタンスへの展開が速くなります。

### 5-3. floodgate 等の公開データ

インスタンスに直接ダウンロードするのが最速です(`wget`/`curl` 導入済み)。

---

## 6. 学習の実行

### 6-1. 単発の学習コマンド(推奨: まずこれで動作確認)

`train_prior.sh` が内部で呼んでいるものと同じ形式です:

```bash
cd /opt/ponkotsu-dlshogi

python3 -m dlshogi.train \
    /workspace/data/<学習データ.hcpe...(複数可)> \
    /workspace/test/<テストデータ.hcpe> \
    --network resnet35x512_fcl512 \
    -e 1 \
    --use_average --use_evalfix --use_amp --amp_dtype bfloat16 \
    --temperature 0 --lr 0.2 \
    --lr_scheduler ReduceLROnPlateau'('eps=1e-20,factor=0.5')' --scheduler_step_mode epoch \
    --checkpoint '/workspace/models/checkpoint-{epoch:03}.pth' \
    --model '/workspace/models/model-{epoch:03}.pth' \
    --cache /workspace/cache/train_cache \
    --log /workspace/models/train_log.txt
```

- 初回はデータの cache 構築に時間がかかります(2回目以降は `--cache` により高速化)
- VRAM が足りない場合は `--batchsize`(デフォルト 1024)を下げる
- GPU 使用状況は別ペインで `watch -n 2 nvidia-smi`

### 6-2. 一括学習スクリプト

`train_prior.sh` / `train_cosine_annealing.sh` は「エポックごとにログ画像を生成して Misskey に通知する」
運用込みのスクリプトです。引数 9 個(`save_dir name data_dir test_dir cache_dir compare_log_dir misskey_base_url token_file visible_user_id`)を取ります:

```bash
bash train_prior.sh /workspace/models run1 /workspace/data /workspace/test \
    /workspace/cache /workspace/compare https://<misskey> /root/.misskey_token <user_id>
```

> **注意**: これらのスクリプトはデータセットのファイル名パターン(`aoba_p3200_2025-XXX` 等)や
> Misskey 通知がハードコードされています。Misskey に到達できない環境ではループが途中終了する
> 可能性があるため、**Misskey を使わない場合は 6-1 の直接実行を推奨**します。
> 自分のデータ構成に合わせて `src=` の行を書き換えてから使ってください。

### 6-3. 学習ログの可視化

```bash
python3 log_plot.py /workspace/compare/train_log.txt /workspace/models/train_log.txt
# → カレントに loss_per_epoch.png / accuracy_per_epoch.png が生成される
```

手元への回収は `rsync`(→ 8章)か、`jupyter lab --allow-root --ip 0.0.0.0` を立てて確認。

### 6-4. (参考)本家最新の Lightning ベース学習

今回の本家マージで `dlshogi.ptl`(PyTorch Lightning + `config.yaml`)系の改善が多数入っています。
`lightning` はインストール済みなので、`external/dlshogi/dlshogi/config.yaml` を編集して
`python3 -m dlshogi.ptl fit --config ...` の形式も利用できます(詳細は本家 README 参照)。

---

## 7. USI エンジン(検証・対局)

### 7-1. モデルを ONNX に変換

```bash
python3 -m dlshogi.convert_model_to_onnx /workspace/models/model-378.pth /workspace/models/model.onnx \
    --network resnet35x512_fcl512
```

### 7-2. エンジン起動

```bash
dlshogi-usi    # = /opt/ponkotsu-dlshogi/external/dlshogi/usi/bin/usi
```

USI プロトコルで `setoption name DNN_Model value /workspace/models/model.onnx` 等を設定して使います。
**初回起動時は TensorRT エンジンのビルド(serialize)で数分かかります**(`.serialized` ファイルが生成され、2回目以降は高速)。

今回のマージで本家の複数最適化プロファイル対応が入っています(TensorRT 10 API に移植済み。詳細は `external/README.md`)。

### 7-3. 強さ計測・パラメータ調整(任意)

`utils/measure_strength.sh` / `utils/param_optimize_spsa.sh` は対局相手(YaneuraOu 等)のパスが
ハードコードされているため、使う場合はスクリプト内のパスを環境に合わせて書き換えてください。
YaneuraOu のソースは `/opt/YaneuraOu` に同梱済みです(要ビルド: `cd /opt/YaneuraOu/source && make -j$(nproc) YANEURAOU_EDITION=YANEURAOU_ENGINE_NNUE`)。

---

## 8. 成果物の回収と撤収

### 8-1. 回収(手元マシンから実行)

```bash
rsync -avP -e "ssh -p <PORT>" root@<sshN.vast.ai>:/workspace/models/ ./models_backup/
```

コードの変更は git push で回収するのが確実です(→ 4章)。

### 8-2. 撤収

1. 成果物(モデル・チェックポイント・ログ)の回収を確認
2. コード変更を push したか確認
3. Instances 画面で **Destroy**(stop はストレージ課金が続く点に注意)

---

## 9. トラブルシューティング

| 症状 | 対処 |
| --- | --- |
| イメージが pull できない | GHCR パッケージが public か確認(→ 1-1)。または vast テンプレートに認証を設定 |
| `CUDA driver version is insufficient` | ホストの driver が古い。CUDA 12.4+ (できれば 12.8+) のマシンを選び直す |
| DataLoader がハング/`shm` エラー | `--dataloader_workers`(または num_workers)を減らす。vast は共有メモリがインスタンス RAM に依存するため RAM 多めのマシンを選ぶ |
| 学習が OOM | `--batchsize` を下げる。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` を設定(train_prior.sh は設定済み) |
| pip の dlshogi が PyPI 版に化けた | `pip3 install ./external/dlshogi` を再実行(requirements.txt の `dlshogi` 行は PyPI 版なので注意) |
| USI エンジンの初回応答が遅い | TensorRT の serialize 待ち。`.serialized` 生成後は高速 |
| ssh が切れて学習が死んだ | 必ず tmux 内で実行する。`tmux attach -t work` で復帰 |

---

## 10. イメージの更新フロー(開発サイクル)

1. 手元(または Claude)で `external/dlshogi` やスクリプトを変更して push
2. GitHub Actions が `docker/Dockerfile.vastai` を自動ビルド(`external/**` の変更で発火。それ以外は Actions から手動実行)
3. 新しいインスタンスは新イメージで起動。**既存インスタンスは `git pull` + `pip3 install ./external/dlshogi` + `make` で更新すれば再作成不要**

ローカルで Dockerfile を検証する場合:

```bash
# コンパイル検証のみ(軽い・PyTorch 層なし)
docker build -f docker/Dockerfile.vastai --target engine-test .

# フルビルド
docker build -f docker/Dockerfile.vastai -t ghcr.io/keinoda/ponkotsu:vastai .
```

本家 dlshogi の追従取り込み手順は `external/README.md` を参照してください。
