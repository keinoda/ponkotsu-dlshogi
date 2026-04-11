#!/bin/bash
# SPSAログをプロットしてMisskeyにノートするスクリプト
# Usage: bash utils/note_spsa_log.sh <access_token_path>

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <access_token_path>"
    exit 1
fi

ACCESS_TOKEN_PATH="$1"
LOG_FILE="/mnt/container/param_optimize/log_spsa_total.txt"
OUT_DIR="/tmp/spsa_plot"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON="/home/jj1guj/ponkotsu_wcsc33/.venv/bin/python"

mkdir -p "$OUT_DIR"
cd "$REPO_DIR"

# プロット生成 (出力先を/tmpに指定)
OUT_BASE="${OUT_DIR}/log_spsa_total"
$PYTHON utils/plot_spsa_log.py "$LOG_FILE" "${OUT_BASE}_plot.png"

PLOT_MAIN="${OUT_BASE}_plot.png"
PLOT_PARAMS="${OUT_BASE}_plot_params.png"
PLOT_SCATTER="${OUT_BASE}_plot_scatter.png"

# 画像アップロード
file_ids=()
for img in "$PLOT_MAIN" "$PLOT_PARAMS" "$PLOT_SCATTER"; do
    if [ ! -f "$img" ]; then
        echo "Warning: $img not found, skipping"
        continue
    fi
    response=$(curl -sS https://jiskey.dev/api/drive/files/create \
        --request POST \
        --header 'Content-Type: multipart/form-data' \
        --header "Authorization: Bearer $(cat "$ACCESS_TOKEN_PATH")" \
        --form 'isSensitive=false' \
        --form 'force=false' \
        -F "file=@${img}")
    fid=$(echo "$response" | jq -r '.id')
    if [ "$fid" != "null" ] && [ -n "$fid" ]; then
        file_ids+=("$fid")
    else
        echo "Warning: Failed to upload $img"
        echo "$response"
    fi
done

# ノート本文作成
text=$($PYTHON -c "
import re, ast
with open('$LOG_FILE') as f:
    text = f.read()
starts = [m.start() for m in re.finditer(r'SPSA optimization start', text)]
if starts:
    text = text[starts[-1]:]
blocks = re.split(r'={50,}', text)
completed = 0
for b in blocks:
    if re.search(r'Updated theta', b):
        completed += 1
m = re.search(r'Initial params: ({.*?})', text)
init_p = ast.literal_eval(m.group(1)) if m else {}
# 最新のUpdated theta
thetas = re.findall(r'Updated theta = ({.*?})', text)
current = ast.literal_eval(thetas[-1]) if thetas else init_p
# 最新のwin_rate
wr_plus = re.findall(r'win_rate\+=([\d.]+)', text)
wr_minus = re.findall(r'win_rate-=([\d.]+)', text)
lines = ['SPSA最適化ログ', f'完了: {completed} iterations']
if wr_plus:
    lines.append(f'最新 WR+: {float(wr_plus[-1])*100:.1f}%')
if wr_minus:
    lines.append(f'最新 WR-: {float(wr_minus[-1])*100:.1f}%')
lines.append(f'現在θ: {current}')
print('\n'.join(lines))
")

# fileIds JSON配列を構築
file_ids_json=$(printf '%s\n' "${file_ids[@]}" | jq -R . | jq -s .)

# ノート投稿 (jqでJSON構築)
json_body=$(jq -n \
    --arg text "$text" \
    --argjson fileIds "$file_ids_json" \
    '{
        localOnly: true,
        visibility: "specified",
        visibleUserIds: ["9gptzj80qf"],
        text: $text,
        fileIds: $fileIds
    }')

response=$(curl -sS https://jiskey.dev/api/notes/create \
    --request POST \
    --header 'Content-Type: application/json' \
    --header "Authorization: Bearer $(cat "$ACCESS_TOKEN_PATH")" \
    --data "$json_body")

echo "Response: $response"
echo "Note posted."
