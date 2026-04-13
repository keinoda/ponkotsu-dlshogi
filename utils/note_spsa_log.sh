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
blocks = re.split(r'={50,}', text)
completed = 0
for b in blocks:
    if re.search(r'Updated theta', b):
        completed += 1
inits = re.findall(r'Initial params: ({.*?})', text)
init_p = ast.literal_eval(inits[-1]) if inits else {}
# 最新のUpdated theta
thetas = re.findall(r'Updated theta = ({.*?})', text)
current = ast.literal_eval(thetas[-1]) if thetas else init_p

def extract_live_wr(log_text):
    # 直近で進行中の対局種別を特定し、そのセクションから途中勝率を拾う
    markers = list(re.finditer(r'Playing theta\+ match \(\d+ games\)\.\.\.|Playing theta- match \(\d+ games\)\.\.\.|Playing theta\+ vs theta- match \(\d+ games\)\.\.\.', log_text))
    if not markers:
        return None, None, False, False

    last = markers[-1]
    sec = log_text[last.start():]
    m_rate = re.findall(r'^.* vs .*: \d+-\d+-\d+ \(([\d.]+)%\)$', sec, flags=re.MULTILINE)
    if not m_rate:
        return None, None, False, False

    latest_pct = float(m_rate[-1])
    marker = last.group(0)
    if 'theta+ vs theta-' in marker:
        return latest_pct, 100.0 - latest_pct, True, True
    if 'theta+ match' in marker:
        return latest_pct, None, True, False
    if 'theta- match' in marker:
        return None, latest_pct, False, True
    return None, None, False, False

def extract_last_completed_wr(log_text):
    wr = re.findall(r'win_rate\+=([\d.]+), win_rate-=([\d.]+)', log_text)
    if not wr:
        return None, None
    p, m = wr[-1]
    return float(p) * 100.0, float(m) * 100.0

base_plus, base_minus = extract_last_completed_wr(text)
live_plus, live_minus, plus_live, minus_live = extract_live_wr(text)

disp_plus = live_plus if plus_live and live_plus is not None else base_plus
disp_minus = live_minus if minus_live and live_minus is not None else base_minus

lines = ['SPSA最適化ログ', f'完了: {completed} iterations']
if disp_plus is not None:
    suffix = ' (進行中)' if plus_live and live_plus is not None else ''
    lines.append(f'最新 WR+: {disp_plus:.1f}%{suffix}')
if disp_minus is not None:
    suffix = ' (進行中)' if minus_live and live_minus is not None else ''
    lines.append(f'最新 WR-: {disp_minus:.1f}%{suffix}')
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
