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
text=$(LOG_FILE="$LOG_FILE" "$PYTHON" - <<'PY'
import ast
import os
import re

with open(os.environ['LOG_FILE']) as f:
    text = f.read()

blocks = re.split(r'={50,}', text)
completed = 0
for b in blocks:
    if re.search(r'Updated theta', b):
        completed += 1

iteration_blocks = [b for b in blocks if re.search(r'Iteration \d+/\d+', b)]
current_block = iteration_blocks[-1] if iteration_blocks else text

inits = re.findall(r'Initial params: ({.*?})', text)
init_p = ast.literal_eval(inits[-1]) if inits else {}
thetas = re.findall(r'Updated theta = ({.*?})', text)
current = ast.literal_eval(thetas[-1]) if thetas else init_p

def extract_section(block_text, start_token, end_tokens):
    start = block_text.find(start_token)
    if start == -1:
        return None

    end_positions = []
    for token in end_tokens:
        pos = block_text.find(token, start + len(start_token))
        if pos != -1:
            end_positions.append(pos)
    end = min(end_positions) if end_positions else len(block_text)
    return block_text[start:end]

def extract_section_pct(section_text):
    if not section_text:
        return None, False

    games = re.findall(r'(\d+) of (\d+) games finished\.\n.*?\(([\d.]+)%\)', section_text, re.DOTALL)
    if not games:
        return None, False

    finished, total, pct = games[-1]
    return float(pct), int(finished) >= int(total)

def extract_last_completed_wr(log_text):
    wr_sep = re.findall(r'win_rate\+=([\d.]+), win_rate-=([\d.]+)', log_text)
    if not wr_sep:
        return None, None
    p, m = wr_sep[-1]
    return float(p) * 100.0, float(m) * 100.0

plus_section = extract_section(current_block, 'Playing theta+ match', ['Playing theta- match', 'Playing theta+ vs theta- match'])
minus_section = extract_section(current_block, 'Playing theta- match', ['Playing theta+ vs theta- match'])
pm_section = extract_section(current_block, 'Playing theta+ vs theta- match', [])

plus_pct, plus_done = extract_section_pct(plus_section)
minus_pct, minus_done = extract_section_pct(minus_section)
pm_pct, pm_done = extract_section_pct(pm_section)

base_plus, base_minus = extract_last_completed_wr(text)

phase_positions = []
for phase_name, token in [
    ('plus', 'Playing theta+ match'),
    ('minus', 'Playing theta- match'),
    ('pm', 'Playing theta+ vs theta- match'),
]:
    pos = current_block.rfind(token)
    if pos != -1:
        phase_positions.append((pos, phase_name))

current_phase = max(phase_positions)[1] if phase_positions else None

disp_plus = plus_pct if plus_pct is not None else base_plus
disp_minus = minus_pct if minus_pct is not None else base_minus
plus_live = False
minus_live = False

if current_phase == 'plus' and plus_pct is not None and not plus_done:
    disp_plus = plus_pct
    plus_live = True
elif current_phase == 'minus':
    if plus_pct is not None:
        disp_plus = plus_pct
    if minus_pct is not None:
        disp_minus = minus_pct
        minus_live = not minus_done
elif current_phase == 'pm' and pm_pct is not None:
    disp_plus = pm_pct
    disp_minus = 100.0 - pm_pct
    plus_live = not pm_done
    minus_live = not pm_done

lines = ['SPSA最適化ログ', f'完了: {completed} iterations']
if disp_plus is not None:
    suffix = ' (進行中)' if plus_live else ''
    lines.append(f'最新 WR+: {disp_plus:.1f}%{suffix}')
if disp_minus is not None:
    suffix = ' (進行中)' if minus_live else ''
    lines.append(f'最新 WR-: {disp_minus:.1f}%{suffix}')
lines.append(f'現在θ: {current}')
print('\n'.join(lines))
PY
)

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
