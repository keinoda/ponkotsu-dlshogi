#!/usr/bin/env python3
"""SPSAパラメータ最適化のテキストログをプロットするスクリプト

使い方:
    python utils/plot_spsa_log.py log_spsa_total.txt [出力画像パス]

テキストログ (tee出力) とJSONLログの両方に対応。
JSONLが存在すればそちらを優先し、なければテキストログからパースする。
"""

import re
import sys
import json
import ast
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

PARAM_NAMES = ["C_init", "C_base", "C_fpu_reduction", "C_init_root", "C_base_root", "Softmax_Temperature"]


def parse_jsonl(path):
    """JSONLログをパース"""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_text_log(path):
    """テキストログ(tee出力)からSPSA各イテレーションの情報を抽出

    最後の有効な実行のみを使用 (再起動があった場合)
    """
    with open(path) as f:
        text = f.read()

    # 最後の "SPSA optimization start" から使用
    starts = [m.start() for m in re.finditer(r'SPSA optimization start', text)]
    if starts:
        text = text[starts[-1]:]

    # 初期パラメータ
    m = re.search(r'Initial params: ({.*?})', text)
    init_params = ast.literal_eval(m.group(1)) if m else {}

    records = []
    # イテレーションブロックを分割
    iter_blocks = re.split(r'={50,}', text)

    for block in iter_blocks:
        m_iter = re.search(r'Iteration (\d+)/(\d+)\s+t=([\d.]+)\s+c_k=([\d.]+)\s+r_k=([\d.]+)', block)
        if not m_iter:
            continue
        iteration = int(m_iter.group(1))
        t = float(m_iter.group(3))
        c_k = float(m_iter.group(4))
        r_k = float(m_iter.group(5))

        m_tp = re.search(r'theta\+ = ({.*?})', block)
        m_tm = re.search(r'theta- = ({.*?})', block)
        theta_plus = ast.literal_eval(m_tp.group(1)) if m_tp else {}
        theta_minus = ast.literal_eval(m_tm.group(1)) if m_tm else {}

        # 勝率の抽出
        wr_matches = re.findall(
            r'\[total\] win_rate([+\-])=([\d.]+), win_rate([+\-])=([\d.]+)',
            block
        )
        win_rate_plus = None
        win_rate_minus = None
        if wr_matches:
            for m in wr_matches:
                if m[0] == '+':
                    win_rate_plus = float(m[1])
                    win_rate_minus = float(m[3])
                elif m[0] == '-':
                    win_rate_minus = float(m[1])
                    win_rate_plus = float(m[3])

        # dev-vs-dev
        m_pm = re.search(r'win_rate\(\+vs-\)=([\d.]+)', block)
        win_rate_pm = float(m_pm.group(1)) if m_pm else None

        # score_diff
        score_diff = None
        if win_rate_plus is not None and win_rate_minus is not None:
            score_diff = win_rate_plus - win_rate_minus
            if win_rate_pm is not None:
                score_diff += (win_rate_pm - 0.5)

        # Updated theta
        m_ut = re.search(r'Updated theta = ({.*?})', block)
        theta = ast.literal_eval(m_ut.group(1)) if m_ut else None

        if theta is None:
            continue

        records.append({
            'iteration': iteration,
            'theta': theta,
            'theta_plus': theta_plus,
            'theta_minus': theta_minus,
            'win_rate_plus': win_rate_plus,
            'win_rate_minus': win_rate_minus,
            'win_rate_pm': win_rate_pm,
            'score_diff': score_diff,
            'c_k': c_k,
            'r_k': r_k,
        })

    return records, init_params


def plot_spsa(records, init_params, output_path):
    if not records:
        print("No completed iterations found.")
        return

    iters = [r['iteration'] for r in records]
    n = len(iters)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(f"SPSA パラメータ最適化 ({n} iterations完了)", fontsize=16, y=0.98)

    # --- 1. 勝率推移 (θ+ vs baseline, θ- vs baseline) ---
    ax = axes[0, 0]
    wr_plus = [r['win_rate_plus'] * 100 for r in records if r['win_rate_plus'] is not None]
    wr_minus = [r['win_rate_minus'] * 100 for r in records if r['win_rate_minus'] is not None]
    iters_wr = [r['iteration'] for r in records if r['win_rate_plus'] is not None]
    ax.plot(iters_wr, wr_plus, 'o-', color='steelblue', label='θ+ vs baseline', markersize=4)
    ax.plot(iters_wr, wr_minus, 's-', color='coral', label='θ- vs baseline', markersize=4)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('勝率 (%)')
    ax.set_title('θ± vs Baseline 勝率推移')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 2. Dev-vs-dev 勝率推移 ---
    ax = axes[0, 1]
    pm_data = [(r['iteration'], r['win_rate_pm'] * 100) for r in records if r['win_rate_pm'] is not None]
    if pm_data:
        pm_iters, pm_wr = zip(*pm_data)
        ax.plot(pm_iters, pm_wr, 'D-', color='green', markersize=4)
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('θ+ 勝率 (%)')
    ax.set_title('Dev-vs-Dev (θ+ vs θ-)')
    ax.grid(True, alpha=0.3)

    # --- 3. score_diff 推移 ---
    ax = axes[1, 0]
    sd_data = [(r['iteration'], r['score_diff']) for r in records if r['score_diff'] is not None]
    if sd_data:
        sd_iters, sd_vals = zip(*sd_data)
        colors = ['steelblue' if v >= 0 else 'coral' for v in sd_vals]
        ax.bar(sd_iters, sd_vals, color=colors, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        # 移動平均
        if len(sd_vals) >= 3:
            window = min(5, len(sd_vals))
            ma = np.convolve(sd_vals, np.ones(window)/window, mode='valid')
            ma_iters = list(sd_iters)[window-1:]
            ax.plot(ma_iters, ma, 'k-', linewidth=2, label=f'{window}iter移動平均')
            ax.legend(fontsize=9)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('score_diff')
    ax.set_title('勾配信号 (score_diff)')
    ax.grid(True, alpha=0.3)

    # --- 4. c_k, r_k 減衰 ---
    ax = axes[1, 1]
    ax.plot(iters, [r['c_k'] for r in records], 'o-', color='purple', label='c_k (摂動)', markersize=3)
    ax.plot(iters, [r['r_k'] for r in records], 's-', color='darkorange', label='r_k (学習率)', markersize=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('値')
    ax.set_title('ハイパーパラメータ減衰')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- 5~10. 各パラメータの推移 ---
    for i, pname in enumerate(PARAM_NAMES):
        row = 2 + i // 2
        col = i % 2
        ax = axes[row, col]

        vals = [r['theta'][pname] for r in records]
        init_val = init_params.get(pname, vals[0] if vals else 0)

        ax.plot(iters, vals, 'o-', color='steelblue', markersize=4, label='θ (現在値)')
        ax.axhline(y=init_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'初期値 ({init_val})')

        # θ+, θ-も薄く表示
        vp = [r['theta_plus'].get(pname, np.nan) for r in records]
        vm = [r['theta_minus'].get(pname, np.nan) for r in records]
        ax.fill_between(iters, vm, vp, alpha=0.15, color='steelblue', label='θ± 範囲')

        ax.set_xlabel('Iteration')
        ax.set_ylabel(pname)
        ax.set_title(f'{pname} 推移')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log_file> [output_image]")
        sys.exit(1)

    log_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(log_path)[0] + '_plot.png'

    # JSONLがあればそちらを優先
    jsonl_path = os.path.splitext(log_path)[0] + '.jsonl'
    if os.path.exists(jsonl_path):
        print(f"Using JSONL: {jsonl_path}")
        raw = parse_jsonl(jsonl_path)
        init_params = raw[0]['theta'] if raw else {}
        # JSOLNにはinit_paramsが直接ないので最初のthetaから推定はしない
        # テキストログからinit_paramsを取得
        if os.path.exists(log_path):
            _, init_params_txt = parse_text_log(log_path)
            if init_params_txt:
                init_params = init_params_txt
        records = raw
    else:
        print(f"Parsing text log: {log_path}")
        records, init_params = parse_text_log(log_path)

    print(f"Found {len(records)} completed iterations")
    if init_params:
        print(f"Initial params: {init_params}")

    if records:
        last = records[-1]
        print(f"Current theta: {last['theta']}")

    plot_spsa(records, init_params, output_path)


if __name__ == '__main__':
    main()
