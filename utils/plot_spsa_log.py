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
from matplotlib.ticker import MaxNLocator
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

        # [total]行がない場合、個別マッチ完了結果からフォールバック抽出
        if win_rate_plus is None or win_rate_minus is None:
            has_plus = 'Playing theta+ match' in block
            has_minus = 'Playing theta- match' in block
            # θ+セクション: "Playing theta+ match" ~ "Playing theta- match"
            if has_plus and win_rate_plus is None:
                plus_end = block.find('Playing theta- match') if has_minus else len(block)
                plus_section = block[block.find('Playing theta+ match'):plus_end]
                games = re.findall(r'(\d+) of (\d+) games finished\.\n.*?\(([\d.]+)%\)', plus_section, re.DOTALL)
                if games:
                    last = games[-1]
                    win_rate_plus = float(last[2]) / 100.0
            # θ-セクション: "Playing theta- match" ~ "Playing theta+ vs theta-" or end
            if has_minus and win_rate_minus is None:
                minus_start = block.find('Playing theta- match')
                pm_pos = block.find('Playing theta+ vs theta-', minus_start)
                minus_end = pm_pos if pm_pos != -1 else len(block)
                minus_section = block[minus_start:minus_end]
                games = re.findall(r'(\d+) of (\d+) games finished\.\n.*?\(([\d.]+)%\)', minus_section, re.DOTALL)
                if games:
                    last = games[-1]
                    win_rate_minus = float(last[2]) / 100.0

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
        print("No iterations found.")
        return

    completed = [r for r in records if r['theta'] is not None]
    iters = [r['iteration'] for r in records]
    n_completed = len(completed)
    n_total = len(records)
    status = f"{n_completed} iterations完了" + (f" +{n_total - n_completed}進行中" if n_total > n_completed else "")

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(f"SPSA パラメータ最適化 ({status})", fontsize=16, y=0.98)

    # --- 1. 勝率推移 (θ+ vs baseline, θ- vs baseline) ---
    ax = axes[0, 0]
    iters_plus = [r['iteration'] for r in records if r['win_rate_plus'] is not None]
    wr_plus = [r['win_rate_plus'] * 100 for r in records if r['win_rate_plus'] is not None]
    iters_minus = [r['iteration'] for r in records if r['win_rate_minus'] is not None]
    wr_minus = [r['win_rate_minus'] * 100 for r in records if r['win_rate_minus'] is not None]
    ax.plot(iters_plus, wr_plus, 'o-', color='steelblue', label='θ+ vs baseline', markersize=4)
    ax.plot(iters_minus, wr_minus, 's-', color='coral', label='θ- vs baseline', markersize=4)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('勝率 (%)')
    ax.set_title('θ± vs Baseline 勝率推移')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))

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
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))

    # --- 3. score_diff 推移 ---
    ax = axes[1, 0]
    sd_data = [(r['iteration'], r['score_diff']) for r in records if r['score_diff'] is not None]
    if sd_data:
        sd_iters, sd_vals = zip(*sd_data)
        colors = ['steelblue' if v >= 0 else 'coral' for v in sd_vals]
        ax.bar(sd_iters, sd_vals, color=colors, alpha=0.7)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
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
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))

    # --- 4. c_k, r_k 減衰 (全イテレーション表示可能) ---
    ax = axes[1, 1]
    ax.plot(iters, [r['c_k'] for r in records], 'o-', color='purple', label='c_k (摂動)', markersize=3)
    ax.plot(iters, [r['r_k'] for r in records], 's-', color='darkorange', label='r_k (学習率)', markersize=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('値')
    ax.set_title('ハイパーパラメータ減衰')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))

    # --- 5. パラメータ絶対値テーブル (最新値 vs 初期値) ---
    ax = axes[2, 0]
    ax.axis('off')
    last_theta = completed[-1]['theta'] if completed else init_params
    table_data = []
    for pname in PARAM_NAMES:
        iv = init_params.get(pname, '-')
        cv = last_theta.get(pname, '-')
        diff = cv - iv if isinstance(cv, (int, float)) and isinstance(iv, (int, float)) else '-'
        sign = '+' if isinstance(diff, (int, float)) and diff > 0 else ''
        table_data.append([pname, str(iv), str(cv), f'{sign}{diff}'])
    table = ax.table(
        cellText=table_data,
        colLabels=['パラメータ', '初期値', '現在値', '差分'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    ax.set_title('パラメータ一覧', fontsize=12, pad=20)

    # --- 6. 勝率まとめ (最新データあるイテレーション) ---
    ax = axes[2, 1]
    ax.axis('off')
    last_r = records[-1]
    summary = [
        ['θ+ vs baseline', f"{last_r['win_rate_plus']*100:.1f}%" if last_r['win_rate_plus'] else '-'],
        ['θ- vs baseline', f"{last_r['win_rate_minus']*100:.1f}%" if last_r['win_rate_minus'] else '-'],
        ['θ+ vs θ- (dev)', f"{last_r['win_rate_pm']*100:.1f}%" if last_r['win_rate_pm'] else '-'],
        ['score_diff', f"{last_r['score_diff']:.4f}" if last_r['score_diff'] else '-'],
        ['c_k', f"{last_r['c_k']:.4f}"],
        ['r_k', f"{last_r['r_k']:.4f}"],
    ]
    table2 = ax.table(
        cellText=summary,
        colLabels=['項目', '最新Iteration'],
        loc='center',
        cellLoc='center',
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 1.5)
    ax.set_title(f'最新 Iteration {last_r["iteration"]} サマリ', fontsize=12, pad=20)

    # --- 7. 全パラメータ変化率 (完了イテレーションのみ) ---
    ax = axes[3, 0]
    colors_p = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    iters_c = [r['iteration'] for r in completed]
    for i, pname in enumerate(PARAM_NAMES):
        vals = [r['theta'][pname] for r in completed]
        init_val = init_params.get(pname, vals[0] if vals else 0)
        pct = [(v - init_val) / max(abs(init_val), 1) * 100 for v in vals]
        ax.plot(iters_c, pct, 'o-', color=colors_p[i], markersize=4, label=pname)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('初期値からの変化率 (%)')
    ax.set_title('全パラメータ変化率')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))

    # --- 8. 全パラメータ正規化推移 ---
    RANGES = {
        'C_init': (100, 200), 'C_base': (20000, 50000), 'C_fpu_reduction': (0, 40),
        'C_init_root': (100, 200), 'C_base_root': (20000, 50000), 'Softmax_Temperature': (100, 200),
    }
    ax = axes[3, 1]
    for i, pname in enumerate(PARAM_NAMES):
        vals = [r['theta'][pname] for r in completed]
        lo, hi = RANGES.get(pname, (min(vals) if vals else 0, max(vals) if vals else 1))
        normed = [(v - lo) / (hi - lo) for v in vals]
        init_normed = (init_params.get(pname, vals[0] if vals else 0) - lo) / (hi - lo)
        ax.plot(iters_c, normed, 'o-', color=colors_p[i], markersize=4, label=pname)
        ax.axhline(y=init_normed, color=colors_p[i], linestyle=':', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('正規化値 [0, 1]')
    ax.set_title('全パラメータ正規化推移')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_params_detail(records, init_params, output_path):
    """6パラメータの個別推移(上段) + パラメータ値 vs 勝率散布図(下段) + ベストパラメータ"""
    if not records:
        return

    completed = [r for r in records if r['theta'] is not None]
    iters = [r['iteration'] for r in records]
    iters_c = [r['iteration'] for r in completed]
    n_completed = len(completed)
    n_total = len(records)
    status = f"{n_completed} iterations完了" + (f" +{n_total - n_completed}進行中" if n_total > n_completed else "")

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(f"SPSA パラメータ詳細 ({status})", fontsize=16, y=0.98)

    # --- 上段 3×2: パラメータ時系列推移 ---
    for i, pname in enumerate(PARAM_NAMES):
        row = i // 2
        col = i % 2
        ax = axes[row, col]

        vals = [r['theta'][pname] for r in completed]
        init_val = init_params.get(pname, vals[0] if vals else 0)

        ax.plot(iters_c, vals, 'o-', color='steelblue', markersize=4, label='θ (現在値)')
        ax.axhline(y=init_val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f'初期値 ({init_val})')

        # θ+, θ-の摂動範囲 (全イテレーション表示可能)
        vp = [r['theta_plus'].get(pname, np.nan) for r in records]
        vm = [r['theta_minus'].get(pname, np.nan) for r in records]
        ax.fill_between(iters, vm, vp, alpha=0.15, color='steelblue', label='θ± 範囲')

        ax.set_xlabel('Iteration')
        ax.set_ylabel(pname)
        ax.set_title(f'{pname} 推移')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))

    # --- 下段左: ベスト θ+ パラメータ ---
    ax = axes[3, 0]
    ax.axis('off')

    # θ+/θ- の全データ点を収集して最良を特定
    best_wr = -1
    best_params = {}
    best_label = ''
    for r in records:
        if r['win_rate_plus'] is not None and r['win_rate_plus'] > best_wr:
            best_wr = r['win_rate_plus']
            best_params = r['theta_plus']
            best_label = f"Iter {r['iteration']} θ+"
        if r['win_rate_minus'] is not None and r['win_rate_minus'] > best_wr:
            best_wr = r['win_rate_minus']
            best_params = r['theta_minus']
            best_label = f"Iter {r['iteration']} θ-"

    table_data = [[f'{best_label}', f'{best_wr*100:.1f}%', '', '']]
    table_data.append(['', '', '', ''])
    for pname in PARAM_NAMES:
        iv = init_params.get(pname, '-')
        bv = best_params.get(pname, '-')
        cv = completed[-1]['theta'].get(pname, '-') if completed else '-'
        table_data.append([pname, str(iv), str(bv), str(cv)])
    table = ax.table(
        cellText=table_data,
        colLabels=['パラメータ', '初期値', 'ベスト評価', '最新θ'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title('ベストパラメータ', fontsize=12, pad=20)

    # --- 下段右: パラメータ値 vs 勝率 散布図 (全パラメータ正規化重ね) ---
    ax = axes[3, 1]
    RANGES = {
        'C_init': (100, 200), 'C_base': (20000, 50000), 'C_fpu_reduction': (0, 40),
        'C_init_root': (100, 200), 'C_base_root': (20000, 50000), 'Softmax_Temperature': (100, 200),
    }
    colors_p = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for pi, pname in enumerate(PARAM_NAMES):
        xs = []
        ys = []
        for r in records:
            lo, hi = RANGES.get(pname, (0, 1))
            if r['win_rate_plus'] is not None:
                v = r['theta_plus'].get(pname, 0)
                xs.append((v - lo) / (hi - lo))
                ys.append(r['win_rate_plus'] * 100)
            if r['win_rate_minus'] is not None:
                v = r['theta_minus'].get(pname, 0)
                xs.append((v - lo) / (hi - lo))
                ys.append(r['win_rate_minus'] * 100)
        ax.scatter(xs, ys, color=colors_p[pi], alpha=0.6, s=20, label=pname)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xlabel('正規化パラメータ値 [0, 1]')
    ax.set_ylabel('勝率 (%)')
    ax.set_title('パラメータ値 vs 勝率 (正規化)')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_params_vs_winrate(records, init_params, output_path):
    """パラメータ値 vs 勝率の散布図を個別に (plot_optuna_log.py スタイル)"""
    if not records:
        return

    n = len(records)
    n_completed = len([r for r in records if r['theta'] is not None])

    # θ+, θ- のデータ点を収集
    points = []  # list of (iteration, params_dict, win_rate, label)
    for r in records:
        if r['win_rate_plus'] is not None:
            points.append((r['iteration'], r['theta_plus'], r['win_rate_plus'], 'θ+'))
        if r['win_rate_minus'] is not None:
            points.append((r['iteration'], r['theta_minus'], r['win_rate_minus'], 'θ-'))

    if not points:
        return

    iters_all = [p[0] for p in points]
    wrs_all = [p[2] * 100 for p in points]
    labels_all = [p[3] for p in points]

    # ベスト特定
    best_idx = max(range(len(points)), key=lambda i: points[i][2])
    best_iter, best_params, best_wr, _ = points[best_idx]

    status = f"{len(points)} データ点, {n_completed} iterations完了" + (f" +{n - n_completed}進行中" if n > n_completed else "")

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle(f"パラメータ vs 勝率 ({status})", fontsize=16, y=0.98)

    # --- 左上: 勝率推移 ---
    ax = axes[0, 0]
    # θ+/θ- を色分けプロット
    for label, color, marker in [('θ+', 'steelblue', 'o'), ('θ-', 'coral', 's')]:
        idxs = [i for i, l in enumerate(labels_all) if l == label]
        if idxs:
            ax.scatter([iters_all[i] for i in idxs], [wrs_all[i] for i in idxs],
                       color=color, marker=marker, alpha=0.7, zorder=2, label=label)
    # ベスト推移 (イテレーション単位の累積ベスト)
    iter_best = {}
    for it, wr in zip(iters_all, wrs_all):
        iter_best[it] = max(iter_best.get(it, 0), wr)
    sorted_iters = sorted(iter_best.keys())
    cum_best = []
    bsf = 0
    for it in sorted_iters:
        bsf = max(bsf, iter_best[it])
        cum_best.append(bsf)
    ax.plot(sorted_iters, cum_best, 'r-', linewidth=2, label='ベスト', zorder=3)
    ax.axhline(y=48.5, color='gray', linestyle='--', linewidth=1.5, label='デフォルト (48.5%)', zorder=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('勝率 (%)')
    ax.set_title('勝率推移')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    # X軸を整数イテレーションに合わせる
    max_iter = max(iters_all) if iters_all else 0
    ax.set_xlim(-0.5, max_iter + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))

    # --- 6パラメータ vs 勝率 散布図: (0,1), (1,0), (1,1), (2,0), (2,1), (3,0) ---
    scatter_positions = [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]
    for i, pname in enumerate(PARAM_NAMES):
        row, col = scatter_positions[i]
        ax = axes[row, col]

        xs = [p[1].get(pname, 0) for p in points]
        sc = ax.scatter(xs, wrs_all, c=range(len(points)), cmap='viridis', alpha=0.7)
        ax.set_xlabel(pname)
        ax.set_ylabel('勝率 (%)')
        ax.set_title(f'{pname} vs 勝率')
        ax.grid(True, alpha=0.3)
        plt.colorbar(sc, ax=ax, label='データ点番号')

    # --- 右下: ベストパラメータ テキスト ---
    ax = axes[3, 1]
    ax.axis('off')
    text_lines = [f'Best: Iter {best_iter}', f'勝率: {best_wr*100:.1f}%', '']
    for pname in PARAM_NAMES:
        text_lines.append(f'{pname}: {best_params.get(pname, "-")}')
    ax.text(0.5, 0.5, '\n'.join(text_lines), transform=ax.transAxes,
            fontsize=11, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.8))
    ax.set_title('ベストパラメータ', fontsize=12, pad=20)

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

    completed = [r for r in records if r['theta'] is not None]
    in_progress = [r for r in records if r['theta'] is None]
    print(f"Found {len(completed)} completed iterations" + (f" (+{len(in_progress)} in progress)" if in_progress else ""))
    if init_params:
        print(f"Initial params: {init_params}")

    if completed:
        last = completed[-1]
        print(f"Current theta: {last['theta']}")

    plot_spsa(records, init_params, output_path)

    # パラメータ個別推移を別ファイルに保存
    base, ext = os.path.splitext(output_path)
    params_path = base + '_params' + ext
    plot_params_detail(records, init_params, params_path)

    # パラメータ vs 勝率 散布図を別ファイルに保存
    scatter_path = base + '_scatter' + ext
    plot_params_vs_winrate(records, init_params, scatter_path)


if __name__ == '__main__':
    main()
