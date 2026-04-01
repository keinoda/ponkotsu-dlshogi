#!/usr/bin/env python3
"""Optunaの探索パラメータ最適化ログをプロットするスクリプト"""

import re
import sys
import ast
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

def parse_log(path):
    pattern = re.compile(
        r"Trial (\d+) finished with value: (-?[\d.]+) and parameters: ({.*?})\."
    )
    trials = []
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                trial_id = int(m.group(1))
                value = float(m.group(2))
                params = ast.literal_eval(m.group(3))
                trials.append({"trial": trial_id, "value": value, **params})
    return trials

def plot(trials, output_path):
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    fig.suptitle("探索パラメータ最適化", fontsize=16, y=0.98)

    trial_ids = [t["trial"] for t in trials]
    values = [t["value"] for t in trials]
    best_values = []
    best = float("inf")
    for v in values:
        best = min(best, v)
        best_values.append(best)

    # 1. 勝率推移（value = -(勝率) なので符号反転して表示）
    ax = axes[0, 0]
    win_rates = [-v * 100 for v in values]
    best_win_rates = [-v * 100 for v in best_values]
    ax.scatter(trial_ids, win_rates, c="steelblue", alpha=0.7, label="各Trial", zorder=2)
    ax.plot(trial_ids, best_win_rates, c="red", linewidth=2, label="ベスト", zorder=3)
    ax.set_xlabel("Trial")
    ax.set_ylabel("勝率 (%)")
    ax.set_title("勝率推移")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2-7. 各パラメータ vs 勝率の散布図
    param_names = ["C_init", "C_base", "C_fpu_reduction", "C_init_root", "C_base_root", "Softmax_Temperature"]
    positions = [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]

    for (row, col), pname in zip(positions, param_names):
        ax = axes[row, col]
        x = [t[pname] for t in trials]
        y = [-t["value"] * 100 for t in trials]
        scatter = ax.scatter(x, y, c=trial_ids, cmap="viridis", alpha=0.8, edgecolors="black", linewidth=0.5)
        ax.set_xlabel(pname)
        ax.set_ylabel("勝率 (%)")
        ax.set_title(f"{pname} vs 勝率")
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Trial")

    # 最後のパネル: ベストパラメータのサマリ
    ax = axes[3, 1]
    ax.axis("off")
    best_trial = min(trials, key=lambda t: t["value"])
    text = f"Best Trial: {best_trial['trial']}\n勝率: {-best_trial['value']*100:.1f}%\n\n"
    for p in param_names:
        text += f"{p}: {best_trial[p]}\n"
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("ベストパラメータ")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "log_optimize_model_resnet35x512.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else log_path.replace(".txt", ".png")
    trials = parse_log(log_path)
    if not trials:
        print("No trial data found.")
        sys.exit(1)
    print(f"Found {len(trials)} trials.")
    plot(trials, output_path)
