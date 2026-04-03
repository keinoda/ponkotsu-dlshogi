#!/usr/bin/env python3
"""Optunaの探索パラメータ最適化ログをプロットするスクリプト"""

import re
import sys
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Hiragino Sans'

PARAM_NAMES = ["C_init", "C_base", "C_fpu_reduction", "C_init_root", "C_base_root", "Softmax_Temperature"]

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
    ax.axhline(y=48.5, color="gray", linestyle="--", linewidth=1.5, label="デフォルト (48.5%)", zorder=1)
    ax.set_xlabel("Trial")
    ax.set_ylabel("勝率 (%)")
    ax.set_title("勝率推移")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2-7. 各パラメータ vs 勝率の散布図
    positions = [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]

    for (row, col), pname in zip(positions, PARAM_NAMES):
        ax = axes[row, col]
        x = [t[pname] for t in trials]
        y = [-t["value"] * 100 for t in trials]
        scatter = ax.scatter(x, y, c=trial_ids, cmap="viridis", alpha=0.8, edgecolors="black", linewidth=0.5)
        ax.set_xlabel(pname)
        ax.set_ylabel("勝率 (%)")
        ax.set_title(f"{pname} vs 勝率")
        ax.axhline(y=48.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Trial")

    # 最後のパネル: ベストパラメータのサマリ
    ax = axes[3, 1]
    ax.axis("off")
    best_trial = min(trials, key=lambda t: t["value"])
    text = f"Best Trial: {best_trial['trial']}\n勝率: {-best_trial['value']*100:.1f}%\n\n"
    for p in PARAM_NAMES:
        text += f"{p}: {best_trial[p]}\n"
    ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("ベストパラメータ")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {output_path}")


def plot_correlation(trials, output_path):
    """パラメータ間の相関行列ヒートマップ（勝率上位半分でフィルタ）"""
    sorted_trials = sorted(trials, key=lambda t: t["value"])
    top_half = sorted_trials[:len(sorted_trials)//2]

    data = np.array([[t[p] for p in PARAM_NAMES] for t in top_half])
    if data.shape[0] < 3:
        return
    corr = np.corrcoef(data.T)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(PARAM_NAMES)))
    ax.set_yticks(range(len(PARAM_NAMES)))
    ax.set_xticklabels(PARAM_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(PARAM_NAMES, fontsize=9)
    for i in range(len(PARAM_NAMES)):
        for j in range(len(PARAM_NAMES)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.5 else "black", fontsize=10)
    plt.colorbar(im, ax=ax)
    ax.set_title("パラメータ相関（勝率上位50%）", fontsize=14)
    plt.tight_layout()
    corr_path = output_path.replace(".png", "_correlation.png")
    plt.savefig(corr_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {corr_path}")


def plot_param_history(trials, output_path):
    """各パラメータのTrial推移（TPEの探索領域の変化を可視化）"""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("パラメータ探索履歴", fontsize=16, y=0.98)

    trial_ids = [t["trial"] for t in trials]
    win_rates = [-t["value"] * 100 for t in trials]

    for ax, pname in zip(axes.flat, PARAM_NAMES):
        values = [t[pname] for t in trials]
        scatter = ax.scatter(trial_ids, values, c=win_rates, cmap="RdYlBu_r", alpha=0.8,
                            edgecolors="black", linewidth=0.5, vmin=min(win_rates), vmax=max(win_rates))
        ax.set_xlabel("Trial")
        ax.set_ylabel(pname)
        ax.set_title(f"{pname} 探索履歴")
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("勝率 (%)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    hist_path = output_path.replace(".png", "_history.png")
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {hist_path}")


def plot_pairwise(trials, output_path):
    """勝率上位/下位で色分けしたペアプロット（重要なパラメータ組み合わせ）"""
    win_rates = [-t["value"] * 100 for t in trials]
    median_wr = np.median(win_rates)

    # 重要度が高そうな3ペアに絞る
    pairs = [
        ("C_init", "C_base"),
        ("C_init_root", "C_base_root"),
        ("Softmax_Temperature", "C_fpu_reduction"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("パラメータペアプロット（緑=上位, 赤=下位）", fontsize=14, y=1.02)

    for ax, (px, py) in zip(axes, pairs):
        for t, wr in zip(trials, win_rates):
            color = "green" if wr >= median_wr else "red"
            alpha = 0.9 if wr >= median_wr else 0.4
            size = 60 if wr >= median_wr else 30
            ax.scatter(t[px], t[py], c=color, alpha=alpha, s=size, edgecolors="black", linewidth=0.3)
        ax.set_xlabel(px)
        ax.set_ylabel(py)
        ax.set_title(f"{px} vs {py}")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pair_path = output_path.replace(".png", "_pairwise.png")
    plt.savefig(pair_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {pair_path}")

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "log_optimize_model_resnet35x512.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else log_path.replace(".txt", ".png")
    trials = parse_log(log_path)
    if not trials:
        print("No trial data found.")
        sys.exit(1)
    print(f"Found {len(trials)} trials.")
    plot(trials, output_path)
    plot_correlation(trials, output_path)
    plot_param_history(trials, output_path)
    plot_pairwise(trials, output_path)
