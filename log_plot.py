#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse multiple training log files and plot results on the same graphs."
    )
    # 複数ファイルを可変長引数で受け取るようにする (nargs='+')
    parser.add_argument(
        "log_files",
        type=str,
        nargs='+',
        help="Paths to one or more log files to parse. (e.g. path/to/dir1/log1.log path/to/dir2/log2.log ...)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory where the output plots will be saved. (default: current directory)"
    )
    return parser.parse_args()


def parse_log_file(log_file_path):
    """
    単一のログファイルを読み込んで、epoch ごとの各種メトリクスを返す。
    """
    # epoch終了時を特定するための正規表現パターン
    pattern_final_epoch = re.compile(
        r'epoch\s*=\s*(\d+),\s*steps\s*=\s*(\d+),\s*'
        r'train loss avr = ([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*'
        r'test loss = ([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*'
        r'test accuracy = ([^,]+),\s*([^,]+)'
        r'(?:,\s*test entropy = ([^,]+),\s*([^,]+))?'  # test entropy は存在しない場合もあるのでオプション
    )

    # SWA 有効時は同一 epoch の評価が後段で追記される場合がある。
    # プロットでは epoch 終了時点の元の集計値を使いたいので、各 epoch の最初の結果だけ採用する。
    results_by_epoch = {}
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            match = pattern_final_epoch.search(line)
            if match:
                epoch = int(match.group(1))
                if epoch in results_by_epoch:
                    continue

                steps = int(match.group(2))

                # train loss: 4つ
                train_loss_policy = float(match.group(3))
                train_loss_val    = float(match.group(4))
                train_loss_reg    = float(match.group(5))
                train_loss_total  = float(match.group(6))

                # test loss: 4つ
                test_loss_policy = float(match.group(7))
                test_loss_val    = float(match.group(8))
                test_loss_reg    = float(match.group(9))
                test_loss_total  = float(match.group(10))

                # test accuracy: 2つ
                test_acc_policy  = float(match.group(11))
                test_acc_value   = float(match.group(12))

                # test entropy (オプション)
                test_entropy_policy = float(match.group(13)) if match.group(13) else None
                test_entropy_value  = float(match.group(14)) if match.group(14) else None

                results_by_epoch[epoch] = {
                    'epoch': epoch,
                    'steps': steps,
                    'train_loss_policy': train_loss_policy,
                    'train_loss_val': train_loss_val,
                    'train_loss_reg': train_loss_reg,
                    'train_loss_total': train_loss_total,
                    'test_loss_policy': test_loss_policy,
                    'test_loss_val': test_loss_val,
                    'test_loss_reg': test_loss_reg,
                    'test_loss_total': test_loss_total,
                    'test_acc_policy': test_acc_policy,
                    'test_acc_value': test_acc_value,
                    'test_entropy_policy': test_entropy_policy,
                    'test_entropy_value': test_entropy_value
                }

    # epoch 順にソート
    results = sorted(results_by_epoch.values(), key=lambda x: x['epoch'])
    return results


def main():
    args = parse_args()
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ログファイルごとのパース結果を格納する
    all_logs_data = {}

    # 各ログファイルをパースし、結果を保存
    for log_file_path in args.log_files:
        # 親ディレクトリ名をラベルとして使用
        # 例:
        #   log_file_path = "/path/to/dirA/log1.log"
        #   os.path.dirname(log_file_path) => "/path/to/dirA"
        #   os.path.basename("/path/to/dirA") => "dirA"
        label = os.path.basename(os.path.dirname(log_file_path))
        results = parse_log_file(log_file_path)
        all_logs_data.setdefault(label, []).extend(results)

    # それぞれのラベルに対して epoch 順にソート (複数ファイルが同じdirにあるケースを想定)
    for label in all_logs_data:
        all_logs_data[label].sort(key=lambda x: x['epoch'])

    # すべてのログデータに対して、train_loss_total・test_loss_total・test_acc_policy・test_acc_value をまとめてプロット
    # -------------------------------------------------------------------------
    # 1) Loss グラフ (train_loss_total と test_loss_total)
    plt.figure(figsize=(8, 5))
    for label, results in all_logs_data.items():
        if not results:
            continue
        epochs = [r['epoch'] for r in results]
        train_loss_total_list = [r['train_loss_total'] for r in results]
        test_loss_total_list  = [r['test_loss_total']  for r in results]

        # train / test を別々のラインで区別
        plt.plot(epochs, train_loss_total_list, label=f"{label} - train_loss")
        plt.plot(epochs, test_loss_total_list,  label=f"{label} - test_loss")

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training & Test Loss per Epoch')
    plt.legend()
    plt.grid()

    loss_plot_path = os.path.join(output_dir, "loss_per_epoch.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # 2) Accuracy グラフ (policy / value)
    plt.figure(figsize=(8, 5))
    for label, results in all_logs_data.items():
        if not results:
            continue
        epochs = [r['epoch'] for r in results]
        test_acc_policy_list = [r['test_acc_policy'] for r in results]
        test_acc_value_list  = [r['test_acc_value']  for r in results]

        plt.plot(epochs, test_acc_policy_list, label=f"{label} - policy_acc")
        plt.plot(epochs, test_acc_value_list,  label=f"{label} - value_acc")

    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy (Policy & Value) per Epoch')
    plt.legend()
    plt.grid()

    accuracy_plot_path = os.path.join(output_dir, "accuracy_per_epoch.png")
    plt.savefig(accuracy_plot_path)
    plt.close()

    print(f"Plots saved:\n  {loss_plot_path}\n  {accuracy_plot_path}")


if __name__ == '__main__':
    main()
