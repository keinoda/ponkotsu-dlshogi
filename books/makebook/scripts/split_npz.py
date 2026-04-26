import argparse
import os

import numpy as np


def split_npz(input_path, shard_size, output_dir):
    if output_dir is None:
        output_dir = os.path.dirname(input_path) or '.'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    basename = os.path.splitext(os.path.basename(input_path))[0]

    npz = np.load(input_path, allow_pickle=True)
    arrays = dict(npz)
    npz.close()

    # 第0軸の長さを取得（全配列で共通と仮定）
    first_key = next(iter(arrays))
    total = len(arrays[first_key])

    num_shards = (total + shard_size - 1) // shard_size
    print(f'Total samples: {total}, shard size: {shard_size}, num shards: {num_shards}')

    for i in range(num_shards):
        start = i * shard_size
        end = min(start + shard_size, total)

        shard = {}
        for key, arr in arrays.items():
            shard[key] = arr[start:end]

        out_path = os.path.join(output_dir, f'{basename}_shard_{i:03d}.npz')
        np.savez(out_path, **shard)
        print(f'  {out_path} ({end - start} samples)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split an npz file into shards')
    parser.add_argument('input', help='Input npz file path')
    parser.add_argument('--shard-size', type=int, default=10000, help='Number of samples per shard (default: 10000)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: same as input)')
    args = parser.parse_args()

    split_npz(args.input, args.shard_size, args.output_dir)
