import argparse
import os

import numpy as np


CSR_OFFSET_KEY = 'child_offsets'
CSR_FLAT_KEYS = ('child_move_flat', 'child_policy_flat')


def split_npz(input_path, shard_size, output_dir):
    if output_dir is None:
        output_dir = os.path.dirname(input_path) or '.'

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    basename = os.path.splitext(os.path.basename(input_path))[0]

    npz = np.load(input_path, allow_pickle=True)
    arrays = dict(npz)
    npz.close()

    has_csr = CSR_OFFSET_KEY in arrays and all(k in arrays for k in CSR_FLAT_KEYS)

    # child_offsets は N+1 要素 (sentinel 付き) なので行数の基準にしない
    if has_csr:
        row_keys = [k for k in arrays if k != CSR_OFFSET_KEY and k not in CSR_FLAT_KEYS]
    else:
        row_keys = list(arrays.keys())

    total = len(arrays[row_keys[0]])
    num_shards = (total + shard_size - 1) // shard_size
    print(f'Total samples: {total}, shard size: {shard_size}, num shards: {num_shards}')

    child_offsets = arrays.get(CSR_OFFSET_KEY)

    for i in range(num_shards):
        row_start = i * shard_size
        row_end = min(row_start + shard_size, total)

        shard = {}
        for key in row_keys:
            shard[key] = arrays[key][row_start:row_end]

        if has_csr:
            flat_start = int(child_offsets[row_start])
            flat_end = int(child_offsets[row_end])
            shard[CSR_OFFSET_KEY] = child_offsets[row_start:row_end + 1] - flat_start
            for fk in CSR_FLAT_KEYS:
                shard[fk] = arrays[fk][flat_start:flat_end]

        out_path = os.path.join(output_dir, f'{basename}_shard_{i:03d}.npz')
        np.savez(out_path, **shard)
        print(f'  {out_path} ({row_end - row_start} samples)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split an npz file into shards')
    parser.add_argument('input', help='Input npz file path')
    parser.add_argument('--shard-size', type=int, default=10000, help='Number of samples per shard (default: 10000)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: same as input)')
    args = parser.parse_args()

    split_npz(args.input, args.shard_size, args.output_dir)
