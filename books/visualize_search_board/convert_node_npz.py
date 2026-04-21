"""Convert pickle-format dl_data_tree to npz format for fast loading.

Usage:
    python convert_node_npz.py input.pickle output.npz

The npz file contains columnar arrays:
    keys            : uint64   (N,)
    board_sfens     : object   (N,)  -- variable-length strings
    values          : float32  (N,)
    child_offsets   : int64    (N+1,) -- CSR-style offsets into flat child arrays
    child_move_flat : int32    (total_children,)
    child_policy_flat: float32 (total_children,)
    has_children    : bool     (N,)
"""
import argparse
import pickle
import numpy as np
import tqdm


def pickle_to_npz(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = pickle.load(f)

    n = len(data)
    keys = np.empty(n, dtype=np.uint64)
    board_sfens = np.empty(n, dtype=object)
    values = np.empty(n, dtype=np.float32)
    has_children = np.empty(n, dtype=np.bool_)

    # First pass: count total children for pre-allocation
    total_children = 0
    for node in data.values():
        if node.child_move is not None and len(node.child_move) > 0:
            total_children += len(node.child_move)

    child_offsets = np.empty(n + 1, dtype=np.int64)
    child_move_flat = np.empty(total_children, dtype=np.int32)
    child_policy_flat = np.empty(total_children, dtype=np.float32)

    offset = 0
    for i, (key, node) in enumerate(tqdm.tqdm(data.items(), total=n)):
        keys[i] = key
        values[i] = float(node.value)

        # board can be a sfen string or a Board object
        if isinstance(node.board, str):
            board_sfens[i] = node.board
        elif node.board is not None:
            board_sfens[i] = node.board.sfen()
        else:
            board_sfens[i] = ''

        child_offsets[i] = offset
        if node.child_move is not None and len(node.child_move) > 0:
            nc = len(node.child_move)
            has_children[i] = True
            for j, m in enumerate(node.child_move):
                child_move_flat[offset + j] = int(m)
            if node.child_policy is not None:
                child_policy_flat[offset:offset + nc] = np.asarray(node.child_policy, dtype=np.float32)
            else:
                child_policy_flat[offset:offset + nc] = 0.0
            offset += nc
        else:
            has_children[i] = False

    child_offsets[n] = offset

    # Encode sfens as bytes for efficient storage
    sfen_bytes = np.array([s.encode('ascii') if s else b'' for s in board_sfens])

    np.savez(
        output_path,
        keys=keys,
        sfen_bytes=sfen_bytes,
        values=values,
        has_children=has_children,
        child_offsets=child_offsets,
        child_move_flat=child_move_flat[:offset],
        child_policy_flat=child_policy_flat[:offset],
    )

    print(f"Converted {n} nodes ({offset} total children) -> {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    pickle_to_npz(args.input, args.output)
