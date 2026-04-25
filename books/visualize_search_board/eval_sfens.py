import argparse
from eval import *
from mcts_on_book_dl import softmax_temperature_with_normalization
import os
import time
import tqdm


def make_rotated_key(board):
    rotated = cshogi.Board(cshogi.rotate_sfen(board.sfen()))
    return rotated.zobrist_hash()


def make_canonical_key(board):
    key = board.zobrist_hash()
    rotated_key = make_rotated_key(board)
    return min(key, rotated_key)


def list_npz_files(out_dir):
    if not os.path.isdir(out_dir):
        return []
    files = []
    for name in os.listdir(out_dir):
        if name.endswith('.npz'):
            files.append(os.path.join(out_dir, name))
    files.sort()
    return files


def load_existing_canonical_keys(out_dir):
    canonical_keys = set()
    npz_files = list_npz_files(out_dir)

    for path in tqdm.tqdm(npz_files, desc='index npz', dynamic_ncols=True):
        try:
            npz = np.load(path, allow_pickle=False)
        except ValueError:
            npz = np.load(path, allow_pickle=True)

        with npz:
            if 'canonical_keys' in npz:
                for key in npz['canonical_keys']:
                    canonical_keys.add(int(key))
                continue

            if 'sfen_bytes' in npz:
                for sb in npz['sfen_bytes']:
                    if len(sb) == 0:
                        continue
                    sfen = sb.decode('ascii') if isinstance(sb, bytes) else str(sb)
                    board = cshogi.Board(sfen=sfen)
                    canonical_keys.add(make_canonical_key(board))
                continue

            if 'keys' in npz:
                for key in npz['keys']:
                    canonical_keys.add(int(key))

    return canonical_keys


def save_npz_shard(out_dir, shard_prefix, shard_index, records):
    n = len(records)
    if n == 0:
        return None

    keys = np.empty(n, dtype=np.uint64)
    canonical_keys = np.empty(n, dtype=np.uint64)
    sfen_bytes_list = []
    values = np.empty(n, dtype=np.float32)
    has_children = np.empty(n, dtype=np.bool_)

    total_children = 0
    for record in records:
        total_children += len(record['child_move'])

    child_offsets = np.empty(n + 1, dtype=np.int64)
    child_move_flat = np.empty(total_children, dtype=np.int32)
    child_policy_flat = np.empty(total_children, dtype=np.float32)

    offset = 0
    for i, record in enumerate(records):
        keys[i] = record['key']
        canonical_keys[i] = record['canonical_key']
        sfen_bytes_list.append(record['sfen_bytes'])
        values[i] = record['value']

        child_offsets[i] = offset
        child_move = record['child_move']
        child_policy = record['child_policy']
        if len(child_move) > 0:
            has_children[i] = True
            nc = len(child_move)
            child_move_flat[offset:offset + nc] = child_move
            child_policy_flat[offset:offset + nc] = child_policy
            offset += nc
        else:
            has_children[i] = False

    child_offsets[n] = offset
    sfen_bytes = np.asarray(sfen_bytes_list)

    out_path = os.path.join(out_dir, f'{shard_prefix}_{shard_index:06d}.npz')
    np.savez(
        out_path,
        keys=keys,
        canonical_keys=canonical_keys,
        sfen_bytes=sfen_bytes,
        values=values,
        has_children=has_children,
        child_offsets=child_offsets,
        child_move_flat=child_move_flat[:offset],
        child_policy_flat=child_policy_flat[:offset],
    )
    return out_path


def parse_eval_sfens(book_path):
    with open(book_path, 'r') as f:
        for line in f:
            if 'sfen' not in line:
                continue
            yield ' '.join(line.split()[1:]).strip()


def eval_sfens_to_dir(session, sfens, batch_size, out_dir, existing_canonical_keys, shard_size):
    x1 = np.empty((batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32)
    x2 = np.empty((batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    run_prefix = f'eval_{int(time.time())}'
    shard_index = 0
    shard_records = []
    batch_boards = []
    batch_canonical_keys = []

    total_seen = 0
    skipped_existing = 0
    saved_positions = 0

    def infer_current_batch():
        nonlocal saved_positions
        if not batch_boards:
            return

        n = len(batch_boards)
        for j in range(batch_size):
            if j < n:
                make_input_features(batch_boards[j], x1[j], x2[j])
            else:
                make_input_features(cshogi.Board(), x1[j], x2[j])

        policy_logits, values = eval(session, x1, x2)

        for j in range(n):
            board = batch_boards[j]
            key = board.zobrist_hash()
            canonical_key = batch_canonical_keys[j]

            child_move = np.asarray(list(board.legal_moves), dtype=np.int32)
            child_policy = make_logits(board, policy_logits[j])
            child_policy = softmax_temperature_with_normalization(child_policy, 1.76).astype(np.float32, copy=False)

            shard_records.append(
                {
                    'key': int(key),
                    'canonical_key': int(canonical_key),
                    'sfen_bytes': board.sfen().encode('ascii'),
                    'value': float(values[j][0]),
                    'child_move': child_move,
                    'child_policy': child_policy,
                }
            )
            existing_canonical_keys.add(int(canonical_key))
            saved_positions += 1

        batch_boards.clear()
        batch_canonical_keys.clear()

    for sfen in sfens:
        total_seen += 1
        board = cshogi.Board(sfen=sfen)
        canonical_key = make_canonical_key(board)
        if canonical_key in existing_canonical_keys:
            skipped_existing += 1
            continue

        batch_boards.append(board)
        batch_canonical_keys.append(canonical_key)

        if len(batch_boards) >= batch_size:
            infer_current_batch()

        if len(shard_records) >= shard_size:
            out_path = save_npz_shard(out_dir, run_prefix, shard_index, shard_records)
            print(f'saved shard: {out_path} ({len(shard_records)} positions)')
            shard_records.clear()
            shard_index += 1

    infer_current_batch()
    if shard_records:
        out_path = save_npz_shard(out_dir, run_prefix, shard_index, shard_records)
        print(f'saved shard: {out_path} ({len(shard_records)} positions)')

    return {
        'total_seen': total_seen,
        'skipped_existing': skipped_existing,
        'saved_positions': saved_positions,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("book")
    parser.add_argument("out_dir")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--shard_size", type=int, default=4096)
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda', 'tensorrt'], default='cpu')
    args = parser.parse_args()

    if args.device == 'cpu':
        providers = ['CPUExecutionProvider']
    elif args.device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:  # tensorrt
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    session = onnxruntime.InferenceSession(args.model, providers=providers)
    batch_size = args.batch_size

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    existing_canonical_keys = load_existing_canonical_keys(args.out_dir)
    print(f'indexed canonical keys: {len(existing_canonical_keys)}')

    stats = eval_sfens_to_dir(
        session,
        parse_eval_sfens(args.book),
        batch_size,
        args.out_dir,
        existing_canonical_keys,
        args.shard_size,
    )
    print(f"total_seen={stats['total_seen']}")
    print(f"skipped_existing={stats['skipped_existing']}")
    print(f"saved_positions={stats['saved_positions']}")
