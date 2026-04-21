import argparse
from eval import *
from mcts_on_book_dl import Node, softmax_temperature_with_normalization
import os
import pickle
import tqdm


def load_out_data(path):
    if not os.path.exists(path):
        return None

    if path.endswith('.npz'):
        try:
            with np.load(path, allow_pickle=False) as npz:
                keys = npz['keys']
                sfen_bytes = npz['sfen_bytes']
                values = npz['values']
                has_children = npz['has_children']
                child_offsets = npz['child_offsets']
                child_move_flat = npz['child_move_flat']
                child_policy_flat = npz['child_policy_flat']
        except ValueError:
            # Backward compatibility for old files that stored object arrays.
            with np.load(path, allow_pickle=True) as npz:
                keys = npz['keys']
                sfen_bytes = npz['sfen_bytes']
                values = npz['values']
                has_children = npz['has_children']
                child_offsets = npz['child_offsets']
                child_move_flat = npz['child_move_flat']
                child_policy_flat = npz['child_policy_flat']

        out = {}
        for i in range(len(keys)):
            node = Node()
            node.board = sfen_bytes[i].decode('ascii') if len(sfen_bytes[i]) > 0 else ''
            node.value = float(values[i])

            if bool(has_children[i]):
                start = int(child_offsets[i])
                end = int(child_offsets[i + 1])
                node.child_move = child_move_flat[start:end].astype(np.int32).tolist()
                node.child_policy = child_policy_flat[start:end].astype(np.float32)
                node.child_move_count = np.zeros(end - start, dtype=np.float32)
                node.child_score_sum = np.zeros(end - start, dtype=np.float32)
            else:
                node.child_move = []
                node.child_policy = None
                node.child_move_count = np.zeros(0, dtype=np.float32)
                node.child_score_sum = np.zeros(0, dtype=np.float32)

            out[int(keys[i])] = node
        return out

    with open(path, 'rb') as f:
        return pickle.load(f)


def save_out_data(path, out):
    if path.endswith('.npz'):
        n = len(out)
        keys = np.empty(n, dtype=np.uint64)
        sfen_list = []
        values = np.empty(n, dtype=np.float32)
        has_children = np.empty(n, dtype=np.bool_)

        total_children = 0
        for node in out.values():
            if node.child_move is not None and len(node.child_move) > 0:
                total_children += len(node.child_move)

        child_offsets = np.empty(n + 1, dtype=np.int64)
        child_move_flat = np.empty(total_children, dtype=np.int32)
        child_policy_flat = np.empty(total_children, dtype=np.float32)

        offset = 0
        for i, (key, node) in enumerate(out.items()):
            keys[i] = key
            values[i] = float(node.value)

            if isinstance(node.board, str):
                board_sfen = node.board
            elif node.board is not None:
                board_sfen = node.board.sfen()
            else:
                board_sfen = ''
            sfen_list.append(board_sfen.encode('ascii') if board_sfen else b'')

            child_offsets[i] = offset
            if node.child_move is not None and len(node.child_move) > 0:
                nc = len(node.child_move)
                has_children[i] = True
                child_move_flat[offset:offset + nc] = np.asarray(node.child_move, dtype=np.int32)
                if node.child_policy is not None:
                    child_policy_flat[offset:offset + nc] = np.asarray(node.child_policy, dtype=np.float32)
                else:
                    child_policy_flat[offset:offset + nc] = 0.0
                offset += nc
            else:
                has_children[i] = False

        child_offsets[n] = offset
        sfen_bytes = np.asarray(sfen_list)
        np.savez(
            path,
            keys=keys,
            sfen_bytes=sfen_bytes,
            values=values,
            has_children=has_children,
            child_offsets=child_offsets,
            child_move_flat=child_move_flat[:offset],
            child_policy_flat=child_policy_flat[:offset],
        )
        return

    with open(path, 'wb') as f:
        pickle.dump(out, f, protocol=5)

# sfen文字列のリストに対し推論を行う
def eval_sfens(session, sfens, batch_size, out=None):
    x1 = np.empty((batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32)
    x2 = np.empty((batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32)
    # ダミー推論をして推論を速くする
    # eval(session, x1, x2)

    if out is None:
        out = dict()
    eval_board_list = []
    for sfen in sfens:
        board = cshogi.Board(sfen=sfen)
        key = board.zobrist_hash()
        if key not in out:
            eval_board_list.append(board)
        elif out[key].child_policy is None:
            eval_board_list.append(board)

    for i in tqdm.tqdm(range(0, len(eval_board_list), batch_size)):
        for j in range(batch_size):
            if i + j < len(eval_board_list):
                make_input_features(eval_board_list[i + j], x1[j], x2[j])
            else:
                make_input_features(cshogi.Board(), x1[j], x2[j])
        policy_logits, values = eval(session, x1, x2)
        for j in range(batch_size):
            if i + j < len(eval_board_list):
                node = Node()
                node.board = eval_board_list[i + j].sfen()
                node.child_move = list(eval_board_list[i + j].legal_moves)
                node.child_move_count = np.zeros(len(node.child_move), dtype=np.float32)
                node.child_score_sum = np.zeros(len(node.child_move), dtype=np.float32)
                node.child_policy = make_logits(eval_board_list[i + j], policy_logits[j])
                node.child_policy = softmax_temperature_with_normalization(node.child_policy, 1.76)
                node.value = values[j][0]
                out[eval_board_list[i + j].zobrist_hash()] = node
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("book")
    parser.add_argument("pickle")
    parser.add_argument("--batch_size", type=int, default=1)
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

    out = load_out_data(args.pickle)

    eval_board_list = []
    with open(args.book, "r") as f:
        lines = f.readlines()
        eval_sfen_list = [" ".join(line.split()[1:]).replace("\n", "") for line in lines if "sfen" in line]

    out = eval_sfens(session, eval_sfen_list, batch_size, out)
    save_out_data(args.pickle, out)
