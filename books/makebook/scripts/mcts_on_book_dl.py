import argparse
import cshogi
from cshogi import NOT_REPETITION, REPETITION_DRAW, REPETITION_WIN, REPETITION_SUPERIOR, BLACK, WHITE
import faulthandler
import numpy as np
import os
import random
import signal
import sys
import time
import tqdm
from collections import OrderedDict

try:
    import mcts_cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

USE_CPP = False

c_puct = 0.1

def score_to_value(score, a=756.0864962951762):
    return 1.0 / (1.0 + np.exp(-score / a))

def softmax_temperature_with_normalization(logits, temperature):
    logits /= temperature
    max_logit = max(logits)
    probabilities = np.exp(logits - max_logit)
    sum_probabilities = sum(probabilities)
    probabilities /= sum_probabilities
    return probabilities

def select_max_ucb_child(node):
    q = np.divide(node.child_score_sum, node.child_move_count, out=np.zeros(len(node.child_move), np.float32), where=node.child_move_count != 0)
    if node.move_count == 0:
        u = 1.0
    else:
        u = np.sqrt(node.move_count / (1 + node.child_move_count))
    ucb = q + c_puct * node.child_policy * u
    return np.argmax(ucb)

def backup(node):
    node.child_q = node.child_score_sum / node.child_move_count

book_tree = dict()
book_child_depth_tree = dict()
dl_data_tree = dict()
dl_shards = []
dl_lookup_keys = np.empty(0, dtype=np.uint64)
dl_lookup_sids = np.empty(0, dtype=np.uint32)
dl_lookup_rows = np.empty(0, dtype=np.uint32)
dl_cache = OrderedDict()
DL_CACHE_SIZE = 50000
depth0_count = 0
DEBUG_MODE = False
DEBUG_SKIP_IS_DRAW = False
DEBUG_TRACE_FILE = None
USE_CSHOGI_IS_DRAW = False
ROTATED_DL_SHARE_STATS = False


def debug_log(message):
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")


def debug_trace(message):
    if not DEBUG_MODE or DEBUG_TRACE_FILE is None:
        return
    with open(DEBUG_TRACE_FILE, "a") as f:
        f.write(message + "\n")
        f.flush()


def to_move_int(move):
    return int(move)


def validate_move_usi(board, move_usi, context):
    if not DEBUG_MODE:
        return

    legal_usi = {cshogi.move_to_usi(move) for move in board.legal_moves}
    if move_usi not in legal_usi:
        raise RuntimeError(
            f"Illegal USI move at {context}: move={move_usi}, turn={board.turn}, sfen={board.sfen()}"
        )


def safe_push(board, move, context):
    move_int = to_move_int(move)
    if DEBUG_MODE:
        legal_moves = set(to_move_int(m) for m in board.legal_moves)
        if move_int not in legal_moves:
            move_usi = cshogi.move_to_usi(move_int)
            raise RuntimeError(
                f"Illegal move at {context}: move={move_int}({move_usi}), turn={board.turn}, sfen={board.sfen()}"
            )
    board.push(move_int)


def safe_is_draw(board, context):
    if DEBUG_SKIP_IS_DRAW:
        return NOT_REPETITION

    debug_trace(
        "before_is_draw"
        f" context={context}"
        f" ply={len(board.history)}"
        f" turn={board.turn}"
        f" sfen={board.sfen()}"
    )
    return board.is_draw()


class Node:
    __slots__ = ('board', 'move_count', 'value', 'sum_value',
                 'child_move', 'child_move_count', 'child_score',
                 'child_score_sum', 'child_policy', 'child_q',
                 'parent_move', 'parent_move_count', 'parent_key')

    def __init__(self):
        self.board = None
        self.move_count = 0
        self.value = 0.0
        self.sum_value = 0.0
        self.child_move = []
        self.child_move_count = []
        self.child_score = []
        self.child_score_sum = []
        self.child_policy = None
        self.child_q = None
        self.parent_move = None
        self.parent_move_count = 0
        self.parent_key = None

    def __getstate__(self):
        return {s: getattr(self, s) for s in self.__slots__}

    def __setstate__(self, state):
        for s in self.__slots__:
            setattr(self, s, state.get(s))


def build_rotated_book_node(node):
    rotated_node = Node()
    rotated_node.board = rotate(node.board).copy()
    rotated_node.child_move = [
        cshogi.to_usi(cshogi.move_rotate(node.board.move_from_usi(move))).decode()
        for move in node.child_move
    ]
    rotated_node.child_move_count = np.zeros(len(rotated_node.child_move))
    rotated_node.child_score = -np.array(node.child_score, dtype=np.float32, copy=True)
    rotated_node.child_score_sum = np.zeros(len(rotated_node.child_move), dtype=np.float32)
    return rotated_node


def get_book_node(board):
    board_key = board.zobrist_hash()
    node = book_tree.get(board_key)
    if node is not None:
        if isinstance(node.board, str):
            node.board = cshogi.Board(sfen=node.board)
        return board_key, node

    rotated_board = rotate(board)
    rotated_board_key = rotated_board.zobrist_hash()
    rotated_node = book_tree.get(rotated_board_key)
    if rotated_node is None:
        return board_key, None

    if isinstance(rotated_node.board, str):
        rotated_node.board = cshogi.Board(sfen=rotated_node.board)
    node = build_rotated_book_node(rotated_node)
    node.board = board.copy()
    book_tree[board_key] = node
    book_child_depth_tree[board_key] = list(book_child_depth_tree.get(rotated_board_key, []))
    return board_key, node


def build_rotated_dl_node(node):
    base_board = node.board if isinstance(node.board, cshogi.Board) else cshogi.Board(sfen=node.board)
    rotated_node = Node()
    rotated_node.board = rotate(base_board).copy()
    # 反転局面の探索統計は、比較実験用に切り替え可能にする。
    if ROTATED_DL_SHARE_STATS:
        rotated_node.move_count = node.move_count
    else:
        rotated_node.move_count = 0
    # 反転局面は評価視点が反転するため、value 系は補数に変換する。
    rotated_node.value = 1.0 - float(node.value)
    if ROTATED_DL_SHARE_STATS:
        rotated_node.sum_value = float(node.move_count) - float(node.sum_value)
    else:
        rotated_node.sum_value = 0.0

    if node.child_move is None:
        rotated_node.child_move = None
        rotated_node.child_move_count = None
        rotated_node.child_score_sum = None
        rotated_node.child_policy = None
        return rotated_node

    rotated_node.child_move = [to_move_int(cshogi.move_rotate(to_move_int(move))) for move in node.child_move]
    if ROTATED_DL_SHARE_STATS:
        rotated_node.child_move_count = np.array(node.child_move_count, copy=True)
        rotated_node.child_score_sum = rotated_node.child_move_count - np.array(node.child_score_sum, copy=True)
    else:
        rotated_node.child_move_count = np.zeros(len(rotated_node.child_move), dtype=np.float32)
        rotated_node.child_score_sum = np.zeros(len(rotated_node.child_move), dtype=np.float32)
    rotated_node.child_policy = np.array(node.child_policy, copy=True)
    return rotated_node


def get_dl_node(board):
    board_key = board.zobrist_hash()
    node = get_dl_node_by_key(board_key)
    if node is not None:
        if not isinstance(node.board, cshogi.Board):
            node.board = board.copy()
        if node.child_move is not None and len(node.child_move) > 0 and not isinstance(node.child_move[0], int):
            node.child_move = [to_move_int(move) for move in node.child_move]
        return board_key, node

    rotated_board = rotate(board)
    rotated_key = rotated_board.zobrist_hash()
    rotated_node = get_dl_node_by_key(rotated_key)
    if rotated_node is None:
        return board_key, None

    node = build_rotated_dl_node(rotated_node)
    node.board = board.copy()
    dl_data_tree[board_key] = node
    return board_key, node


def pick_repetition_detour(path_entries, repeated_key):
    # 指定局面から辿った全経路の候補手から、合法手かつ depth!=9999 の代替手を選ぶ
    # 候補は「最善手との差分が最小」を優先し、同点時はより深い局面を優先する
    best_choice = None
    best_diff = None
    best_depth = None
    for depth, entry in enumerate(path_entries):
        node = entry["node"]
        entry_board = entry["board_snapshot"]
        if not node.child_move or len(node.child_move) < 2:
            continue

        node_child_depth = book_child_depth_tree.get(entry["key"])
        legal_usi = {cshogi.move_to_usi(move) for move in entry_board.legal_moves}

        best_score = node.child_score[0]
        for child_index, score in enumerate(node.child_score):
            if child_index == 0:
                continue

            candidate_move = node.child_move[child_index]
            if candidate_move not in legal_usi:
                continue

            candidate_board = entry_board.copy()
            validate_move_usi(candidate_board, candidate_move, "pick_repetition_detour")
            candidate_board.push_usi(candidate_move)
            candidate_depth = None
            if node_child_depth is not None and child_index < len(node_child_depth):
                candidate_depth = node_child_depth[child_index]

            if candidate_depth == 9999:
                continue

            diff = abs(float(score) - float(best_score))
            if best_diff is None or diff < best_diff or (diff == best_diff and (best_depth is None or depth > best_depth)):
                best_diff = diff
                best_depth = depth
                best_choice = (entry, child_index, candidate_board)

    return best_choice

def select_root_board(sfen='', turn=BLACK, eval_diff=0, book_moves_threshold=4):
    root_board = cshogi.Board(sfen=sfen)
    root_key, current_node = get_book_node(root_board)
    if current_node is None:
        raise KeyError(f"Root board is not in book tree: {root_board.sfen()}")
    current_node.board = root_board.copy()
    current_key = root_key
    root_board_val = current_node.child_score[0]

    val_sum_threshold = 0.95
    next_board = current_node.board.copy()
    path_entries = []
    seen_path_keys = {current_key}
    detour_applied = False
    while True:
        path_entries.append({"key": current_key, "node": current_node, "board_snapshot": current_node.board.copy()})

        # policyの上位何手でval_sum_thresholdを超えるか確認する
        _, current_dl_node = get_dl_node(current_node.board)
        if current_dl_node is None or current_dl_node.child_policy is None:
            print("Current board is not in dl_data_tree. Stop at current board.")
            break

        child_value_sorted = np.sort(current_dl_node.child_policy)[::-1]
        val_sum_threshold_count = 0
        val_sum = 0.0
        while val_sum < val_sum_threshold and val_sum_threshold_count < len(child_value_sorted):
            val_sum += child_value_sorted[val_sum_threshold_count]
            val_sum_threshold_count += 1

        # val_sum_thresholdを超える手が閾値未満なら手を進める
        if val_sum_threshold_count >= book_moves_threshold and len(current_node.child_move) < book_moves_threshold:
            print(f"Reached threshold at depth with {len(current_node.child_move)} moves")
            break

        if current_node.board.turn == turn or eval_diff == 0:
            # ペタショック化された定跡はソート済みなので必ず先頭がbestmoveになっている
            best_child_index = 0
        else:
            search_moves_list = [index for index, score in enumerate(current_node.child_score)
                                 if root_board_val - eval_diff <= score <= root_board_val]
            if len(search_moves_list) > 0:
                best_child_index = random.choice(search_moves_list)
            else:
                best_child_index = 0

        best_move = current_node.child_move[best_child_index]
        next_board = current_node.board.copy()
        validate_move_usi(next_board, best_move, "select_root_board.best_move")
        next_board.push_usi(best_move)
        next_board_key_in_path = next_board.zobrist_hash()

        # 同一局面の再訪を千日手として扱い、最善手に近い代替手へ迂回する
        if eval_diff == 0 and next_board_key_in_path in seen_path_keys:
            if not detour_applied:
                detour = pick_repetition_detour(path_entries, next_board_key_in_path)
                if detour is not None:
                    detour_entry, detour_child_index, detour_board = detour
                    detour_node = detour_entry["node"]
                    detour_key = detour_entry["key"]
                    detour_move = detour_node.child_move[detour_child_index]

                    print(
                        f"Repetition detected. Use closest-eval detour move: {detour_move} "
                        f"(score={detour_node.child_score[detour_child_index]}, best={detour_node.child_score[0]})"
                    )

                    next_board = detour_board
                    next_board_key, next_node = get_book_node(next_board)
                    current_key = next_board_key

                    if next_node is None:
                        # 探索開始局面が定跡ツリーに登録されていなかったら1手戻した局面を探索開始局面にする
                        next_board.pop()
                        next_board_key = next_board.zobrist_hash()
                        current_key = next_board_key
                        break

                    current_node = next_node
                    current_node.board = next_board.copy()  # history保持のためboardごとコピーする
                    detour_applied = True
                    path_entries = [
                        {"key": detour_key, "node": detour_node, "board_snapshot": detour_node.board.copy()},
                        {"key": current_key, "node": current_node, "board_snapshot": current_node.board.copy()},
                    ]
                    seen_path_keys = {detour_key, current_key}
                    continue

            print("Repetition detected but no detour candidate found. Stop at current board.")
            break

        next_board_key, next_node = get_book_node(next_board)
        current_key = next_board_key
        if next_node is None:
            # 探索開始局面が定跡ツリーに登録されていなかったら1手戻した局面を探索開始局面にする
            next_board.pop()
            next_board_key = next_board.zobrist_hash()
            current_key = next_board_key
            break
        current_node = next_node
        current_node.board = next_board.copy() # history保持のためboardごとコピーする
        seen_path_keys.add(current_key)
        root_board_val *= -1

    first_board = next_board
    first_board_key = current_key
    return first_board, first_board_key

# DLで推論したツリー上でPV-MCTSを行う
# valueについては定跡ツリーに登録されていれば定跡ツリー上の値を優先する
visited_nodes = set()
def search(node, path_keys=None):
    global depth0_count
    if path_keys is None:
        path_keys = set()

    node_key = node.board.zobrist_hash()
    if node_key in path_keys:
        # 同一探索経路で局面を再訪したら千日手として扱う
        return 0.5

    path_keys.add(node_key)

    node.move_count += 1

    try:
        if not node.child_move:
            visited_nodes.add(node_key)
            return node.value

        search_node = select_max_ucb_child(node)
        node.child_move_count[search_node] += 1

        next_board = node.board.copy()
        safe_push(next_board, node.child_move[search_node], "search.next_board")
        next_board_key = next_board.zobrist_hash()

        # 同一探索経路で再訪したら千日手
        if next_board_key in path_keys:
            # 千日手
            return 0.5

        # 互換性のため、必要時のみ cshogi 側の判定を使う
        if USE_CSHOGI_IS_DRAW:
            draw = safe_is_draw(next_board, "search")
            if draw != NOT_REPETITION:
                if draw == REPETITION_DRAW:
                    return 0.5
                elif draw == REPETITION_WIN or draw == REPETITION_SUPERIOR:
                    return 1.0
                else:
                    return 0.0

        # 次の局面が定跡ツリーに登録されていなければ定跡ツリーに追加する
        _, existing_next_node = get_dl_node(next_board)
        if existing_next_node is None:
            _, current_dl_node = get_dl_node(node.board)
            if current_dl_node is None:
                raise KeyError(f"Current board is not in dl_data_tree: {node.board.sfen()}")

            dl_data_tree[next_board_key] = Node()
            dl_data_tree[next_board_key].board = next_board.copy()
            dl_data_tree[next_board_key].child_move = None
            # 次の局面については未評価なので1-(現局面の評価値)で仮置きする
            dl_data_tree[next_board_key].value = 1.0 - current_dl_node.value

        # 次の局面が末端ノードの場合定跡ツリーに登録されているか確認し、登録されていれば定跡ツリーの値で置き換える
        next_node_for_leaf = existing_next_node if existing_next_node is not None else dl_data_tree[next_board_key]
        if not next_node_for_leaf.child_move:
            _, current_book_node = get_book_node(node.board)
            if current_book_node is not None and node.child_move[search_node] in current_book_node.child_move:
                index = current_book_node.child_move.index(node.child_move[search_node])
                next_node_for_leaf.value = 1.0 - score_to_value(current_book_node.child_score[index])
            depth0_count += 1

        _, next_node = get_dl_node(next_board)
        if next_node is None:
            raise KeyError(f"Next board is not in dl_data_tree: {next_board.sfen()}")

        next_node.board = next_board # history保持のためboardごとコピーする
        value = search(next_node, path_keys)
        value = 1.0 - value

        node.sum_value += value
        node.child_score_sum[search_node] += value
        return value
    finally:
        path_keys.remove(node_key)

def get_history(node, history=None):
    if history is None:
        history = []
    if node.parent_key is None or node.board.zobrist_hash() == cshogi.Board().zobrist_hash():
        return
    history.append(node.parent_move)
    parent_node = book_tree[node.parent_key]
    get_history(parent_node, history)

def rotate(board):
    return cshogi.Board(cshogi.rotate_sfen(board.sfen()))


def list_npz_files(path):
    if not os.path.isdir(path):
        raise NotADirectoryError(f"Expected directory path for dl_dir: {path}")

    files = [os.path.join(path, name) for name in os.listdir(path) if name.endswith('.npz')]
    files.sort()
    return files


def _cache_get(key):
    node = dl_cache.get(key)
    if node is None:
        return None
    dl_cache.move_to_end(key)
    return node


def _cache_put(key, node):
    dl_cache[key] = node
    dl_cache.move_to_end(key)
    if len(dl_cache) > DL_CACHE_SIZE:
        dl_cache.popitem(last=False)


def _find_dl_location(key):
    if len(dl_lookup_keys) == 0:
        return None

    key_u64 = np.uint64(key)
    idx = int(np.searchsorted(dl_lookup_keys, key_u64, side='left'))
    if idx >= len(dl_lookup_keys) or dl_lookup_keys[idx] != key_u64:
        return None

    return int(dl_lookup_sids[idx]), int(dl_lookup_rows[idx])


def _load_node_from_location(shard_id, row_index):
    shard = dl_shards[shard_id]
    node = Node()

    sb = shard['sfen_bytes'][row_index]
    node.board = sb.decode('ascii') if isinstance(sb, bytes) else str(sb)
    node.value = float(shard['values'][row_index])

    if bool(shard['has_children'][row_index]):
        start = int(shard['child_offsets'][row_index])
        end = int(shard['child_offsets'][row_index + 1])
        node.child_move = np.asarray(shard['child_move_flat'][start:end], dtype=np.int32).tolist()
        node.child_move_count = np.zeros(end - start, dtype=np.float32)
        node.child_score_sum = np.zeros(end - start, dtype=np.float32)
        node.child_policy = np.asarray(shard['child_policy_flat'][start:end], dtype=np.float32).copy()
    else:
        node.child_move = None

    return node


def get_dl_node_by_key(key):
    if key in dl_data_tree:
        return dl_data_tree[key]

    node = _cache_get(key)
    if node is not None:
        return node

    location = _find_dl_location(key)
    if location is None:
        return None

    node = _load_node_from_location(location[0], location[1])
    _cache_put(key, node)
    return node


def load_dl_data_tree_npz_dir(path):
    global dl_shards, dl_lookup_keys, dl_lookup_sids, dl_lookup_rows

    dl_shards = []
    dl_lookup_keys = np.empty(0, dtype=np.uint64)
    dl_lookup_sids = np.empty(0, dtype=np.uint32)
    dl_lookup_rows = np.empty(0, dtype=np.uint32)
    dl_cache.clear()

    npz_files = list_npz_files(path)
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No npz files found under: {path}")

    all_keys = []
    all_sids = []
    all_rows = []

    for shard_id, npz_path in enumerate(tqdm.tqdm(npz_files, desc="Open dl npz", dynamic_ncols=True)):
        try:
            npz = np.load(npz_path, allow_pickle=False, mmap_mode='r')
        except ValueError:
            npz = np.load(npz_path, allow_pickle=True, mmap_mode='r')

        keys = np.asarray(npz['keys'], dtype=np.uint64)
        n = len(keys)
        dl_shards.append(
            {
                'npz': npz,
                'keys': keys,
                'sfen_bytes': npz['sfen_bytes'],
                'values': npz['values'],
                'has_children': npz['has_children'],
                'child_offsets': npz['child_offsets'],
                'child_move_flat': npz['child_move_flat'],
                'child_policy_flat': npz['child_policy_flat'],
            }
        )

        all_keys.append(keys)
        all_sids.append(np.full(n, shard_id, dtype=np.uint32))
        all_rows.append(np.arange(n, dtype=np.uint32))

    merged_keys = np.concatenate(all_keys)
    merged_sids = np.concatenate(all_sids)
    merged_rows = np.concatenate(all_rows)

    order = np.argsort(merged_keys, kind='mergesort')
    keys_sorted = merged_keys[order]
    sids_sorted = merged_sids[order]
    rows_sorted = merged_rows[order]

    # Keep the last duplicate key so newer shards override older entries.
    if len(keys_sorted) > 1:
        keep = np.ones(len(keys_sorted), dtype=np.bool_)
        keep[:-1] = keys_sorted[:-1] != keys_sorted[1:]
        keys_sorted = keys_sorted[keep]
        sids_sorted = sids_sorted[keep]
        rows_sorted = rows_sorted[keep]

    dl_lookup_keys = keys_sorted
    dl_lookup_sids = sids_sorted
    dl_lookup_rows = rows_sorted

    return len(dl_lookup_keys)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('book')
    args.add_argument('dl_dir')
    args.add_argument('sfens')
    args.add_argument('--boards', type=str)
    args.add_argument('--root_sfens', type=str, nargs='*', default=[])
    args.add_argument('--book_moves_threshold', type=int, default=4)
    args.add_argument('--eval_diff', type=int, default=30)
    args.add_argument('--first_board_sfen_output', type=str, default='first_board_sfens.txt')
    args.add_argument('--debug', action='store_true')
    args.add_argument('--debug-skip-is-draw', action='store_true')
    args.add_argument('--debug-trace-file', type=str, default='mcts_debug_trace.log')
    args.add_argument('--use-cshogi-is-draw', action='store_true')
    args.add_argument('--rotated-dl-share-stats', action='store_true')
    args.add_argument('--use-cpp', action='store_true')
    args.add_argument('--dl-cache-size', type=int, default=50000)
    args.add_argument('--visited-nodes-limit', type=int, default=1000 * 10)
    args = args.parse_args()

    DL_CACHE_SIZE = args.dl_cache_size

    if args.use_cpp:
        if _HAS_CPP:
            USE_CPP = True
        else:
            print("WARNING: --use-cpp specified but mcts_cpp not found. Falling back to pure Python.")

    ROTATED_DL_SHARE_STATS = args.rotated_dl_share_stats
    USE_CSHOGI_IS_DRAW = args.use_cshogi_is_draw
    DEBUG_SKIP_IS_DRAW = args.debug_skip_is_draw
    DEBUG_MODE = args.debug
    DEBUG_TRACE_FILE = args.debug_trace_file if args.debug else None
    if DEBUG_MODE:
        faulthandler.enable(all_threads=True)
        try:
            faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
        except Exception:
            pass
        debug_log("Debug mode is enabled")
        if DEBUG_SKIP_IS_DRAW:
            debug_log("is_draw checks are skipped (--debug-skip-is-draw)")
        if USE_CSHOGI_IS_DRAW:
            debug_log("cshogi is_draw is enabled (--use-cshogi-is-draw)")
        if ROTATED_DL_SHARE_STATS:
            debug_log("rotated dl node shares search stats (--rotated-dl-share-stats)")
        if DEBUG_TRACE_FILE is not None:
            with open(DEBUG_TRACE_FILE, "w") as f:
                f.write("start_debug_trace\n")

    book_parse_start = time.time()
    if USE_CPP and hasattr(mcts_cpp, 'parse_book_cpp'):
        book_tree, book_child_depth_tree = mcts_cpp.parse_book_cpp(args.book, Node)
        print(f"Book parse (C++): {time.time() - book_parse_start:.2f}s ({len(book_tree)} positions)")
    else:
        with open(args.book, "r") as f:
            books = f.readlines()
            books = [s.replace("\n", "") for s in books[1:]]

        # 定跡をパースする
        board = cshogi.Board()
        board_key = None
        for book in books:
            if book.startswith("sfen"):
                # 直前までのbook_tree[board_key]のchile_move_count, child_score, child_score_sumをnumpy配列に変換する
                if board_key is not None:
                    book_tree[board_key].child_move_count = np.zeros(len(book_tree[board_key].child_move))
                    book_tree[board_key].child_score = np.array(book_tree[board_key].child_score, dtype=np.float32)
                    book_tree[board_key].child_score_sum = np.zeros(len(book_tree[board_key].child_move), dtype=np.float32)

                board.set_sfen(" ".join(book.split(" ")[1:]))
                board_key = board.zobrist_hash()
                book_tree[board_key] = Node()
                book_tree[board_key].board = board.sfen()
                book_child_depth_tree[board_key] = []
            else:
                next_move_info = book.strip().split(" ")
                move_usi = next_move_info[0]
                book_tree[board_key].child_move.append(move_usi)
                book_tree[board_key].child_score.append(int(next_move_info[2]))

                move_depth = None
                for token in next_move_info[3:]:
                    try:
                        move_depth = int(token)
                        break
                    except ValueError:
                        continue
                book_child_depth_tree[board_key].append(move_depth)

        book_tree[board_key].child_move_count = np.zeros(len(book_tree[board_key].child_move))
        book_tree[board_key].child_score = np.array(book_tree[board_key].child_score, dtype=np.float32)
        book_tree[board_key].child_score_sum = np.zeros(len(book_tree[board_key].child_move), dtype=np.float32)
        book_parse_elapsed = time.time() - book_parse_start
        print(f"Book parse: {book_parse_elapsed:.2f}s ({len(book_tree)} positions)")

    # DLで評価したノードを読み込む（mmap + 遅延ロード）
    pickle_load_start = time.time()
    indexed_count = load_dl_data_tree_npz_dir(args.dl_dir)
    pickle_load_elapsed = time.time() - pickle_load_start
    print(f"DL index load: {pickle_load_elapsed:.2f}s ({indexed_count} nodes, cache={DL_CACHE_SIZE})")

    if USE_CPP:
        mcts_cpp.init(dl_data_tree, book_tree, visited_nodes,
                      USE_CSHOGI_IS_DRAW, get_dl_node, get_book_node, Node)
        search_func = mcts_cpp.search_cpp
        print("Using C++-optimized MCTS")
    else:
        search_func = search
        print("Using pure Python MCTS")

    # ルート局面から定跡ツリー上で最善手を辿り、登録されている候補手が閾値を初めて下回った局面をfirst_boardとする
    root_board_sfen_list = ['']
    for root_sfens in args.root_sfens:
        if os.path.exists(root_sfens):
            with open(root_sfens, "r") as f:
                root_board_sfen_list += [s.replace("\n", "") for s in f.readlines()]

    first_board_sfen_list = []
    bestmove_board_sfen_list = []
    playout_num = 200000
    total_playout_count = 0
    search_total_start = time.time()
    for root_sfen in root_board_sfen_list:
        first_board, first_board_key = select_root_board(sfen=root_sfen, book_moves_threshold=args.book_moves_threshold)
        first_board_sfen_list.append(f"{first_board.sfen()}\n")
        print(f"Root sfen: {root_sfen}")
        print(f"search board history: {' '.join([cshogi.move_to_usi(move) for move in first_board.history])}")

        _, first_dl_node = get_dl_node(first_board)
        if first_dl_node is None:
            continue

        first_dl_node.board = first_board.copy()

        # 探索開始局面における最善手の局面を探索対象局面に含める
        bestmove_board = first_board.copy()
        _, first_book_node = get_book_node(first_board)
        if first_book_node is None:
            continue

        bestmove_board.push_usi(first_book_node.child_move[0])
        _, bestmove_book_node = get_book_node(bestmove_board)
        if bestmove_book_node is None:
            bestmove_board_sfen_list.append(f"sfen {bestmove_board.sfen()}\n")

        count = 0
        print("Starting search...")
        pbar = tqdm.tqdm(desc="MCTS", dynamic_ncols=True)
        while count < playout_num:
            search_func(first_dl_node)
            pbar.update(1)
            count += 1
        pbar.close()
        total_playout_count += playout_num

    cnt = 0
    while len(visited_nodes) < args.visited_nodes_limit:
        cnt += 1
        if cnt % 2 == 0:
            turn = BLACK
        else:
            turn = WHITE
        first_board, first_board_key = select_root_board(turn=turn, eval_diff=args.eval_diff, book_moves_threshold=args.book_moves_threshold)
        first_board_sfen_list.append(f"{first_board.sfen()}\n")
        print(f"search board history: {' '.join([cshogi.move_to_usi(move) for move in first_board.history])}")

        _, first_dl_node = get_dl_node(first_board)
        if first_dl_node is None:
            continue

        first_dl_node.board = first_board.copy()

        # 探索開始局面における最善手の局面を探索対象局面に含める
        bestmove_board = first_board.copy()
        _, first_book_node = get_book_node(first_board)
        if first_book_node is None:
            continue

        bestmove_board.push_usi(first_book_node.child_move[0])
        _, bestmove_book_node = get_book_node(bestmove_board)
        if bestmove_book_node is None:
            bestmove_board_sfen_list.append(f"sfen {bestmove_board.sfen()}\n")

        count = 0
        print("Starting search...")
        pbar = tqdm.tqdm(desc="MCTS", dynamic_ncols=True)
        while count < playout_num:
            search_func(first_dl_node)
            pbar.update(1)
            count += 1
        pbar.close()
        total_playout_count += playout_num

    search_total_elapsed = time.time() - search_total_start
    print(f"Total search: {search_total_elapsed:.2f}s ({total_playout_count} playouts, {total_playout_count/search_total_elapsed:.0f} playouts/sec)")

    if USE_CPP:
        mcts_cpp.sync_cpp_to_python()

    print(f"visited_nodes: {len(visited_nodes)}")
    move_count_list = []
    for key in visited_nodes:
        node = get_dl_node_by_key(key)
        if node is None:
            continue
        move_count_list.append((node.move_count, key))
    move_count_list.sort(reverse=True)
    move_count_list = move_count_list[:min(len(move_count_list), 1000)]

    sfens_list = []
    moves_list = []
    for _, key in move_count_list:
        node = get_dl_node_by_key(key)
        if node is None:
            continue
        if not isinstance(node.board, cshogi.Board):
            node.board = cshogi.Board(sfen=node.board)
        sfens_list.append(f"sfen {node.board.sfen()}\n")
        if args.boards:
            board = node.board.copy()
            history = " ".join([cshogi.move_to_usi(move) for move in board.history])
            current_sfen = board.sfen()
            moves_list.append(f"{current_sfen}, {history}\n")

    sfens_list += bestmove_board_sfen_list
    with open(args.sfens, "w") as f:
        f.writelines(sfens_list)

    with open(args.first_board_sfen_output, "w") as f:
        f.writelines(first_board_sfen_list)

    if args.boards:
        with open(args.boards, "w") as f:
            f.writelines(moves_list)
