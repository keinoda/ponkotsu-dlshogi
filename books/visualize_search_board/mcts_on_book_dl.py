import argparse
import cshogi
from cshogi import NOT_REPETITION, REPETITION_DRAW, REPETITION_WIN, REPETITION_SUPERIOR, BLACK, WHITE
import numpy as np
import os
import pickle
import random
import tqdm

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
dl_data_tree = dict()
depth0_count = 0


class Node:
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
        self.parent_move = None
        self.parent_move_count = 0
        self.parent_key = None


def pick_repetition_detour(path_entries, repeated_key):
    # 千日手ルート内の全候補手から、最善手(先頭)に最も近い評価値の代替手を選ぶ
    cycle_start = 0
    for index, entry in enumerate(path_entries):
        if entry["key"] == repeated_key:
            cycle_start = index
            break

    best_choice = None
    best_diff = None
    for entry in path_entries[cycle_start:]:
        node = entry["node"]
        if not node.child_move or len(node.child_move) < 2:
            continue

        best_score = node.child_score[0]
        for child_index, score in enumerate(node.child_score):
            if child_index == 0:
                continue
            diff = abs(float(score) - float(best_score))
            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_choice = (entry, child_index)

    return best_choice

def select_root_board(sfen='', turn=BLACK, eval_diff=0, book_moves_threshold=4):
    root_board = cshogi.Board(sfen=sfen)
    root_key = root_board.zobrist_hash()
    current_node = book_tree[root_key]
    current_node.board = root_board.copy()
    current_key = root_key
    root_board_val = book_tree[root_key].child_score[0]

    val_sum_threshold = 0.95
    next_board = current_node.board.copy()
    path_entries = []
    detour_applied = False
    while True:
        path_entries.append({"key": current_key, "node": current_node})

        # policyの上位何手でval_sum_thresholdを超えるか確認する
        child_value_sorted = np.sort(dl_data_tree[current_key].child_policy)[::-1]
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
        next_board.push_usi(best_move)

        # 千日手のとき千日手ルート内の全候補手から、最善手(先頭)に最も近い評価値の代替手を選ぶ
        if eval_diff == 0 and next_board.is_draw() == REPETITION_DRAW:
            if not detour_applied:
                detour = pick_repetition_detour(path_entries, next_board.zobrist_hash())
                if detour is not None:
                    detour_entry, detour_child_index = detour
                    detour_node = detour_entry["node"]
                    detour_key = detour_entry["key"]
                    detour_move = detour_node.child_move[detour_child_index]

                    print(
                        f"Repetition detected. Use closest-eval detour move: {detour_move} "
                        f"(score={detour_node.child_score[detour_child_index]}, best={detour_node.child_score[0]})"
                    )

                    next_board = detour_node.board.copy()
                    next_board.push_usi(detour_move)
                    next_board_key = next_board.zobrist_hash()
                    current_key = next_board_key

                    if current_key not in book_tree:
                        # 探索開始局面が定跡ツリーに登録されていなかったら1手戻した局面を探索開始局面にする
                        next_board.pop()
                        next_board_key = next_board.zobrist_hash()
                        current_key = next_board_key
                        break

                    current_node = book_tree[next_board_key]
                    current_node.board = next_board  # history保持のためboardごとコピーする
                    detour_applied = True
                    path_entries = [{"key": detour_key, "node": detour_node}, {"key": current_key, "node": current_node}]
                    continue

            print("Repetition detected but no detour candidate found. Stop at current board.")
            break

        next_board_key = next_board.zobrist_hash()
        current_key = next_board_key
        if current_key not in book_tree:
            # 探索開始局面が定跡ツリーに登録されていなかったら1手戻した局面を探索開始局面にする
            next_board.pop()
            next_board_key = next_board.zobrist_hash()
            current_key = next_board_key
            break
        current_node = book_tree[next_board_key]
        current_node.board = next_board # history保持のためboardごとコピーする
        root_board_val *= -1

    first_board = next_board
    first_board_key = current_key
    return first_board, first_board_key

# DLで推論したツリー上でPV-MCTSを行う
# valueについては定跡ツリーに登録されていれば定跡ツリー上の値を優先する
visited_nodes = set()
def search(node):
    global depth0_count
    node.move_count += 1

    if not node.child_move:
        visited_nodes.add(node.board.zobrist_hash())
        return node.value

    search_node = select_max_ucb_child(node)
    node.child_move_count[search_node] += 1

    next_board = node.board.copy()
    next_board.push(node.child_move[search_node])
    next_board_key = next_board.zobrist_hash()

    # 引き分けの判定
    draw = next_board.is_draw()
    if draw != NOT_REPETITION:
        if draw == REPETITION_DRAW:
            # 千日手
            return 0.5
        elif draw == REPETITION_WIN or draw == REPETITION_SUPERIOR:
            # 連続王手の千日手で勝ちもしくは優越局面
            return 1.0
        else:
            # 連続王手の千日手で負けもしくは劣等局面
            return 0.0

    # 次の局面が定跡ツリーに登録されていなければ定跡ツリーに追加する
    if next_board_key not in dl_data_tree:
        dl_data_tree[next_board_key] = Node()
        dl_data_tree[next_board_key].board = next_board.copy()
        dl_data_tree[next_board_key].child_move = None
        # 次の局面については未評価なので1-(現局面の評価値)で仮置きする
        dl_data_tree[next_board_key].value = 1.0 - dl_data_tree[node.board.zobrist_hash()].value

    # 次の局面が末端ノードの場合定跡ツリーに登録されているか確認し、登録されていれば定跡ツリーの値で置き換える
    if not dl_data_tree[next_board_key].child_move:
        if node.board.zobrist_hash() in book_tree and node.child_move[search_node] in book_tree[node.board.zobrist_hash()].child_move:
            index = book_tree[node.board.zobrist_hash()].child_move.index(node.child_move[search_node])
            dl_data_tree[next_board_key].value = 1.0 - score_to_value(book_tree[node.board.zobrist_hash()].child_score[index])
        depth0_count += 1

    next_node = dl_data_tree[next_board.zobrist_hash()]
    next_node.board = next_board # history保持のためboardごとコピーする
    value = search(next_node)
    value = 1.0 - value

    node.sum_value += value
    node.child_score_sum[search_node] += value
    return value

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

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('book')
    args.add_argument('dl_pickle')
    args.add_argument('sfens')
    args.add_argument('--boards', type=str)
    args.add_argument('--root_sfens', type=str, nargs='*', default=[])
    args.add_argument('--book_moves_threshold', type=int, default=4)
    args.add_argument('--eval_diff', type=int, default=30)
    args.add_argument('--first_board_sfen_output', type=str, default='first_board_sfens.txt')
    args = args.parse_args()

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
            book_tree[board_key].board = board.copy()
        else:
            next_move_info = book.strip().split(" ")
            move_usi = next_move_info[0]
            book_tree[board_key].child_move.append(move_usi)
            book_tree[board_key].child_score.append(int(next_move_info[2]))

    book_tree[board_key].child_move_count = np.zeros(len(book_tree[board_key].child_move))
    book_tree[board_key].child_score = np.array(book_tree[board_key].child_score, dtype=np.float32)
    book_tree[board_key].child_score_sum = np.zeros(len(book_tree[board_key].child_move), dtype=np.float32)

    # 反転が含まれていなければ追加する
    book_tree_rotated = dict()
    for key in book_tree.keys():
        board = book_tree[key].board
        rotated_board = rotate(board)
        rotated_board_key = rotated_board.zobrist_hash()
        if rotated_board_key not in book_tree:
            book_tree_rotated[rotated_board_key] = Node()
            book_tree_rotated[rotated_board_key].board = rotated_board.copy()
            book_tree_rotated[rotated_board_key].child_move = [cshogi.to_usi(cshogi.move_rotate(board.move_from_usi(move))).decode() for move in book_tree[key].child_move]
            book_tree_rotated[rotated_board_key].child_move_count = np.zeros(len(book_tree[key].child_move))
            book_tree_rotated[rotated_board_key].child_score = -book_tree[key].child_score
            book_tree_rotated[rotated_board_key].child_score_sum = np.zeros(len(book_tree[key].child_move), dtype=np.float32)

    book_tree.update(book_tree_rotated)

    # DLで評価したノードを読み込む
    with open(args.dl_pickle, "rb") as f:
        dl_data_tree = pickle.load(f)

    # ルート局面から定跡ツリー上で最善手を辿り、登録されている候補手が閾値を初めて下回った局面をfirst_boardとする
    root_board_sfen_list = ['']
    for root_sfens in args.root_sfens:
        if os.path.exists(root_sfens):
            with open(root_sfens, "r") as f:
                root_board_sfen_list += [s.replace("\n", "") for s in f.readlines()]

    first_board_sfen_list = []
    bestmove_board_sfen_list = []
    playout_num = 200000
    for root_sfen in root_board_sfen_list:
        first_board, first_board_key = select_root_board(sfen=root_sfen, book_moves_threshold=args.book_moves_threshold)
        first_board_sfen_list.append(f"{first_board.sfen()}\n")
        print(f"Root sfen: {root_sfen}")
        print(f"search board history: {' '.join([cshogi.move_to_usi(move) for move in first_board.history])}")

        dl_data_tree[first_board_key].board = first_board.copy()

        # 探索開始局面における最善手の局面を探索対象局面に含める
        bestmove_board = first_board.copy()
        bestmove_board.push_usi(book_tree[first_board_key].child_move[0])
        if bestmove_board.zobrist_hash() not in book_tree:
            bestmove_board_sfen_list.append(f"sfen {bestmove_board.sfen()}\n")

        count = 0
        print("Starting search...")
        pbar = tqdm.tqdm(desc="MCTS", dynamic_ncols=True)
        while count < playout_num:
            search(dl_data_tree[first_board_key])
            pbar.update(1)
            count += 1
        pbar.close()

    cnt = 0
    while len(visited_nodes) < 1000 * 10:
        cnt += 1
        if cnt % 2 == 0:
            turn = BLACK
        else:
            turn = WHITE
        first_board, first_board_key = select_root_board(turn=turn, eval_diff=args.eval_diff, book_moves_threshold=args.book_moves_threshold)
        first_board_sfen_list.append(f"{first_board.sfen()}\n")
        print(f"search board history: {' '.join([cshogi.move_to_usi(move) for move in first_board.history])}")

        dl_data_tree[first_board_key].board = first_board.copy()

        # 探索開始局面における最善手の局面を探索対象局面に含める
        bestmove_board = first_board.copy()
        bestmove_board.push_usi(book_tree[first_board_key].child_move[0])
        if bestmove_board.zobrist_hash() not in book_tree:
            bestmove_board_sfen_list.append(f"sfen {bestmove_board.sfen()}\n")

        count = 0
        print("Starting search...")
        pbar = tqdm.tqdm(desc="MCTS", dynamic_ncols=True)
        while count < playout_num:
            search(dl_data_tree[first_board_key])
            pbar.update(1)
            count += 1
        pbar.close()

    print(f"visited_nodes: {len(visited_nodes)}")
    move_count_list = [(dl_data_tree[key].move_count, key) for key in visited_nodes]
    move_count_list.sort(reverse=True)
    move_count_list = move_count_list[:min(len(move_count_list), 1000)]

    sfens_list = []
    moves_list = []
    for _, key in move_count_list:
        sfens_list.append(f"sfen {dl_data_tree[key].board.sfen()}\n")
        if args.boards and os.path.exists(args.boards):
            history = " ".join([cshogi.move_to_usi(move) for move in dl_data_tree[key].board.history])
            history_len = len(dl_data_tree[key].board.history)
            current_sfen = dl_data_tree[key].board.sfen()
            for i in range(history_len):
                dl_data_tree[key].board.pop()
            start_sfen = dl_data_tree[key].board.sfen()
            moves_list.append(f"{current_sfen}, startpos {start_sfen} moves {history}\n")

    sfens_list += bestmove_board_sfen_list
    with open(args.sfens, "w") as f:
        f.writelines(sfens_list)

    with open(args.first_board_sfen_output, "w") as f:
        f.writelines(first_board_sfen_list)

    if args.boards and os.path.exists(args.boards):
        with open(args.boards, "w") as f:
            f.writelines(moves_list)
