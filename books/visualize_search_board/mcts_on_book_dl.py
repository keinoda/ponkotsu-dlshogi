import argparse
import cshogi
from cshogi import NOT_REPETITION, REPETITION_DRAW, REPETITION_WIN, REPETITION_SUPERIOR
import numpy as np
import pickle
import tqdm

c_puct = 0.1

def score_to_value(score, a):
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

# DLで推論したツリー上でPV-MCTSを行う
# valueについては定跡ツリーに登録されていれば定跡ツリー上の値を優先する
def search(node):
    global depth0_count
    node.move_count += 1

    if not node.child_move:
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
            dl_data_tree[next_board_key].value = 1.0 - book_tree[node.board.zobrist_hash()].child_score[index]
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
    args.add_argument('--boards', type=str, default='test.pickle')
    args.add_argument('--book_moves_threshold', type=int, default=4)
    args = args.parse_args()

    with open(args.book, "r") as f:
        books = f.readlines()
        books = [s.replace("\n", "") for s in books[1:]]

    # 定跡をパースする
    board = cshogi.Board()
    for book in books:
        if book.startswith("sfen"):
            board.set_sfen(" ".join(book.split(" ")[1:]))
            board_key = board.zobrist_hash()
            book_tree[board_key] = Node()
            book_tree[board_key].board = board.copy()
        else:
            next_move_info = book.strip().split(" ")
            move_usi = next_move_info[0]
            book_tree[board_key].child_move.append(move_usi)
            book_tree[board_key].child_move_count.append(0)
            book_tree[board_key].child_score.append(int(next_move_info[2]))
            book_tree[board_key].child_score_sum.append(0)

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
            book_tree_rotated[rotated_board_key].child_move_count = [0 for i in range(len(book_tree[key].child_move))]
            book_tree_rotated[rotated_board_key].child_score = [-score for score in book_tree[key].child_score]
            book_tree_rotated[rotated_board_key].child_score_sum = [0 for i in range(len(book_tree[key].child_move))]

    book_tree.update(book_tree_rotated)

    a = 756.0864962951762
    for key in book_tree.keys():
        book_tree[key].child_move_count = np.array(book_tree[key].child_move_count)
        book_tree[key].child_score = np.array(book_tree[key].child_score, dtype=np.float32)
        book_tree[key].child_score_sum = np.array(book_tree[key].child_score_sum, dtype=np.float32)
        book_tree[key].child_score = score_to_value(book_tree[key].child_score, a)
        book_tree[key].child_policy = softmax_temperature_with_normalization(book_tree[key].child_score, 1.0)

    # DLで評価したノードを読み込む
    with open(args.dl_pickle, "rb") as f:
        dl_data = pickle.load(f)

    for key, node in dl_data.items():
        dl_data_tree[key] = Node()
        dl_data_tree[key].board = cshogi.Board(sfen=node.sfen)
        if node.legal_moves is not None:
            if len(node.legal_moves) > 0:
                dl_data_tree[key].child_move = node.legal_moves
                dl_data_tree[key].child_move_count = np.zeros(len(node.legal_moves), dtype=np.float32)
                # dl_data_tree[key].child_score = [None for i in range(len(node.legal_moves))]
                dl_data_tree[key].child_score_sum = np.zeros(len(node.legal_moves), dtype=np.float32)
                dl_data_tree[key].child_policy = softmax_temperature_with_normalization(node.policy_logits, 1.76)
        dl_data_tree[key].value = node.value

    # ルート局面から定跡ツリー上で最善手を辿り、登録されている候補手が閾値を初めて下回った局面をfirst_boardとする
    root_board = cshogi.Board()
    root_key = root_board.zobrist_hash()
    current_node = book_tree[root_key]
    current_key = root_key

    val_sum_threshold = 0.9
    while True:
        # policyの上位何手でval_sum_thresholdを超えるか確認する
        child_value_sorted = np.sort(dl_data_tree[current_key].child_policy)[::-1]
        val_sum_threshold_count = 0
        val_sum = 0.0
        while val_sum < val_sum_threshold and val_sum_threshold_count < len(child_value_sorted):
            val_sum += child_value_sorted[val_sum_threshold_count]
            val_sum_threshold_count += 1

        # val_sum_thresholdを超える手が閾値未満なら手を進める
        if val_sum_threshold_count >= args.book_moves_threshold and len(current_node.child_move) < args.book_moves_threshold or current_key not in book_tree:
            print(len(current_node.child_move))
            break
        best_child_index = np.argmax(current_node.child_score)
        best_move = current_node.child_move[best_child_index]
        next_board = current_node.board.copy()
        next_board.push_usi(best_move)
        next_board_key = next_board.zobrist_hash()
        current_node = book_tree[next_board_key]
        current_key = next_board_key

    first_board = book_tree[next_board_key].board
    first_board_key = current_key

    count = 0
    print("Starting search...")
    pbar = tqdm.tqdm(desc="MCTS", dynamic_ncols=True)
    while count < 200000:
        search(dl_data_tree[first_board_key])
        pbar.update(1)
        count += 1
    pbar.close()
    move_count_list = [(node.move_count, key) for node, key in zip(dl_data_tree.values(), dl_data_tree.keys()) if not node.child_move and node.move_count > 0]
    move_count_list.sort(reverse=True)
    move_count_list = move_count_list[:min(len(move_count_list), 1000)]

    moves_list = []
    sfens_list = []
    for _, key in move_count_list:
        sfens_list.append(f"sfen {dl_data_tree[key].board.sfen()}\n")
        moves_list.append(dl_data_tree[key].board.history)

    with open(args.sfens, "w") as f:
        f.writelines(sfens_list)

    with open(args.boards, "wb") as f:
        pickle.dump(moves_list, f)
