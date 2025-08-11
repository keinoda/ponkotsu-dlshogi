import argparse
import cshogi
import numpy as np
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
    ucb = q + c_puct * node.child_p * u
    return np.argmax(ucb)

def backup(node):
    node.child_q = node.child_score_sum / node.child_move_count

book_tree = dict()
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
        self.child_p = None

def search(node):
    global depth0_count
    node.move_count += 1
    if not node.child_move:
        return node.value

    search_node = select_max_ucb_child(node)
    node.child_move_count[search_node] += 1

    next_board = node.board.copy()
    next_board.push_usi(node.child_move[search_node])
    next_board_key = next_board.zobrist_hash()
    if next_board_key not in book_tree:
        book_tree[next_board_key] = Node()
        book_tree[next_board_key].board = next_board.copy()
        book_tree[next_board_key].value = 1.0 - node.child_score[search_node]
        depth0_count += 1
    next_node = book_tree[next_board.zobrist_hash()]
    value = search(next_node)
    value = 1.0 - value

    node.sum_value += value
    node.child_score_sum[search_node] += value
    return value

def rotate(board):
    return cshogi.Board(cshogi.rotate_sfen(board.sfen()))

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('book')
    args.add_argument('sfens')
    # args.add_argument('boards')
    # args.add_argument('num')
    args = args.parse_args()

    book_path = args.book
    # sfens_path = args.sfens

    with open(book_path, "r") as f:
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
        book_tree[key].child_p = softmax_temperature_with_normalization(book_tree[key].child_score, 1.0)

    first_board = cshogi.Board()
    first_board_key = first_board.zobrist_hash()

    count = 1
    print("Starting search...")
    pbar = tqdm.tqdm(desc="MCTS", dynamic_ncols=True)
    while count < 200000:
        search(book_tree[first_board_key])
        pbar.update(1)
        count += 1
    pbar.close()
    move_count_list = [(node.move_count, key) for node, key in zip(book_tree.values(), book_tree.keys()) if not node.child_move]
    move_count_list.sort(reverse=True)
    move_count_list = move_count_list[:1000]
    sfens_list = [f"sfen {book_tree[key].board.sfen()}\n" for _, key in move_count_list]
    with open(args.sfens, "w") as f:
        f.writelines(sfens_list)
