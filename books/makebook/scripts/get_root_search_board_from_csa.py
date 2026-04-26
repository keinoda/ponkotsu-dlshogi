import argparse
import cshogi
import glob
import os

from cshogi import CSA

try:
    import mcts_cpp
    HAS_MCTS_CPP = hasattr(mcts_cpp, "parse_book_cpp")
except Exception:
    mcts_cpp = None
    HAS_MCTS_CPP = False

class Node:
    def __init__(self):
        self.board = None
        self.child_move = []
        self.child_score = []
        self.child_depth = []

def rotate_board(board):
    return cshogi.Board(cshogi.rotate_sfen(board.sfen()))


def build_rotated_node(node):
    rotated_board = rotate_board(node.board)
    rotated_node = Node()
    rotated_node.board = rotated_board.copy()
    rotated_node.child_move = [
        cshogi.to_usi(cshogi.move_rotate(node.board.move_from_usi(move))).decode()
        for move in node.child_move
    ]
    rotated_node.child_score = [-score for score in node.child_score]
    rotated_node.child_depth = node.child_depth.copy()
    return rotated_node


def get_book_node(book_tree, board):
    board_key = board.zobrist_hash()
    node = book_tree.get(board_key)
    if node is not None:
        return board_key, node

    rotated_board = rotate_board(board)
    rotated_key = rotated_board.zobrist_hash()
    rotated_node = book_tree.get(rotated_key)
    if rotated_node is None:
        return board_key, None

    node = build_rotated_node(rotated_node)
    node.board = board.copy()
    book_tree[board_key] = node
    return board_key, node

def parse_book(book_path):
    if HAS_MCTS_CPP:
        book_tree, book_child_depth_tree = mcts_cpp.parse_book_cpp(book_path, Node)
        for key, node in book_tree.items():
            # parse_book_cpp returns board as sfen string in this code path.
            if isinstance(node.board, str):
                node.board = cshogi.Board(sfen=node.board)

            # Normalize score/depth to the same shape/type as the Python parser.
            node.child_score = [int(score) for score in node.child_score]
            depths = book_child_depth_tree.get(key, [])
            node.child_depth = [-1 if depth is None else int(depth) for depth in depths]
        return book_tree

    # parse book entry
    with open(book_path, "r") as f:
        books = f.readlines()
        books = [line.strip() for line in books[1:]]

    book_tree = dict()
    book_key = None
    board = cshogi.Board()
    for book in books:
        if book.startswith("sfen"):
            board.set_sfen(book[5:])
            book_key = board.zobrist_hash()
            book_tree[book_key] = Node()
            book_tree[book_key].board = board.copy()
        else:
            next_move_info = book.strip().split(" ")
            # 取るべき情報
            # 0: 指し手 (USI形式)
            # 2: スコア (int)
            # 3: 深さ (int)
            move_usi = next_move_info[0]
            score = int(next_move_info[2])
            depth = int(next_move_info[3])
            book_tree[book_key].child_move.append(move_usi)
            book_tree[book_key].child_score.append(score)
            book_tree[book_key].child_depth.append(depth)
    return book_tree

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("book_path", type=str)
    args.add_argument("csa_dir", type=str)
    args.add_argument("sfens_path", type=str)
    args.add_argument("--max-abs-score", type=int, default=500)
    args = args.parse_args()
    book_path = args.book_path

    csa_path_list = glob.glob(os.path.join(args.csa_dir, "*.csa"))
    root_search_sfens = []
    book_tree = parse_book(book_path)

    for csa_path in csa_path_list:
        parser = CSA.Parser()
        parser.parse_csa_file(csa_path)
        moves_usi = [cshogi.move_to_usi(move) for move in parser.moves]

        board = cshogi.Board(sfen=parser.sfen)
        board_key, node = get_book_node(book_tree, board)
        if node is None:
            continue

        score_now = node.child_score[0]
        depth_now = node.child_depth[0]
        move_now = node.child_move[0]

        for move in parser.moves:
            board.push(move)
            board_key, node = get_book_node(book_tree, board)
            score_now *= -1
            depth_now -= 1
            if node is not None:
                score_next = node.child_score[0]
                depth_next = node.child_depth[0]
                if score_now != score_next or (depth_now != depth_next and depth_next != 9999):
                    score_now = score_next
                    depth_now = depth_next
                    if abs(score_next) <= args.max_abs_score:
                        root_search_sfens.append(board.sfen())

    with open(args.sfens_path, "w") as f:
        f.write("\n".join(root_search_sfens))
