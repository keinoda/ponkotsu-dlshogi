import argparse
import cshogi
import glob
import os

from cshogi import CSA

class Node:
    def __init__(self):
        self.board = None
        self.child_move = []
        self.child_score = []
        self.child_depth = []

def rotate_board(board):
    return cshogi.Board(cshogi.rotate_sfen(board.sfen()))

def parse_book(book_path):
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

    # 反転が含まれていなければ追加する
    book_tree_rotated = dict()
    for key in book_tree.keys():
        board = book_tree[key].board
        board_rotated = rotate_board(board).copy()
        key_rotated = board_rotated.zobrist_hash()
        if key_rotated not in book_tree:
            book_tree_rotated[key_rotated] = Node()
            book_tree_rotated[key_rotated].board = board_rotated
            book_tree_rotated[key_rotated].child_move = [cshogi.to_usi(cshogi.move_rotate(board.move_from_usi(move))).decode() \
                                                        for move in book_tree[key].child_move]
            book_tree_rotated[key_rotated].chils_score = [-score for score in book_tree[key].child_score]
            book_tree_rotated[key_rotated].child_depth = book_tree[key].child_depth.copy

    book_tree.update(book_tree_rotated)
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

    for csa_path in csa_path_list:
        book_tree = parse_book(book_path)

        parser = CSA.Parser()
        parser.parse_csa_file(csa_path)
        moves_usi = [cshogi.move_to_usi(move) for move in parser.moves]

        board = cshogi.Board(sfen=parser.sfen)
        board_key = board.zobrist_hash()
        score_now = book_tree[board_key].child_score[0]
        depth_now = book_tree[board_key].child_depth[0]
        move_now = book_tree[board_key].child_move[0]

        for move in parser.moves:
            board.push(move)
            board_key = board.zobrist_hash()
            score_now *= -1
            depth_now -= 1
            if board_key in book_tree:
                score_next = book_tree[board_key].child_score[0]
                depth_next = book_tree[board_key].child_depth[0]
                if score_now != score_next or depth_now != depth_next:
                    score_now = score_next
                    depth_now = depth_next
                    if abs(score_next) <= args.max_abs_score:
                        root_search_sfens.append(board.sfen())

    with open(args.sfens_path, "w") as f:
        f.write("\n".join(root_search_sfens))
