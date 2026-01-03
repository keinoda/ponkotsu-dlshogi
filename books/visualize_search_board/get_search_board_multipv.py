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
    args.add_argument("first_board_sfen", type=str)
    args.add_argument("sfens_path", type=str)
    args.add_argument('--book_moves_threshold', type=int, default=4)
    args = args.parse_args()
    book_path = args.book_path

    book_tree = parse_book(book_path)

    with open(args.first_board_sfen, "r") as f:
        first_board_sfen_list = [line.strip() for line in f.readlines()]


    sfens_list = []
    for sfen in first_board_sfen_list:
        board = cshogi.Board(sfen=sfen)
        board_key = board.zobrist_hash()
        if len(book_tree[board_key].child_move) < args.book_moves_threshold:
            sfens_list.append(f"sfen {sfen}\n")

    with open(args.sfens_path, "w") as f:
        f.write("\n".join(sfens_list))
