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
        _, node = get_book_node(book_tree, board)
        if node is not None and len(node.child_move) < args.book_moves_threshold:
            sfens_list.append(f"sfen {sfen}\n")

    with open(args.sfens_path, "w") as f:
        f.write("\n".join(sfens_list))
