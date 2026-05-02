import argparse
import cshogi
import glob
import os

from cshogi import CSA


class Node:
    __slots__ = ("board", "child_move", "child_score", "child_depth")

    def __init__(self):
        self.board = None
        self.child_move = []
        self.child_score = []
        self.child_depth = []


def normalize_sfen_key(sfen):
    return " ".join(sfen.split()[:3])


def rotate_sfen_text(sfen):
    rotated = cshogi.rotate_sfen(sfen)
    if isinstance(rotated, bytes):
        rotated = rotated.decode()
    return rotated


def build_rotated_node(node):
    rotated_node = Node()
    rotated_node.board = rotate_sfen_text(node.board)
    source_board = cshogi.Board(sfen=node.board)
    rotated_node.child_move = [
        cshogi.to_usi(cshogi.move_rotate(source_board.move_from_usi(move)))
        for move in node.child_move
    ]
    rotated_node.child_move = [
        move.decode() if isinstance(move, bytes) else move
        for move in rotated_node.child_move
    ]
    rotated_node.child_score = [-score for score in node.child_score]
    rotated_node.child_depth = node.child_depth.copy()
    return rotated_node


def get_book_node(book_tree, board):
    board_key = normalize_sfen_key(board.sfen())
    node = book_tree.get(board_key)
    if node is not None:
        return node

    rotated_key = normalize_sfen_key(rotate_sfen_text(board.sfen()))
    rotated_node = book_tree.get(rotated_key)
    if rotated_node is None:
        return None

    node = build_rotated_node(rotated_node)
    node.board = board.sfen()
    book_tree[board_key] = node
    return node


def collect_target_keys(csa_path_list):
    """Pass 1: Replay CSA games to collect all SFEN keys we need from the book."""
    target_keys = set()
    for csa_path in csa_path_list:
        parser = CSA.Parser()
        parser.parse_csa_file(csa_path)
        board = cshogi.Board(sfen=parser.sfen)
        key = normalize_sfen_key(board.sfen())
        target_keys.add(key)
        target_keys.add(normalize_sfen_key(rotate_sfen_text(board.sfen())))
        for move in parser.moves:
            board.push(move)
            key = normalize_sfen_key(board.sfen())
            target_keys.add(key)
            target_keys.add(normalize_sfen_key(rotate_sfen_text(board.sfen())))
    return target_keys


def parse_book(book_path, target_keys):
    """Pass 2: Stream the book file, only building Nodes for positions in target_keys."""
    book_tree = {}
    current_node = None

    with open(book_path, "r") as f:
        first_line = True
        for raw_line in f:
            if first_line:
                first_line = False
                continue

            line = raw_line.strip()
            if not line:
                continue

            if line.startswith("sfen "):
                book_key = normalize_sfen_key(line[5:])
                if book_key in target_keys:
                    current_node = book_tree.get(book_key)
                    if current_node is None:
                        current_node = Node()
                        current_node.board = line[5:].strip()
                        book_tree[book_key] = current_node
                else:
                    current_node = None
                continue

            if current_node is None:
                continue

            next_move_info = line.split()
            if len(next_move_info) < 4:
                continue
            move_usi = next_move_info[0]
            score = int(next_move_info[2])
            depth = int(next_move_info[3])
            current_node.child_move.append(move_usi)
            current_node.child_score.append(score)
            current_node.child_depth.append(depth)

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

    # Pass 1: Collect target SFEN keys from CSA games.
    target_keys = collect_target_keys(csa_path_list)
    print(f"target keys: {len(target_keys)}")

    # Pass 2: Parse only matching positions from the book.
    book_tree = parse_book(book_path, target_keys)
    print(f"matched book nodes: {len(book_tree)}")

    # Pass 3: Replay CSA games and find root search positions.
    root_search_sfens = []
    for csa_path in csa_path_list:
        parser = CSA.Parser()
        parser.parse_csa_file(csa_path)

        board = cshogi.Board(sfen=parser.sfen)
        node = get_book_node(book_tree, board)
        if node is None:
            continue

        score_now = node.child_score[0]
        depth_now = node.child_depth[0]

        for move in parser.moves:
            board.push(move)
            node = get_book_node(book_tree, board)
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
