import argparse
import cshogi

class Node:
    __slots__ = ("child_move", "child_score", "child_depth")

    def __init__(self):
        self.child_move = []
        self.child_score = []
        self.child_depth = []


def normalize_sfen_key(sfen):
    # Ignore move number to match zobrist-hash-style position identity.
    return " ".join(sfen.split(" ")[:3])

def rotate_board(board):
    return cshogi.Board(cshogi.rotate_sfen(board.sfen()))


def build_rotated_node(board, node):
    rotated_board = rotate_board(board)
    rotated_node = Node()
    rotated_node.child_move = [
        cshogi.to_usi(cshogi.move_rotate(rotated_board.move_from_usi(move))).decode()
        for move in node.child_move
    ]
    rotated_node.child_score = [-score for score in node.child_score]
    rotated_node.child_depth = node.child_depth.copy()
    return rotated_node


def get_book_node(book_tree, board):
    board_key = normalize_sfen_key(board.sfen())
    node = book_tree.get(board_key)
    if node is not None:
        return node

    rotated_key = normalize_sfen_key(cshogi.rotate_sfen(board.sfen()))
    rotated_node = book_tree.get(rotated_key)
    if rotated_node is None:
        return None

    node = build_rotated_node(board, rotated_node)
    book_tree[board_key] = node
    return node

def parse_book(book_path, target_sfen_keys):
    # Parse only target positions to avoid building a full in-memory tree.
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
                if book_key in target_sfen_keys:
                    current_node = book_tree.get(book_key)
                    if current_node is None:
                        current_node = Node()
                        book_tree[book_key] = current_node
                else:
                    current_node = None
                continue

            if current_node is None:
                continue

            next_move_info = line.split()
            if len(next_move_info) < 4:
                continue

            # 取るべき情報
            # 0: 指し手 (USI形式)
            # 2: スコア (int)
            # 3: 深さ (int)
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
    args.add_argument("first_board_sfen", type=str)
    args.add_argument("sfens_path", type=str)
    args.add_argument('--book_moves_threshold', type=int, default=4)
    args = args.parse_args()
    book_path = args.book_path

    with open(args.first_board_sfen, "r") as f:
        first_board_sfen_list = [line.strip() for line in f if line.strip()]

    target_sfen_keys = set()
    for sfen in first_board_sfen_list:
        target_sfen_keys.add(normalize_sfen_key(sfen))
        target_sfen_keys.add(normalize_sfen_key(cshogi.rotate_sfen(sfen)))

    book_tree = parse_book(book_path, target_sfen_keys)


    sfens_list = []
    for sfen in first_board_sfen_list:
        board = cshogi.Board(sfen=sfen)
        node = get_book_node(book_tree, board)
        if node is not None and len(node.child_move) < args.book_moves_threshold:
            sfens_list.append(f"sfen {sfen}")

    with open(args.sfens_path, "w") as f:
        f.write("\n".join(sfens_list))
