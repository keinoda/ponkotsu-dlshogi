import argparse
import importlib.util
from pathlib import Path
import sys

import cshogi
import numpy as np


def load_module():
    module_path = Path(__file__).resolve().parent / "mcts_on_book_dl.py"
    spec = importlib.util.spec_from_file_location("mcts_on_book_dl", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBoard:
    def __init__(self, legal_moves, draw_by_move):
        self.legal_moves = list(legal_moves)
        self._draw_by_move = dict(draw_by_move)
        self._last_move = None

    def copy(self):
        copied = FakeBoard(self.legal_moves, self._draw_by_move)
        copied._last_move = self._last_move
        return copied

    def push_usi(self, move):
        self._last_move = move

    def is_draw(self):
        return self._draw_by_move.get(self._last_move, 0)


class FakeNode:
    def __init__(self, board, child_move, child_score):
        self.board = board
        self.child_move = child_move
        self.child_score = child_score


def parse_first_int(tokens):
    for token in tokens:
        try:
            return int(token)
        except ValueError:
            continue
    return None


def rotate(board):
    return cshogi.Board(cshogi.rotate_sfen(board.sfen()))


def load_book_tree(book_path):
    book_tree = {}
    book_child_depth_tree = {}

    board = cshogi.Board()
    board_key = None

    with open(book_path, "r") as f:
        _header = f.readline()
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line:
                continue

            if line.startswith("sfen"):
                if board_key is not None:
                    book_tree[board_key].child_score = np.array(book_tree[board_key].child_score, dtype=np.float32)

                board.set_sfen(" ".join(line.split(" ")[1:]))
                board_key = board.zobrist_hash()
                node = FakeNode(board=board.copy(), child_move=[], child_score=[])
                book_tree[board_key] = node
                book_child_depth_tree[board_key] = []
                continue

            cols = line.strip().split(" ")
            move_usi = cols[0]
            book_tree[board_key].child_move.append(move_usi)
            book_tree[board_key].child_score.append(int(cols[2]))
            book_child_depth_tree[board_key].append(parse_first_int(cols[3:]))

    if board_key is not None and not isinstance(book_tree[board_key].child_score, np.ndarray):
        book_tree[board_key].child_score = np.array(book_tree[board_key].child_score, dtype=np.float32)

    book_tree_rotated = {}
    book_child_depth_tree_rotated = {}
    for key in list(book_tree.keys()):
        board = book_tree[key].board
        rotated_board = rotate(board)
        rotated_board_key = rotated_board.zobrist_hash()
        if rotated_board_key not in book_tree:
            rotated_moves = [
                cshogi.to_usi(cshogi.move_rotate(board.move_from_usi(move))).decode()
                for move in book_tree[key].child_move
            ]
            rotated_node = FakeNode(
                board=rotated_board.copy(),
                child_move=rotated_moves,
                child_score=-book_tree[key].child_score,
            )
            book_tree_rotated[rotated_board_key] = rotated_node
            book_child_depth_tree_rotated[rotated_board_key] = list(book_child_depth_tree.get(key, []))

    book_tree.update(book_tree_rotated)
    book_child_depth_tree.update(book_child_depth_tree_rotated)
    return book_tree, book_child_depth_tree


def build_path_entries_to_first_repetition(book_tree, start_sfen, max_steps):
    board = cshogi.Board(sfen=start_sfen)
    path_entries = []
    traced_moves = []
    seen_keys = {board.zobrist_hash()}

    for step in range(max_steps):
        current_key = board.zobrist_hash()
        if current_key not in book_tree:
            return path_entries, None, board, f"book_missing_at_step_{step}", traced_moves

        current_node = book_tree[current_key]
        current_node.board = board.copy()
        path_entries.append({"key": current_key, "node": current_node, "board_snapshot": current_node.board.copy()})

        if not current_node.child_move:
            return path_entries, None, board, f"no_child_move_at_step_{step}", traced_moves

        best_move = current_node.child_move[0]
        legal_usi = {cshogi.move_to_usi(move) for move in board.legal_moves}
        if best_move not in legal_usi:
            return path_entries, None, board, f"illegal_best_move_at_step_{step}:{best_move}", traced_moves

        traced_moves.append(best_move)
        next_board = board.copy()
        next_board.push_usi(best_move)

        next_key = next_board.zobrist_hash()

        # 実戦的にはREPETITION_DRAWにならないまま同一局面を巡回することがあるため、
        # zobrist再訪を循環検出として扱う。
        if next_key in seen_keys:
            return path_entries, next_key, board, "cycle_detected", traced_moves

        seen_keys.add(next_key)

        board = next_board

    return path_entries, None, board, "max_steps_reached", traced_moves


def main():
    parser = argparse.ArgumentParser(
        description="Trace best line and validate repetition detour behavior without assertions"
    )
    parser.add_argument("--book", required=True, help="Path to petashock book file")
    parser.add_argument(
        "--sfen",
        required=True,
        nargs="+",
        help="Initial SFEN to start tracing best line (quoted or unquoted)",
    )
    parser.add_argument(
        "--expected-move",
        default=None,
        help="Optional expected detour move for pass/fail reporting",
    )
    parser.add_argument("--max-steps", type=int, default=256, help="Maximum best-line steps to trace")
    args = parser.parse_args()

    mod = load_module()
    book_tree, book_child_depth_tree = load_book_tree(args.book)
    mod.book_tree = book_tree
    mod.book_child_depth_tree = book_child_depth_tree

    start_sfen = " ".join(args.sfen)
    path_entries, repeated_key, board_before_repetition, reason, traced_moves = build_path_entries_to_first_repetition(
        book_tree=book_tree,
        start_sfen=start_sfen,
        max_steps=args.max_steps,
    )

    if repeated_key is None:
        print("RESULT: NG (no repetition detected on traced best line)")
        return 1

    detour = mod.pick_repetition_detour(path_entries, repeated_key)
    if detour is None:
        print("RESULT: NG (detour not found)")
        return 1

    detour_entry, detour_child_index, detour_board = detour
    detour_move = detour_entry["node"].child_move[detour_child_index]

    detour_key = detour_entry["key"]
    detour_path_index = None
    for idx, entry in enumerate(path_entries):
        if entry["key"] != detour_key:
            continue
        legal = {cshogi.move_to_usi(m) for m in entry["node"].board.legal_moves}
        if detour_move in legal:
            detour_path_index = idx
            break

    detour_depth = None
    if detour_key in mod.book_child_depth_tree:
        depths = mod.book_child_depth_tree[detour_key]
        if detour_child_index < len(depths):
            detour_depth = depths[detour_child_index]

    if args.expected_move is not None:
        move_match = detour_move == args.expected_move
        if not move_match:
            print(f"RESULT: NG (selected move differs from expected move: got {detour_move}, expected {args.expected_move})")
            return 1

    if detour_depth == 9999:
        print("RESULT: NG (selected move is depth=9999)")
        return 1

    detour_prefix = traced_moves[:detour_path_index] if detour_path_index is not None else traced_moves[:-1]
    final_sequence = " ".join(detour_prefix + [detour_move])
    print(f"千日手回避手順: {final_sequence}")
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
