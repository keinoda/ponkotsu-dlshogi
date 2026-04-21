import argparse
import pickle

import cshogi


def to_board(value):
    if isinstance(value, cshogi.Board):
        return value.copy()
    if isinstance(value, str):
        return cshogi.Board(sfen=value)
    return None


def make_canonical_key(board):
    key = board.zobrist_hash()
    rotated = cshogi.Board(cshogi.rotate_sfen(board.sfen()))
    rotated_key = rotated.zobrist_hash()
    return min(key, rotated_key)


def filter_rotated_entries(data):
    filtered = {}
    seen = set()
    removed = 0
    skipped = 0

    for key, node in data.items():
        board = to_board(getattr(node, "board", None))
        if board is None:
            # Keep entries we cannot interpret as board states.
            filtered[key] = node
            skipped += 1
            continue

        canonical = make_canonical_key(board)
        if canonical in seen:
            removed += 1
            continue

        seen.add(canonical)
        filtered[key] = node

    return filtered, removed, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Remove rotated-board duplicate entries from eval_sfens pickle output",
    )
    parser.add_argument("input_pickle", type=str)
    parser.add_argument("output_pickle", type=str)
    args = parser.parse_args()

    with open(args.input_pickle, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError("Expected pickle content to be dict")

    filtered, removed, skipped = filter_rotated_entries(data)

    with open(args.output_pickle, "wb") as f:
        pickle.dump(filtered, f, protocol=5)

    print(f"input_entries={len(data)}")
    print(f"output_entries={len(filtered)}")
    print(f"removed_rotated={removed}")
    print(f"skipped_unreadable={skipped}")


if __name__ == "__main__":
    main()
