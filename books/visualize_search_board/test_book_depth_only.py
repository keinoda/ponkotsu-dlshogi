import argparse
from pathlib import Path

TARGET_SFEN_PREFIX = (
    "sfen ln3k1nl/7g1/pr1p1g2p/7p1/1Pb1p1S2/4P4/PS1P4P/2K1G1R2/LN1G3NL b S2Pbs6p"
)


def parse_first_int(tokens):
    for token in tokens:
        try:
            return int(token)
        except ValueError:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Validate book depth parsing logic using only a book file"
    )
    parser.add_argument("book", type=Path, help="Path to petashock book file")
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Scan all entries and assert move/depth append consistency",
    )
    args = parser.parse_args()

    if not args.book.exists():
        raise FileNotFoundError(f"Book file not found: {args.book}")

    sfen_count = 0
    move_line_count = 0
    malformed_move_line_count = 0
    score_parse_error_count = 0
    depth_none_count = 0

    block_move_count = 0
    block_depth_count = 0
    in_block = False
    mismatch_block_count = 0

    target_found = False
    target_moves = []
    in_target_block = False

    with args.book.open("r", encoding="utf-8", errors="replace", newline="") as f:
        _header = f.readline()
        for line_no, raw in enumerate(f, start=2):
            line = raw.rstrip("\r\n")
            if not line:
                continue

            if line.startswith("sfen "):
                if args.check_all and in_block and block_move_count != block_depth_count:
                    mismatch_block_count += 1

                sfen_count += 1
                in_block = True
                block_move_count = 0
                block_depth_count = 0

                in_target_block = line.startswith(TARGET_SFEN_PREFIX)
                if in_target_block:
                    target_found = True
                continue

            parts = line.split(" ")
            if len(parts) < 3:
                malformed_move_line_count += 1
                continue

            move_line_count += 1
            _move = parts[0]
            try:
                int(parts[2])
            except ValueError:
                score_parse_error_count += 1

            move_depth = parse_first_int(parts[3:])
            if move_depth is None:
                depth_none_count += 1

            if args.check_all:
                block_move_count += 1
                block_depth_count += 1

            if in_target_block:
                target_moves.append((parts[0], move_depth))

        if args.check_all and in_block and block_move_count != block_depth_count:
            mismatch_block_count += 1

    assert target_found, "Target SFEN block was not found in the book"
    assert len(target_moves) > 0, "Target SFEN block has no move lines"

    target_depth = {move: depth for move, depth in target_moves}
    assert target_depth.get("8g7f") == 9999, "8g7f depth is not 9999"
    assert target_depth.get("P*4d") == 9999, "P*4d depth is not 9999"

    if args.check_all:
        assert malformed_move_line_count == 0, (
            f"Malformed move lines detected: {malformed_move_line_count}"
        )
        assert score_parse_error_count == 0, (
            f"Score parse errors detected: {score_parse_error_count}"
        )
        assert mismatch_block_count == 0, (
            f"Move/depth count mismatched blocks: {mismatch_block_count}"
        )

    print("OK: target depth checks passed")
    print(f"sfen_count={sfen_count} move_line_count={move_line_count}")
    print(
        f"target_moves={target_moves[:5]}"
    )
    if args.check_all:
        print(
            "OK: full consistency checks passed "
            f"(malformed={malformed_move_line_count}, score_parse_err={score_parse_error_count}, "
            f"depth_none={depth_none_count}, mismatch_blocks={mismatch_block_count})"
        )


if __name__ == "__main__":
    main()
