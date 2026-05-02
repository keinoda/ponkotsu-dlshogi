import argparse
import os
import tempfile
from pathlib import Path

import cshogi


def split_sfen_parts(sfen_line):
    sfen = sfen_line[5:].strip() if sfen_line.startswith("sfen ") else sfen_line.strip()
    parts = sfen.split(None, 3)
    position_key = " ".join(parts[:3])
    ply = parts[3] if len(parts) >= 4 else "?"
    return position_key, ply, sfen


def normalize_move_line_for_viewer(line):
    # ShogiHome viewer is sensitive to non-standard 5-field move lines.
    # Convert: "move none score depth count" -> "move none score count"
    fields = line.strip().split()
    if len(fields) == 5:
        fields = [fields[0], fields[1], fields[2], fields[4]]
    return " ".join(fields) + "\n"


def rotated_position_key(sfen_line):
    _, _, sfen = split_sfen_parts(sfen_line)
    rotated_sfen = cshogi.rotate_sfen(sfen)
    if isinstance(rotated_sfen, bytes):
        rotated_sfen = rotated_sfen.decode()
    rotated_parts = rotated_sfen.split(None, 3)
    return " ".join(rotated_parts[:3])


def rotate_move_lines_for_target(source_sfen_line, source_moves):
    # Convert source moves into the rotated board orientation.
    _, _, source_sfen = split_sfen_parts(source_sfen_line)
    board = cshogi.Board(sfen=source_sfen)
    rotated_lines = []
    for line in source_moves:
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split()
        move = board.move_from_usi(fields[0])
        rotated_move = cshogi.to_usi(cshogi.move_rotate(move))
        if isinstance(rotated_move, bytes):
            rotated_move = rotated_move.decode()
        fields[0] = rotated_move
        rotated_lines.append(normalize_move_line_for_viewer(" ".join(fields)))
    return rotated_lines


def build_source_move_map(source_path):
    move_map = {}
    sfen_line_map = {}
    key_order = []
    duplicate_overwrites = 0
    key_to_ply = {}
    key_to_full_sfen = {}
    source_blocks = 0
    current_key = None
    current_moves = None

    def commit_block(key, moves):
        nonlocal duplicate_overwrites, source_blocks
        if key is None:
            return
        source_blocks += 1
        if key in move_map:
            duplicate_overwrites += 1
        else:
            key_order.append(key)
        move_map[key] = moves

    with source_path.open("r", encoding="utf-8", buffering=1024 * 1024) as src:
        for raw_line in src:
            stripped = raw_line.strip()
            if not stripped:
                continue

            if stripped.startswith("sfen "):
                commit_block(current_key, current_moves)

                key, ply, full_sfen = split_sfen_parts(stripped)
                current_key = key
                current_moves = []
                sfen_line_map[key] = "sfen " + full_sfen
                key_to_ply.setdefault(key, set()).add(ply)
                key_to_full_sfen.setdefault(key, set()).add(full_sfen)
                continue

            if current_moves is not None:
                current_moves.append(normalize_move_line_for_viewer(stripped))

    commit_block(current_key, current_moves)

    return (
        move_map,
        sfen_line_map,
        key_order,
        duplicate_overwrites,
        key_to_ply,
        key_to_full_sfen,
        source_blocks,
    )


def overwrite_petashock_book_streaming(
    input_path,
    output_path,
    source_move_map,
    source_sfen_line_map,
    source_key_order,
    source_key_to_ply,
):
    replaced = 0
    replaced_rotated = 0
    missing = 0
    replaced_with_different_ply = 0
    appended = 0
    target_blocks = 0
    matched_keys = set()
    rotated_move_cache = {}

    same_path = input_path.resolve() == output_path.resolve()

    fd, stage_name = tempfile.mkstemp(
        prefix=output_path.name + ".stage.",
        dir=str(output_path.parent),
        text=True,
    )
    os.close(fd)
    stage_path = Path(stage_name)

    if same_path:
        fd, final_name = tempfile.mkstemp(
            prefix=output_path.name + ".final.",
            dir=str(output_path.parent),
            text=True,
        )
        os.close(fd)
        final_path = Path(final_name)
    else:
        final_path = output_path

    replacing_current_block = False
    seen_first_sfen = False

    with input_path.open("r", encoding="utf-8", buffering=1024 * 1024) as src, stage_path.open(
        "w", encoding="utf-8", buffering=1024 * 1024
    ) as dst:
        for raw_line in src:
            stripped = raw_line.strip()

            if stripped.startswith("sfen "):
                seen_first_sfen = True
                target_blocks += 1

                key, ply, _ = split_sfen_parts(stripped)
                rkey = rotated_position_key(stripped)

                dst.write(stripped + "\n")

                source_key = None
                replacement = source_move_map.get(key)
                if replacement is not None:
                    source_key = key
                else:
                    source_key = rkey
                    replacement = source_move_map.get(source_key)
                    if replacement is not None:
                        rotated_cached = rotated_move_cache.get(source_key)
                        if rotated_cached is None:
                            rotated_cached = rotate_move_lines_for_target(
                                source_sfen_line_map[source_key],
                                replacement,
                            )
                            rotated_move_cache[source_key] = rotated_cached
                        replacement = rotated_cached

                if replacement is None:
                    missing += 1
                    replacing_current_block = False
                else:
                    matched_keys.add(source_key)
                    replaced += 1
                    if source_key == rkey and key != source_key:
                        replaced_rotated += 1
                    if ply not in source_key_to_ply.get(source_key, set()):
                        replaced_with_different_ply += 1
                    dst.writelines(replacement)
                    replacing_current_block = True
                continue

            # Header area before first SFEN.
            if not seen_first_sfen:
                dst.write(raw_line)
                continue

            if replacing_current_block:
                # Skip original move lines for replaced blocks.
                continue

            # Unreplaced block: preserve original line.
            dst.write(raw_line)

    # Build source-only blocks and sort by SFEN line for correct merge order.
    source_only_blocks = []
    for key in source_key_order:
        if key in matched_keys:
            continue
        source_only_blocks.append((source_sfen_line_map[key], source_move_map[key]))
    source_only_blocks.sort(key=lambda block: block[0])
    appended = len(source_only_blocks)

    # Merge stage book and source-only blocks while keeping global SFEN order.
    with stage_path.open("r", encoding="utf-8", buffering=1024 * 1024) as src, final_path.open(
        "w", encoding="utf-8", buffering=1024 * 1024
    ) as dst:
        pending_sfen = None
        pending_line = None

        # Preserve headers before the first SFEN line.
        for raw_line in src:
            if raw_line.startswith("sfen "):
                pending_sfen = raw_line.rstrip("\n")
                break
            dst.write(raw_line)

        source_idx = 0
        source_len = len(source_only_blocks)

        while pending_sfen is not None:
            move_lines = []

            # Read current block move lines until next SFEN or EOF.
            while True:
                pending_line = src.readline()
                if not pending_line:
                    next_sfen = None
                    break
                if pending_line.startswith("sfen "):
                    next_sfen = pending_line.rstrip("\n")
                    break
                move_lines.append(pending_line)

            # Output source-only blocks that should come before this target block.
            while source_idx < source_len and source_only_blocks[source_idx][0] < pending_sfen:
                dst.write(source_only_blocks[source_idx][0] + "\n")
                dst.writelines(source_only_blocks[source_idx][1])
                source_idx += 1

            # Output target block.
            dst.write(pending_sfen + "\n")
            dst.writelines(move_lines)

            pending_sfen = next_sfen

        # Output remaining source-only blocks.
        while source_idx < source_len:
            dst.write(source_only_blocks[source_idx][0] + "\n")
            dst.writelines(source_only_blocks[source_idx][1])
            source_idx += 1

    if same_path:
        final_path.replace(output_path)
    stage_path.unlink(missing_ok=True)
    if same_path:
        final_path = output_path

    return replaced, replaced_rotated, missing, replaced_with_different_ply, appended, target_blocks


def main():
    parser = argparse.ArgumentParser(
        description=(
            "SFEN一致局面の指し手を、dlshogiで作成した定跡の内容で上書きします。"
        )
    )
    parser.add_argument("petashock_book", help="上書き対象の定跡ファイル")
    parser.add_argument("dlshogi_book", help="上書き元(dlshogi)の定跡ファイル")
    parser.add_argument("output_book", help="出力先の定跡ファイル")
    args = parser.parse_args()

    petashock_path = Path(args.petashock_book)
    dlshogi_path = Path(args.dlshogi_book)
    output_path = Path(args.output_book)

    if not petashock_path.exists():
        raise FileNotFoundError(f"petashock book not found: {petashock_path}")
    if not dlshogi_path.exists():
        raise FileNotFoundError(f"dlshogi book not found: {dlshogi_path}")

    (
        source_move_map,
        source_sfen_line_map,
        source_key_order,
        duplicate_overwrites,
        source_key_to_ply,
        source_key_to_full_sfen,
        source_blocks,
    ) = build_source_move_map(dlshogi_path)

    (
        replaced,
        replaced_rotated,
        missing,
        replaced_with_different_ply,
        appended,
        target_blocks,
    ) = overwrite_petashock_book_streaming(
        petashock_path,
        output_path,
        source_move_map,
        source_sfen_line_map,
        source_key_order,
        source_key_to_ply,
    )

    print(
        "done "
        f"replaced={replaced} "
        f"replaced_rotated={replaced_rotated} "
        f"appended={appended} "
        f"unmatched={missing} "
        f"target_blocks={target_blocks} "
        f"source_blocks={source_blocks} "
        f"source_duplicates_overwritten={duplicate_overwrites} "
        "match_mode=position+rotated"
    )

    source_multi_ply_positions = sum(1 for plies in source_key_to_ply.values() if len(plies) > 1)
    source_multi_full_sfen_positions = sum(
        1 for sfens in source_key_to_full_sfen.values() if len(sfens) > 1
    )
    print(
        "position_mode_check "
        "ply_ignored=yes "
        f"source_multi_ply_positions={source_multi_ply_positions} "
        "target_multi_ply_positions=skipped_for_memory "
        f"source_multi_full_sfen_positions={source_multi_full_sfen_positions} "
        f"replaced_with_different_ply={replaced_with_different_ply}"
    )


if __name__ == "__main__":
    main()
