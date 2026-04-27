#!/usr/bin/env python3
import argparse
import collections
import pathlib
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import cshogi
import requests
import yaml


CASTLE_YAML_URL = (
    "https://raw.githubusercontent.com/mizar/sylwi-kifu-vue/refs/heads/main/src/assets/castle/castle.yaml"
)
HISTORY_PREFIX = "search board history:"

PIECE_TYPE_TO_LETTER = {
    cshogi.PAWN: "P",
    cshogi.LANCE: "L",
    cshogi.KNIGHT: "N",
    cshogi.SILVER: "S",
    cshogi.GOLD: "G",
    cshogi.BISHOP: "B",
    cshogi.ROOK: "R",
    cshogi.KING: "K",
}
HAND_INDEX_TO_LETTER = {
    0: "P",
    1: "L",
    2: "N",
    3: "S",
    4: "G",
    5: "B",
    6: "R",
}


@dataclass
class Rule:
    rule_id: str
    name: str
    tesuu_max: int
    moves: List[str]
    pieces: List[str]
    hand: List[str]
    hand_exclude: List[str]
    capture: List[str]
    tags_required: List[str]
    tags_exclude: List[str]
    hide: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze 'search board history' logs with castle.yaml shape rules."
    )
    parser.add_argument(
        "--log-file",
        type=pathlib.Path,
        help="Path to a text log file. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--yaml-url",
        default=CASTLE_YAML_URL,
        help="URL for castle.yaml (default: sylwi-kifu-vue).",
    )
    parser.add_argument(
        "--show-hidden",
        action="store_true",
        help="Include hide: true entries in result breakdown.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Show top N shapes (default: 20).",
    )
    parser.add_argument(
        "--id-regex",
        default=r"^SHAPE_",
        help="Only count matched rule IDs that match this regex (default: ^SHAPE_).",
    )
    return parser.parse_args()


def load_yaml_rules(yaml_url: str) -> Tuple[List[Rule], Dict[str, Rule]]:
    response = requests.get(yaml_url, timeout=30)
    response.raise_for_status()
    raw = yaml.safe_load(response.text)

    if not isinstance(raw, list):
        raise ValueError("castle.yaml format is not a top-level list")

    rules: List[Rule] = []
    by_id: Dict[str, Rule] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        rule_id = item.get("id")
        if not rule_id:
            continue
        name = item.get("name", {}).get("ja_JP", rule_id)
        tesuu_raw = item.get("tesuu_max", 10**9)
        try:
            tesuu_max = int(float(tesuu_raw))
        except (TypeError, ValueError):
            tesuu_max = 10**9

        rule = Rule(
            rule_id=rule_id,
            name=name,
            tesuu_max=tesuu_max,
            moves=[str(v) for v in item.get("moves", [])],
            pieces=[str(v) for v in item.get("pieces", [])],
            hand=[str(v) for v in item.get("hand", [])],
            hand_exclude=[str(v) for v in item.get("hand_exclude", [])],
            capture=[str(v) for v in item.get("capture", [])],
            tags_required=[str(v) for v in item.get("tags_required", [])],
            tags_exclude=[str(v) for v in item.get("tags_exclude", [])],
            hide=bool(item.get("hide", False)),
        )
        rules.append(rule)
        by_id[rule.rule_id] = rule

    return rules, by_id


def extract_histories(text: str) -> List[List[str]]:
    histories: List[List[str]] = []
    for line in text.splitlines():
        if HISTORY_PREFIX not in line:
            continue
        _, rhs = line.split(HISTORY_PREFIX, 1)
        moves = [m for m in rhs.strip().split() if re.fullmatch(r"[1-9][a-i][1-9][a-i]\+?|[PLNSGBR]\*[1-9][a-i]", m)]
        if moves:
            histories.append(moves)
    return histories


def square_from_usi(square_usi: str) -> int:
    return cshogi.SQUARE_NAMES.index(square_usi)


def side_piece_symbol(piece_code: int) -> Optional[str]:
    if piece_code == cshogi.NONE:
        return None
    piece_type = cshogi.piece_to_piece_type(piece_code)
    base = PIECE_TYPE_TO_LETTER.get(piece_type)
    if base is None:
        return None
    return base if piece_code < cshogi.WPAWN else base.lower()


def parse_move_signature_spec(spec: str) -> Optional[Tuple[str, str, str]]:
    m = re.fullmatch(r"([KRGBSNLPkrgbsnlp])\*([1-9][a-i])([1-9][a-i])", spec)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def parse_piece_spec(spec: str) -> Optional[Tuple[str, str]]:
    m = re.fullmatch(r"([KRGBSNLPkrgbsnlp])\*([1-9][a-i])", spec)
    if not m:
        return None
    return m.group(1), m.group(2)


def hand_symbol_counts(board: cshogi.Board) -> Dict[str, int]:
    sente, gote = board.pieces_in_hand
    result: Dict[str, int] = {}
    for i, count in enumerate(sente):
        if count:
            result[HAND_INDEX_TO_LETTER[i]] = int(count)
    for i, count in enumerate(gote):
        if count:
            result[HAND_INDEX_TO_LETTER[i].lower()] = int(count)
    return result


def collect_game_state(moves_usi: List[str]) -> Tuple[List[Dict[str, object]], Optional[str]]:
    board = cshogi.Board()
    states: List[Dict[str, object]] = []
    move_signatures_seen: Set[str] = set()
    captures_seen: Set[str] = set()

    # ply 0
    states.append(
        {
            "ply": 0,
            "move_signatures": set(),
            "captures": set(),
            "board": board.copy(),
            "hand": hand_symbol_counts(board),
        }
    )

    for ply, usi in enumerate(moves_usi, start=1):
        if not re.fullmatch(r"[1-9][a-i][1-9][a-i]\+?|[PLNSGBR]\*[1-9][a-i]", usi):
            return states, f"invalid usi move: {usi}"

        move = board.move_from_usi(usi)
        if not board.is_legal(move):
            return states, f"illegal move at ply {ply}: {usi}"

        move_to = cshogi.move_to(move)
        captured_piece = board.piece(move_to)
        from_sq = cshogi.move_from(move)
        mover_symbol: Optional[str]
        if cshogi.move_is_drop(move):
            hand_piece = cshogi.move_drop_hand_piece(move)
            base = HAND_INDEX_TO_LETTER.get(hand_piece)
            mover_symbol = base if board.turn == cshogi.BLACK else base.lower()
            from_usi = "00"
        else:
            piece_code = board.piece(from_sq)
            mover_symbol = side_piece_symbol(piece_code)
            from_usi = cshogi.SQUARE_NAMES[from_sq]
        to_usi = cshogi.SQUARE_NAMES[move_to]

        if mover_symbol is not None and from_usi != "00":
            move_signatures_seen.add(f"{mover_symbol}*{from_usi}{to_usi}")

        cap_symbol = side_piece_symbol(captured_piece)
        if cap_symbol is not None:
            captures_seen.add(cap_symbol)

        board.push(move)
        states.append(
            {
                "ply": ply,
                "move_signatures": set(move_signatures_seen),
                "captures": set(captures_seen),
                "board": board.copy(),
                "hand": hand_symbol_counts(board),
            }
        )

    return states, None


def rule_base_match(rule: Rule, state: Dict[str, object]) -> bool:
    ply = int(state["ply"])
    if ply > rule.tesuu_max:
        return False

    move_signatures = state["move_signatures"]
    captures = state["captures"]
    board = state["board"]
    hand = state["hand"]

    for move_spec in rule.moves:
        if move_spec not in move_signatures:
            return False

    for cap in rule.capture:
        if cap not in captures:
            return False

    for piece_spec in rule.pieces:
        parsed = parse_piece_spec(piece_spec)
        if parsed is None:
            continue
        symbol, sq_usi = parsed
        sq = square_from_usi(sq_usi)
        cur = side_piece_symbol(board.piece(sq))
        if cur != symbol:
            return False

    for hs in rule.hand:
        if hand.get(hs, 0) <= 0:
            return False

    for hs in rule.hand_exclude:
        if hand.get(hs, 0) > 0:
            return False

    return True


def classify_history(rules: List[Rule], moves_usi: List[str]) -> Tuple[Set[str], Optional[str]]:
    states, error = collect_game_state(moves_usi)
    if error:
        return set(), error

    matched: Set[str] = set()
    for state in states:
        changed = True
        while changed:
            changed = False
            for rule in rules:
                if rule.rule_id in matched:
                    continue
                if not rule_base_match(rule, state):
                    continue
                if not set(rule.tags_required).issubset(matched):
                    continue
                if set(rule.tags_exclude).intersection(matched):
                    continue
                matched.add(rule.rule_id)
                changed = True

    return matched, None


def read_input_text(log_file: Optional[pathlib.Path]) -> str:
    if log_file is not None:
        return log_file.read_text(encoding="utf-8")
    return sys.stdin.read()


def main() -> int:
    args = parse_args()
    id_pattern = re.compile(args.id_regex)

    rules, by_id = load_yaml_rules(args.yaml_url)
    text = read_input_text(args.log_file)
    histories = extract_histories(text)

    if not histories:
        print("No 'search board history:' lines found.", file=sys.stderr)
        return 1

    counter: collections.Counter = collections.Counter()
    errors: List[str] = []
    for i, moves in enumerate(histories, start=1):
        matched, error = classify_history(rules, moves)
        if error:
            errors.append(f"history#{i}: {error}")
            continue

        for rid in matched:
            rule = by_id.get(rid)
            if rule is None:
                continue
            if not id_pattern.search(rule.rule_id):
                continue
            if rule.hide and not args.show_hidden:
                continue
            counter[rid] += 1

    print(f"histories_total: {len(histories)}")
    print(f"histories_error: {len(errors)}")
    print("shape_counts:")
    for rid, count in counter.most_common(args.top):
        rule = by_id[rid]
        print(f"  {rid}\t{rule.name}\t{count}")

    if errors:
        print("errors:", file=sys.stderr)
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
