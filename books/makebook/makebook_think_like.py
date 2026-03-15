#!/usr/bin/env python3
"""
EVAL_LEARNを使わずに、makebook thinkに近い処理を行う補助ツール。

このツールはUSIエンジンを使って候補手を探索し、
最終的に engine 側の `makebook from_sfen` を呼び出してbookファイルを生成する。

コマンド形式は makebook think と同系にしている:

    python3 tools/makebook_think_like.py \
        --engine ./YaneuraOu-by-gcc \
        think <sfen_file> <book_name> \
            moves 16 depth 18 nodes 0 startmoves 1 cluster 1 1 book_save_interval 900 multipv 4

    python3 tools/makebook_think_like.py \
        --engine ./YaneuraOu-by-gcc \
        think bw <sfen_file_black> <sfen_file_white> <book_name> moves 16 depth 18 multipv 4

備考:
- book_save_interval は互換のため受理するが、このツールでは未使用。
- 指定スレッド数分の局面を同時並行で探索するため、内部でエンジンプロセスを複数起動する。
"""

from __future__ import annotations

import argparse
import collections
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait


BESTMOVE_RE = re.compile(r"^bestmove\s+(\S+)")
MULTIPV_RE = re.compile(r"\bmultipv\s+(\d+)\b")
PV_RE = re.compile(r"\bpv\s+(.+)$")
DEPTH_RE = re.compile(r"\bdepth\s+(\d+)\b")


class EngineError(RuntimeError):
    pass


@dataclass
class SearchResult:
    bestmove: str
    multipv: Dict[int, List[str]]
    multipv_depth: Dict[int, int]
    last_depth: int


BOOK_MOVE_COUNT = 800


@dataclass
class WorkerOutput:
    lines: List[str]
    depth_map: Dict[Tuple[str, str], int]
    count_map: Dict[Tuple[str, str], int]


class USISession:
    def __init__(self, engine_path: str, cwd: Optional[str] = None) -> None:
        self.proc = subprocess.Popen(
            [engine_path],
            cwd=cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if not self.proc.stdin or not self.proc.stdout:
            raise EngineError("failed to open engine stdio")

        self._stdin = self.proc.stdin
        self._stdout = self.proc.stdout
        self._q: "queue.Queue[str]" = queue.Queue()
        self._recent: "collections.deque[str]" = collections.deque(maxlen=120)
        self._send_lock = threading.Lock()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

    def _reader_loop(self) -> None:
        try:
            for line in self._stdout:
                s = line.rstrip("\n")
                self._recent.append(s)
                self._q.put(s)
        finally:
            self._q.put("__EOF__")

    def send(self, cmd: str) -> None:
        if self.proc.poll() is not None:
            raise EngineError(f"engine exited unexpectedly with code {self.proc.returncode}")
        with self._send_lock:
            self._stdin.write(cmd + "\n")
            self._stdin.flush()

    def request_stop(self) -> None:
        # best-effort stop. Engine may already be exiting.
        if self.proc.poll() is not None:
            return
        try:
            self.send("stop")
        except Exception:
            pass

    def _read_line(self, timeout: float) -> str:
        try:
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            raise EngineError("timeout waiting for engine output")
        if line == "__EOF__":
            code = self.proc.poll()
            tail = "\n".join(self._recent)
            raise EngineError(f"engine output closed (exit={code})\nlast output:\n{tail}")
        return line

    def wait_usiok(self, timeout: float = 30.0) -> None:
        self.send("usi")
        t0 = time.time()
        while True:
            line = self._read_line(timeout=max(0.1, timeout - (time.time() - t0)))
            if line == "usiok":
                return

    def wait_readyok(self, timeout: float = 30.0) -> None:

        self.send("isready")
        t0 = time.time()
        while True:
            line = self._read_line(timeout=max(0.1, timeout - (time.time() - t0)))
            if line == "readyok":
                return

    def setoption(self, name: str, value: str) -> None:
        self.send(f"setoption name {name} value {value}")

    def search(self, position_spec: str, go_cmd: str, timeout: float) -> SearchResult:
        self.send(f"position {position_spec}")
        self.send(go_cmd)

        t0 = time.time()
        mps: Dict[int, List[str]] = {}
        mp_depth: Dict[int, int] = {}
        last_depth = 0
        bestmove = ""

        while True:
            remain = timeout - (time.time() - t0)
            if remain <= 0:
                raise EngineError(f"search timeout for position: {position_spec}")

            line = self._read_line(timeout=max(0.1, remain))

            bm = BESTMOVE_RE.match(line)
            if bm:
                bestmove = bm.group(1)
                break

            if not line.startswith("info"):
                continue

            mp_match = MULTIPV_RE.search(line)
            pv_match = PV_RE.search(line)
            depth_match = DEPTH_RE.search(line)
            if depth_match:
                try:
                    last_depth = int(depth_match.group(1))
                except ValueError:
                    pass
            if not mp_match or not pv_match:
                continue

            try:
                mp_idx = int(mp_match.group(1))
            except ValueError:
                continue

            pv_moves = [tok for tok in pv_match.group(1).split() if tok and tok != "(none)"]
            if pv_moves:
                mps[mp_idx] = pv_moves
                if depth_match:
                    try:
                        mp_depth[mp_idx] = int(depth_match.group(1))
                    except ValueError:
                        pass

        return SearchResult(bestmove=bestmove, multipv=mps, multipv_depth=mp_depth, last_depth=last_depth)

    def close(self) -> None:
        try:
            if self.proc.poll() is None:
                self.send("quit")
                self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def append_moves(position_spec: str, moves: List[str]) -> str:
    s = position_spec.strip()
    if not s:
        return s
    if " moves " in s:
        return s + " " + " ".join(moves)
    return s + " moves " + " ".join(moves)


def valid_move_token(token: str) -> bool:
    if token in ("resign", "win", "(none)", "none"):
        return False
    return True


def choose_root_candidates(result: SearchResult, multipv: int) -> List[Tuple[List[str], int]]:
    candidates: List[Tuple[List[str], int]] = []

    # Prefer explicit multipv lines from info output.
    if result.multipv:
        for i in range(1, multipv + 1):
            pv = result.multipv.get(i)
            if not pv:
                continue
            if valid_move_token(pv[0]):
                candidates.append((pv[:2], result.multipv_depth.get(i, result.last_depth)))

    # Fallback to bestmove only.
    if not candidates and valid_move_token(result.bestmove):
        candidates.append(([result.bestmove], result.last_depth))

    return candidates


@dataclass
class ThinkParams:
    from_bw: bool
    sfen_file_black: str
    sfen_file_white: Optional[str]
    book_name: str
    start_moves: int = 1
    moves: int = 16
    depth: int = 24
    depth_specified: bool = False
    nodes: int = 0
    cluster_id: int = 1
    cluster_num: int = 1
    book_save_interval: int = 15 * 60
    multipv: int = 1


@dataclass
class BookDepthStat:
    max_depth: int
    count_at_target_depth: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="makebook think-like helper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--engine", required=True, help="Path to engine binary")
    p.add_argument("--eval-dir", default=None, help="Value for USI option EvalDir")
    p.add_argument("--cwd", default=None, help="Engine working directory")
    p.add_argument("--threads", type=int, default=1, help="Parallel worker count")
    p.add_argument(
        "--usi-hash",
        type=int,
        default=None,
        help="USI_Hash per engine process (omit to use engine default)",
    )
    p.add_argument("--timeout", type=float, default=120.0, help="Timeout per search (sec)")
    p.add_argument("--quiet", action="store_true", help="Less progress output")
    p.add_argument("cmd", choices=["think"], help="Command mode")
    p.add_argument("think_args", nargs=argparse.REMAINDER, help="think command arguments")
    return p.parse_args()


def parse_think_args(tokens: Sequence[str]) -> ThinkParams:
    t = list(tokens)
    if not t:
        raise ValueError("think arguments are empty")

    i = 0
    from_bw = False
    sfen_b = ""
    sfen_w: Optional[str] = None

    if t[i] == "bw":
        from_bw = True
        if len(t) < 4:
            raise ValueError("think bw requires: bw <black_file> <white_file> <book_name>")
        sfen_b = t[i + 1]
        sfen_w = t[i + 2]
        book_name = t[i + 3]
        i += 4
    else:
        if len(t) < 2:
            raise ValueError("think requires: <sfen_file> <book_name>")
        sfen_b = t[i]
        book_name = t[i + 1]
        i += 2

    p = ThinkParams(from_bw=from_bw, sfen_file_black=sfen_b, sfen_file_white=sfen_w, book_name=book_name)

    while i < len(t):
        key = t[i]
        i += 1

        if key == "moves":
            p.moves = int(t[i]); i += 1
        elif key == "depth":
            p.depth = int(t[i]); i += 1
            p.depth_specified = True
        elif key == "nodes":
            p.nodes = int(t[i]); i += 1
        elif key == "startmoves":
            p.start_moves = int(t[i]); i += 1
        elif key == "cluster":
            p.cluster_id = int(t[i]); p.cluster_num = int(t[i + 1]); i += 2
        elif key == "book_save_interval":
            p.book_save_interval = int(t[i]); i += 1
        elif key == "multipv":
            p.multipv = int(t[i]); i += 1
        elif key == "":
            continue
        else:
            raise ValueError(f"illegal token: {key}")

    if p.cluster_num <= 0:
        raise ValueError("cluster_num must be >= 1")
    if not (1 <= p.cluster_id <= p.cluster_num):
        raise ValueError("cluster_id must satisfy 1 <= cluster_id <= cluster_num")
    if p.start_moves <= 0:
        raise ValueError("startmoves must be >= 1")
    if p.moves <= 0:
        raise ValueError("moves must be >= 1")
    if p.depth <= 0 and p.nodes <= 0:
        raise ValueError("either depth > 0 or nodes > 0 is required")
    if p.multipv <= 0:
        raise ValueError("multipv must be >= 1")
    return p


def build_go_cmd(depth: int, nodes: int, depth_override: Optional[int] = None) -> str:
    if depth_override is not None:
        return f"go depth {depth_override}"
    if nodes and nodes > 0:
        return f"go nodes {nodes}"
    return f"go depth {depth}"


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.strip().startswith("#")]


def split_position_and_moves(line: str) -> Tuple[str, List[str], str]:
    tokens = line.strip().split()
    if not tokens:
        raise ValueError("empty position line")

    if tokens[0] == "startpos":
        base = "startpos"
        stm0 = "b"
        moves: List[str] = []
        if len(tokens) >= 2 and tokens[1] == "moves":
            moves = tokens[2:]
        return base, moves, stm0

    if tokens[0] == "sfen":
        if len(tokens) < 5:
            raise ValueError(f"invalid sfen line: {line}")
        stm0 = tokens[2]
        base = " ".join(tokens[:5])
        moves: List[str] = []
        if len(tokens) > 5:
            if tokens[5] == "moves":
                moves = tokens[6:]
            else:
                moves = tokens[5:]
        return base, moves, stm0

    raise ValueError(f"unsupported position format: {line}")


def side_after_plies(stm0: str, ply: int) -> str:
    if stm0 not in ("b", "w"):
        return "?"
    if ply % 2 == 0:
        return stm0
    return "w" if stm0 == "b" else "b"


def filter_side_ok(side_filter: Optional[str], side: str) -> bool:
    if side_filter is None:
        return True
    return side_filter == side


def build_candidate_positions(lines: List[Tuple[str, Optional[str]]], p: ThinkParams) -> List[str]:
    positions: List[str] = []

    for line, side_filter in lines:
        base, move_list, stm0 = split_position_and_moves(line)

        for i in range(p.moves):
            if i < len(move_list):
                cur = base if i == 0 else (base + " moves " + " ".join(move_list[:i]))
                side = side_after_plies(stm0, i)
                if i >= p.start_moves - 1 and filter_side_ok(side_filter, side):
                    positions.append(cur)
                continue

            cur = base if i == 0 else (base + " moves " + " ".join(move_list[:i]))
            side = side_after_plies(stm0, i)
            if i >= p.start_moves - 1 and filter_side_ok(side_filter, side):
                positions.append(cur)
            break

    # 重複排除（順序保持）
    dedup: Dict[str, None] = {}
    for s in positions:
        dedup.setdefault(s, None)
    uniq = list(dedup.keys())

    # cluster間引き
    if p.cluster_id != 1 or p.cluster_num != 1:
        filtered: List[str] = []
        for i, s in enumerate(uniq):
            if (i % p.cluster_num) == (p.cluster_id - 1):
                filtered.append(s)
        uniq = filtered

    return uniq


def parse_book_move_depth(line: str) -> int:
    # book line format: <move> <ponder> [value] [depth] [count]
    tokens = line.strip().split()
    if len(tokens) >= 4:
        try:
            return int(tokens[3])
        except ValueError:
            return 0
    return 0


def parse_book_move_fields(line: str) -> Tuple[str, str, int, int, int]:
    tokens = line.strip().split()
    if len(tokens) < 2:
        raise ValueError(f"invalid book move line: {line}")

    move = tokens[0]
    ponder = tokens[1]

    value = 0
    depth = 0
    count = 1

    if len(tokens) >= 3:
        try:
            value = int(tokens[2])
        except ValueError:
            value = 0
    if len(tokens) >= 4:
        try:
            depth = int(tokens[3])
        except ValueError:
            depth = 0
    if len(tokens) >= 5:
        try:
            count = int(tokens[4])
        except ValueError:
            count = 1

    return move, ponder, value, depth, count


def python_sort_book_file(
    src: str,
    dst: str,
    depth_override: Optional[int] = None,
    depth_map: Optional[Dict[Tuple[str, str], int]] = None,
    count_map: Optional[Dict[Tuple[str, str], int]] = None,
) -> None:
    # makebook sort 相当: sfen順で並べ、各sfen配下の手を move_count降順→value降順で並べる。
    header_lines: List[str] = []
    blocks: Dict[str, List[Tuple[str, str, int, int, int]]] = {}
    current_sfen: Optional[str] = None

    with open(src, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                current_sfen = None
                continue
            if line.startswith("#") or line.startswith("//"):
                if current_sfen is None:
                    header_lines.append(line)
                continue
            if line.startswith("sfen "):
                current_sfen = line[5:].strip()
                blocks.setdefault(current_sfen, [])
                continue

            if current_sfen is None:
                continue

            e = parse_book_move_fields(line)
            m, p, v, d, c = e
            if depth_map is not None:
                d = depth_map.get((current_sfen, m), d)
            elif depth_override is not None:
                d = depth_override
            if count_map is not None:
                c = count_map.get((current_sfen, m), c)
            e = (m, p, v, d, c)
            blocks[current_sfen].append(e)

    with open(dst, "w", encoding="utf-8") as o:
        for h in header_lines:
            o.write(h + "\n")
        if header_lines:
            o.write("\n")

        for sfen in sorted(blocks.keys()):
            o.write(f"sfen {sfen}\n")
            moves = blocks[sfen]
            moves.sort(key=lambda x: (-x[4], -x[2]))
            for m, p, v, d, c in moves:
                o.write(f"{m} {p} {v} {d} {c}\n")
            o.write("\n")


def load_book_blocks(
    path: str,
    depth_override: Optional[int] = None,
    depth_map: Optional[Dict[Tuple[str, str], int]] = None,
    count_map: Optional[Dict[Tuple[str, str], int]] = None,
) -> Tuple[List[str], Dict[str, List[Tuple[str, str, int, int, int]]]]:
    header_lines: List[str] = []
    blocks: Dict[str, List[Tuple[str, str, int, int, int]]] = {}
    current_sfen: Optional[str] = None

    if not os.path.exists(path):
        return header_lines, blocks

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                current_sfen = None
                continue
            if line.startswith("#") or line.startswith("//"):
                if current_sfen is None:
                    header_lines.append(line)
                continue
            if line.startswith("sfen "):
                current_sfen = line[5:].strip()
                blocks.setdefault(current_sfen, [])
                continue

            if current_sfen is None:
                continue

            e = parse_book_move_fields(line)
            m, p, v, d, c = e
            if depth_map is not None:
                d = depth_map.get((current_sfen, m), d)
            elif depth_override is not None:
                d = depth_override
            if count_map is not None:
                c = count_map.get((current_sfen, m), c)
            e = (m, p, v, d, c)
            blocks[current_sfen].append(e)

    return header_lines, blocks


def merge_book_files(
    existing_path: str,
    new_path: str,
    merged_path: str,
    new_depth_override: Optional[int] = None,
    new_depth_map: Optional[Dict[Tuple[str, str], int]] = None,
    new_count_map: Optional[Dict[Tuple[str, str], int]] = None,
) -> None:
    header_old, old_blocks = load_book_blocks(existing_path)
    header_new, new_blocks = load_book_blocks(
        new_path,
        depth_override=new_depth_override,
        depth_map=new_depth_map,
        count_map=new_count_map,
    )

    headers = header_old if header_old else header_new

    merged: Dict[str, Dict[str, Tuple[str, str, int, int, int]]] = {}

    def put_entry(sfen: str, entry: Tuple[str, str, int, int, int], prefer_new: bool) -> None:
        move, ponder, value, depth, count = entry
        sfen_map = merged.setdefault(sfen, {})
        old = sfen_map.get(move)
        if old is None:
            sfen_map[move] = entry
            return

        _, _, old_value, old_depth, old_count = old

        merged_count = old_count + count

        # depthは深い方。value/ponderは深い方(同深さならvalue高い方)を採用しつつcountは加算。
        if depth > old_depth:
            sfen_map[move] = (move, ponder, value, depth, merged_count)
            return
        if depth < old_depth:
            sfen_map[move] = (old[0], old[1], old_value, old_depth, merged_count)
            return

        if value > old_value:
            sfen_map[move] = (move, ponder, value, depth, merged_count)
            return
        if value < old_value:
            sfen_map[move] = (old[0], old[1], old_value, old_depth, merged_count)
            return

        if prefer_new:
            sfen_map[move] = (move, ponder, value, depth, merged_count)
        else:
            sfen_map[move] = (old[0], old[1], old_value, old_depth, merged_count)

    for sfen, moves in old_blocks.items():
        for e in moves:
            put_entry(sfen, e, prefer_new=False)

    for sfen, moves in new_blocks.items():
        for e in moves:
            put_entry(sfen, e, prefer_new=True)

    with open(merged_path, "w", encoding="utf-8") as o:
        for h in headers:
            o.write(h + "\n")
        if headers:
            o.write("\n")

        for sfen in sorted(merged.keys()):
            o.write(f"sfen {sfen}\n")
            moves = list(merged[sfen].values())
            moves.sort(key=lambda x: (-x[4], -x[2]))
            for m, p, v, d, c in moves:
                o.write(f"{m} {p} {v} {d} {c}\n")
            o.write("\n")


def load_book_depth_stats(
    book_path: str,
    target_depth: int,
    interested_sfens: Optional[set[str]] = None,
) -> Tuple[Dict[str, BookDepthStat], int]:
    stats: Dict[str, BookDepthStat] = {}
    if not os.path.exists(book_path):
        return stats, 0

    current_sfen: Optional[str] = None
    current_max_depth = 0
    current_target_count = 0
    total_positions = 0

    def flush_current() -> None:
        nonlocal current_sfen, current_max_depth, current_target_count, total_positions
        if current_sfen is None:
            return
        total_positions += 1

        if interested_sfens is None or current_sfen in interested_sfens:
            stats[current_sfen] = BookDepthStat(
                max_depth=current_max_depth,
                count_at_target_depth=current_target_count,
            )

        current_sfen = None
        current_max_depth = 0
        current_target_count = 0

    with open(book_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                flush_current()
                continue
            if line.startswith("#") or line.startswith("//"):
                continue
            if line.startswith("sfen "):
                flush_current()
                current_sfen = line[5:].strip()
                current_max_depth = 0
                current_target_count = 0
                continue
            if current_sfen is None:
                continue

            d = parse_book_move_depth(line)
            if d > current_max_depth:
                current_max_depth = d
            if d == target_depth:
                current_target_count += 1

    flush_current()
    return stats, total_positions


def should_skip_by_existing_book(
    sfen: str,
    target_depth: int,
    multipv: int,
    stats: Dict[str, BookDepthStat],
) -> bool:
    st = stats.get(sfen)
    if st is None:
        return False

    # 既存のほうが深い探索結果なら再探索しない。
    if st.max_depth > target_depth:
        return True

    # 同一深さで候補手数がMultiPV以上なら再探索しない。
    if st.max_depth == target_depth:
        return st.count_at_target_depth >= multipv

    return False


def filter_positions_by_existing_book(
    positions: List[str],
    p: ThinkParams,
    stats: Dict[str, BookDepthStat],
) -> Tuple[List[str], int]:
    if not stats:
        return positions, 0

    kept: List[str] = []
    skipped = 0
    for s in positions:
        lookup_key = s[5:].strip() if s.startswith("sfen ") else s
        if should_skip_by_existing_book(lookup_key, p.depth, p.multipv, stats):
            skipped += 1
        else:
            kept.append(s)
    return kept, skipped


def build_input_pairs(p: ThinkParams) -> List[Tuple[str, Optional[str]]]:
    pairs: List[Tuple[str, Optional[str]]] = []
    if p.from_bw:
        assert p.sfen_file_white is not None
        for ln in read_lines(p.sfen_file_black):
            pairs.append((ln, "b"))
        if p.sfen_file_white != "no_file":
            for ln in read_lines(p.sfen_file_white):
                pairs.append((ln, "w"))
    else:
        for ln in read_lines(p.sfen_file_black):
            pairs.append((ln, None))
    return pairs


def run_makebook_from_sfen(
    engine_path: str,
    cwd: Optional[str],
    src_file: str,
    book_name: str,
    eval_dir: Optional[str],
    depth_override: Optional[int],
    depth_map: Optional[Dict[Tuple[str, str], int]],
    count_map: Optional[Dict[Tuple[str, str], int]],
) -> None:
    cmd = [engine_path]
    with NamedTemporaryFile("w", encoding="utf-8", suffix=".db", delete=False) as tf_db:
        unsorted_book = tf_db.name
    with NamedTemporaryFile("w", encoding="utf-8", suffix=".db", delete=False) as tf_merge:
        merged_book = tf_merge.name

    eval_dir_cmd = ""
    if eval_dir:
        eval_dir_cmd = f"setoption name EvalDir value {eval_dir}\n"

    usi_script = (
        "usi\n"
        f"{eval_dir_cmd}"
        "setoption name BookFile value no_book\n"
        "setoption name IgnoreBookPly value false\n"
        "isready\n"
        f"makebook from_sfen {src_file} {unsorted_book} moves 1\n"
        "quit\n"
    )
    cp = subprocess.run(
        cmd,
        cwd=cwd,
        input=usi_script,
        text=True,
        capture_output=True,
        check=False,
    )

    # from_sfen の出力ができていなければ失敗。
    if not (os.path.exists(unsorted_book) and os.path.getsize(unsorted_book) > 0):
        if cp.returncode != 0:
            raise EngineError(
                "makebook from_sfen failed\n"
                f"stdout:\n{cp.stdout[-4000:]}\n"
                f"stderr:\n{cp.stderr[-4000:]}"
            )
        raise EngineError(
            "makebook from_sfen did not produce output book\n"
            f"stdout:\n{cp.stdout[-4000:]}\n"
            f"stderr:\n{cp.stderr[-4000:]}"
        )

    # 出力先bookが既にある場合は、常に新規結果をマージする。
    if os.path.exists(book_name) and os.path.getsize(book_name) > 0:
        merge_book_files(
            book_name,
            unsorted_book,
            merged_book,
            new_depth_override=depth_override,
            new_depth_map=depth_map,
            new_count_map=count_map,
        )
        # sort相当処理。makebook sortがクラッシュするビルドがあるためPython側で同等ソートする。
        python_sort_book_file(merged_book, book_name)
    else:
        # sort相当処理。makebook sortがクラッシュするビルドがあるためPython側で同等ソートする。
        python_sort_book_file(
            unsorted_book,
            book_name,
            depth_override=depth_override,
            depth_map=depth_map,
            count_map=count_map,
        )

    if os.path.exists(book_name) and os.path.getsize(book_name) > 0:
        try:
            os.remove(unsorted_book)
        except OSError:
            pass
        try:
            os.remove(merged_book)
        except OSError:
            pass
        return

    if cp.returncode != 0:
        raise EngineError(
            "makebook from_sfen failed\n"
            f"stdout:\n{cp.stdout[-4000:]}\n"
            f"stderr:\n{cp.stderr[-4000:]}"
        )

    # returncodeは0だがbookができていないケースも失敗として扱う。
    raise EngineError(
        "sorted output book was not produced\n"
        f"stdout:\n{cp.stdout[-4000:]}\n"
        f"stderr:\n{cp.stderr[-4000:]}"
    )


def save_snapshot_book(
    engine_path: str,
    cwd: Optional[str],
    eval_dir: Optional[str],
    lines: List[str],
    book_name: str,
    depth_override: Optional[int],
    depth_map: Optional[Dict[Tuple[str, str], int]],
    count_map: Optional[Dict[Tuple[str, str], int]],
) -> None:
    with NamedTemporaryFile("w", encoding="utf-8", suffix=".sfen", delete=False) as tf:
        temp_sfen = tf.name
        for ln in lines:
            tf.write(ln + "\n")

    run_makebook_from_sfen(
        engine_path,
        cwd,
        temp_sfen,
        book_name,
        eval_dir,
        depth_override,
        depth_map,
        count_map,
    )


def make_periodic_snapshot_name(book_name: str, index: int) -> str:
    if book_name.endswith(".db"):
        return f"{book_name}-{index}.db"
    return f"{book_name}.db-{index}.db"


def worker_task(
    worker_id: int,
    engine_path: str,
    cwd: Optional[str],
    timeout: float,
    usi_hash: Optional[int],
    eval_dir: Optional[str],
    p: ThinkParams,
    positions: List[str],
    on_position_done: Optional[Callable[[int, str], None]] = None,
    stop_event: Optional[threading.Event] = None,
    on_session_start: Optional[Callable[[USISession], None]] = None,
    on_session_end: Optional[Callable[[USISession], None]] = None,
) -> WorkerOutput:
    sess = USISession(engine_path, cwd=cwd)
    if on_session_start is not None:
        on_session_start(sess)
    try:
        sess.wait_usiok()
        sess.setoption("USI_Ponder", "false")
        sess.setoption("Threads", "1")
        if usi_hash is not None:
            sess.setoption("USI_Hash", str(usi_hash))
        sess.setoption("MultiPV", str(p.multipv))
        if eval_dir:
            sess.setoption("EvalDir", eval_dir)
        sess.setoption("BookFile", "no_book")
        sess.setoption("IgnoreBookPly", "false")
        sess.wait_readyok(timeout=60)

        out_lines: List[str] = []
        out_depth_map: Dict[Tuple[str, str], int] = {}
        out_count_map: Dict[Tuple[str, str], int] = {}
        go_cmd = build_go_cmd(p.depth, p.nodes)

        for pos_spec in positions:
            if stop_event is not None and stop_event.is_set():
                break

            result = sess.search(pos_spec, go_cmd, timeout=timeout)

            if on_position_done is not None:
                on_position_done(worker_id, pos_spec)

            cands = choose_root_candidates(result, p.multipv)
            normalized_sfen = pos_spec[5:].strip() if pos_spec.startswith("sfen ") else pos_spec

            for pv_moves, pv_depth in cands:
                if not pv_moves:
                    continue
                if not valid_move_token(pv_moves[0]):
                    continue

                if p.depth_specified:
                    depth_for_book = p.depth
                else:
                    depth_for_book = pv_depth if pv_depth > 0 else result.last_depth

                m1 = pv_moves[0]
                m2: Optional[str] = None
                if len(pv_moves) >= 2 and valid_move_token(pv_moves[1]):
                    m2 = pv_moves[1]

                out_depth_map[(normalized_sfen, m1)] = max(1, depth_for_book)
                out_count_map[(normalized_sfen, m1)] = BOOK_MOVE_COUNT
                out_moves = [m1] if m2 is None else [m1, m2]
                out_lines.append(append_moves(pos_spec, out_moves))

        return WorkerOutput(lines=out_lines, depth_map=out_depth_map, count_map=out_count_map)
    finally:
        if on_session_end is not None:
            on_session_end(sess)
        sess.close()


def main() -> int:
    args = parse_args()

    engine_path = args.engine
    if not os.path.isabs(engine_path):
        engine_path = os.path.abspath(engine_path)

    try:
        p = parse_think_args(args.think_args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    stop_event = threading.Event()
    sessions_lock = threading.Lock()
    active_sessions: List[USISession] = []

    def on_session_start(sess: USISession) -> None:
        with sessions_lock:
            active_sessions.append(sess)

    def on_session_end(sess: USISession) -> None:
        with sessions_lock:
            try:
                active_sessions.remove(sess)
            except ValueError:
                pass

    def request_stop_all_sessions() -> None:
        with sessions_lock:
            targets = list(active_sessions)
        for s in targets:
            s.request_stop()

    def _on_sigint(signum, frame):
        del signum, frame
        stop_event.set()
        print("interrupt received: stopping workers...", file=sys.stderr)
        request_stop_all_sessions()

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        pairs = build_input_pairs(p)
        positions = build_candidate_positions(pairs, p)

        existing_stats: Dict[str, BookDepthStat] = {}
        # 既存bookによるスキップ判定は、比較基準のdepthが明確なときだけ有効化する。
        # (nodes指定のみなどdepth未指定時は、実探索深さが局面ごとに変わるため事前判定しない)
        if p.depth_specified and os.path.exists(p.book_name):
            target_set = {
                (x[5:].strip() if x.startswith("sfen ") else x)
                for x in positions
            }
            existing_stats, existing_total = load_book_depth_stats(
                p.book_name,
                p.depth,
                target_set,
            )
            print(f"existing book positions: {existing_total}")

        positions, skipped = filter_positions_by_existing_book(positions, p, existing_stats)

        if not positions:
            if skipped > 0:
                print("no candidate positions: all positions were skipped by existing book")
                return 0
            print("error: no candidate positions", file=sys.stderr)
            return 2

        print(f"actual search positions: {len(positions)}")

        if not args.quiet:
            print(f"candidates: {len(positions)}")
            if existing_stats:
                print(f"skipped by existing book: {skipped}")

        worker_count = max(1, args.threads)
        chunks: List[List[str]] = [[] for _ in range(worker_count)]
        for i, s in enumerate(positions):
            chunks[i % worker_count].append(s)

        progress_lock = threading.Lock()
        progress_done = 0
        progress_total = len(positions)
        progress_start = time.time()
        next_speed_report_at = progress_start + 60.0

        def on_position_done(worker_id: int, pos_spec: str) -> None:
            nonlocal progress_done, next_speed_report_at
            if args.quiet:
                return
            with progress_lock:
                progress_done += 1
                now_ts = time.time()
                now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))
                print(f"[{progress_done}/{progress_total}:{worker_id}] {now_str} : {pos_spec}")

                if now_ts >= next_speed_report_at:
                    elapsed_min = max((now_ts - progress_start) / 60.0, 1e-9)
                    avg_speed = progress_done / elapsed_min
                    if avg_speed < 1.0:
                        inv = 1.0 / max(avg_speed, 1e-9)
                        print(f"[speed] avg {inv:.2f} min/position ({progress_done}/{progress_total})")
                    else:
                        print(f"[speed] avg {avg_speed:.2f} positions/min ({progress_done}/{progress_total})")
                    while next_speed_report_at <= now_ts:
                        next_speed_report_at += 60.0

        out_lines: List[str] = []
        out_depth_map: Dict[Tuple[str, str], int] = {}
        out_count_map: Dict[Tuple[str, str], int] = {}

        depth_override = p.depth if p.depth_specified else None
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            futures = [
                ex.submit(
                    worker_task,
                    worker_id,
                    engine_path,
                    args.cwd,
                    args.timeout,
                    args.usi_hash,
                    args.eval_dir,
                    p,
                    ck,
                    on_position_done,
                    stop_event,
                    on_session_start,
                    on_session_end,
                )
                for worker_id, ck in enumerate(chunks) if ck
            ]
            pending = set(futures)
            done_workers = 0

            save_interval = max(1, p.book_save_interval)
            next_save_at = time.time() + save_interval
            snapshot_no = 1
            last_saved_count = 0

            while pending:
                if stop_event.is_set():
                    request_stop_all_sessions()
                    break

                done_set, pending = wait(pending, timeout=2, return_when=FIRST_COMPLETED)

                for fut in done_set:
                    r = fut.result()
                    out_lines.extend(r.lines)
                    out_depth_map.update(r.depth_map)
                    out_count_map.update(r.count_map)
                    done_workers += 1
                    if not args.quiet:
                        print(f"worker done: {done_workers}/{len(futures)}")

                now = time.time()
                if now >= next_save_at:
                    if len(out_lines) > last_saved_count:
                        snapshot_name = make_periodic_snapshot_name(p.book_name, snapshot_no)
                        if not args.quiet:
                            print(
                                f"periodic save: {snapshot_name} "
                                f"(records={len(out_lines)})"
                            )
                        try:
                            save_snapshot_book(
                                engine_path,
                                args.cwd,
                                args.eval_dir,
                                out_lines,
                                snapshot_name,
                                depth_override,
                                out_depth_map,
                                out_count_map,
                            )
                            snapshot_no += 1
                            last_saved_count = len(out_lines)
                        except EngineError as e:
                            print(f"warning: periodic save failed: {e}", file=sys.stderr)

                    next_save_at = now + save_interval

            # 中断時も、すでに完了しているfutureの結果は回収しておく。
            if stop_event.is_set():
                done_set, pending = wait(pending, timeout=0, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    r = fut.result()
                    out_lines.extend(r.lines)
                    out_depth_map.update(r.depth_map)
                    out_count_map.update(r.count_map)
                    done_workers += 1

        if not out_lines:
            print("error: no result lines were generated", file=sys.stderr)
            return 1

        if not args.quiet:
            print(f"generated lines: {len(out_lines)}")

        if stop_event.is_set():
            return 130

        save_snapshot_book(
            engine_path,
            args.cwd,
            args.eval_dir,
            out_lines,
            p.book_name,
            depth_override,
            out_depth_map,
            out_count_map,
        )

        if not args.quiet:
            print(f"done: book created -> {p.book_name}")
        return 0

    except EngineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
