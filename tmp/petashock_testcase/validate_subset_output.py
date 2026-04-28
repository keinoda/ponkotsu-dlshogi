import argparse
from pathlib import Path
import subprocess
import sys
import time

import cshogi

BOOK_MOVES_THRESHOLD = 4


def repo_root_from_this_file():
    return Path(__file__).resolve().parents[2]


def norm(s):
    return ' '.join(s.strip().split(' ')[:3])


def build_first_path_from_book(book_path):
    first_path = book_path.parent / (book_path.stem + '.first_board_sfens.txt')
    line_count = 0
    with book_path.open() as src, first_path.open('w') as dst:
        first = True
        for raw in src:
            if first:
                first = False
                continue
            line = raw.strip()
            if line.startswith('sfen '):
                dst.write(line[5:] + '\n')
                line_count += 1
    print(f'generated_first_path={first_path} lines={line_count}')
    return first_path


def run_target_script(python_exe, script_path, book_path, first_path, out_path, threshold):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(python_exe),
        str(script_path),
        str(book_path),
        str(first_path),
        str(out_path),
        '--book_moves_threshold',
        str(threshold),
    ]
    print('run_command=' + ' '.join(cmd))
    start = time.perf_counter()
    subprocess.run(cmd, check=True)
    elapsed = time.perf_counter() - start
    return elapsed


def build_counts(book_path):
    counts = {}
    cur = None
    with book_path.open() as f:
        first = True
        for raw in f:
            if first:
                first = False
                continue
            line = raw.strip()
            if not line:
                continue
            if line.startswith('sfen '):
                cur = norm(line[5:])
                counts.setdefault(cur, 0)
            elif cur is not None:
                counts[cur] += 1
    return counts


def build_expected(first_path, counts, threshold):
    first_sfens = [l.strip() for l in first_path.read_text().splitlines() if l.strip()]
    expected = []
    for sfen in first_sfens:
        key = norm(sfen)
        rkey = norm(cshogi.rotate_sfen(sfen))
        c = counts.get(key)
        if c is None:
            c = counts.get(rkey)
        if c is not None and c < threshold:
            expected.append(f'sfen {sfen}')
    return expected


def read_output_lines(path):
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def main():
    script_path = repo_root_from_this_file() / 'books' / 'makebook' / 'scripts' / 'get_search_board_multipv.py'
    baseline_script_path = Path(__file__).resolve().parent / 'get_search_board_multipv_old.py'

    parser = argparse.ArgumentParser()
    parser.add_argument('book_path', type=Path)
    parser.add_argument('out_path', type=Path)
    args = parser.parse_args()

    first_path = build_first_path_from_book(args.book_path)

    optimized_elapsed = run_target_script(
        Path(sys.executable),
        script_path,
        args.book_path,
        first_path,
        args.out_path,
        BOOK_MOVES_THRESHOLD,
    )

    counts = build_counts(args.book_path)
    expected = build_expected(first_path, counts, BOOK_MOVES_THRESHOLD)

    actual = read_output_lines(args.out_path)
    print(f'expected={len(expected)} actual={len(actual)}')
    print('match=' + str(expected == actual))
    print(f'optimized_elapsed_sec={optimized_elapsed:.3f}')

    if expected != actual:
        print('validation_failed=true')
        sys.exit(1)

    if baseline_script_path.exists():
        baseline_out_path = args.out_path.with_name(args.out_path.stem + '.baseline.sfens')
        baseline_elapsed = run_target_script(
            Path(sys.executable),
            baseline_script_path,
            args.book_path,
            first_path,
            baseline_out_path,
            BOOK_MOVES_THRESHOLD,
        )
        baseline_actual = read_output_lines(baseline_out_path)
        baseline_match = expected == baseline_actual
        print(f'baseline_elapsed_sec={baseline_elapsed:.3f}')
        print('baseline_match=' + str(baseline_match))

        if baseline_match:
            speedup = baseline_elapsed / optimized_elapsed if optimized_elapsed > 0 else float('inf')
            print(f'speedup_vs_old={speedup:.3f}x')
        else:
            print('speedup_vs_old=unavailable')
    else:
        print(f'baseline_script_missing={baseline_script_path}')

    print('validation_failed=false')


if __name__ == "__main__":
    main()
