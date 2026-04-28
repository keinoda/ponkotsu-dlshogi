import argparse
from pathlib import Path
import subprocess
import sys

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
    subprocess.run(cmd, check=True)


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


def main():
    script_path = repo_root_from_this_file() / 'books' / 'makebook' / 'scripts' / 'get_search_board_multipv.py'

    parser = argparse.ArgumentParser()
    parser.add_argument('book_path', type=Path)
    parser.add_argument('out_path', type=Path)
    args = parser.parse_args()

    first_path = build_first_path_from_book(args.book_path)

    run_target_script(
        Path(sys.executable),
        script_path,
        args.book_path,
        first_path,
        args.out_path,
        BOOK_MOVES_THRESHOLD,
    )

    counts = build_counts(args.book_path)
    expected = build_expected(first_path, counts, BOOK_MOVES_THRESHOLD)

    actual = [l.strip() for l in args.out_path.read_text().splitlines() if l.strip()]
    print(f'expected={len(expected)} actual={len(actual)}')
    print('match=' + str(expected == actual))

    if expected != actual:
        print('validation_failed=true')
        sys.exit(1)

    print('validation_failed=false')


if __name__ == "__main__":
    main()
