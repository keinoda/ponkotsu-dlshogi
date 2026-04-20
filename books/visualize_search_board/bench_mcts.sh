#!/bin/bash
set -eu

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- 引数パース ---
BOOK=""
DL_PICKLE=""
ROOT_SFENS_ARGS=()
EXTRA_ARGS=()

usage() {
    echo "Usage: $0 --book <book_file> --dl-pickle <dl_pickle_file> [--root_sfens <file>...] [-- extra args...]"
    echo ""
    echo "Cython vs pure Python MCTS ベンチマーク (実データ比較)"
    echo ""
    echo "Options:"
    echo "  --book           定跡ファイルのパス (必須)"
    echo "  --dl-pickle      DL推論 pickle のパス (必須)"
    echo "  --root_sfens     探索開始局面ファイル (複数指定可)"
    echo "  -- extra args    mcts_on_book_dl.py に渡す追加引数"
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        --book)
            BOOK="$2"; shift 2 ;;
        --dl-pickle)
            DL_PICKLE="$2"; shift 2 ;;
        --root_sfens)
            shift
            while [ $# -gt 0 ] && [ "${1:0:2}" != "--" ]; do
                ROOT_SFENS_ARGS+=("$1"); shift
            done
            ;;
        --)
            shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)
            usage ;;
        *)
            EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [ -z "$BOOK" ] || [ -z "$DL_PICKLE" ]; then
    echo "ERROR: --book と --dl-pickle は必須です"
    usage
fi

# --- 一時ディレクトリ ---
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

SFENS_PY="$TMPDIR/sfens_python.txt"
SFENS_CY="$TMPDIR/sfens_cython.txt"
FIRST_PY="$TMPDIR/first_board_python.txt"
FIRST_CY="$TMPDIR/first_board_cython.txt"
LOG_PY="$TMPDIR/log_python.txt"
LOG_CY="$TMPDIR/log_cython.txt"

# --- root_sfens 引数の組み立て ---
ROOT_SFENS_CMD=()
if [ ${#ROOT_SFENS_ARGS[@]} -gt 0 ]; then
    ROOT_SFENS_CMD=("--root_sfens" "${ROOT_SFENS_ARGS[@]}")
fi

# --- Cython ビルド ---
echo "============================================================"
echo "Cython ビルド確認"
echo "============================================================"
if ! python -c "import mcts_core" 2>/dev/null; then
    echo "mcts_core が見つかりません。ビルドします..."
    python setup_cython.py build_ext --inplace
    echo "ビルド完了"
else
    echo "mcts_core は既にビルド済み"
fi
echo ""

# --- Python 版 ---
echo "============================================================"
echo "[1/2] Pure Python で実行"
echo "============================================================"
START_PY=$(python -c "import time; print(time.time())")
python mcts_on_book_dl.py \
    "$BOOK" "$DL_PICKLE" "$SFENS_PY" \
    --visited-nodes-limit 0 \
    --first_board_sfen_output "$FIRST_PY" \
    "${ROOT_SFENS_CMD[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_PY"
END_PY=$(python -c "import time; print(time.time())")
TIME_PY=$(python -c "print(f'{$END_PY - $START_PY:.2f}')")
echo ""

# --- Cython 版 ---
echo "============================================================"
echo "[2/2] Cython で実行"
echo "============================================================"
START_CY=$(python -c "import time; print(time.time())")
python mcts_on_book_dl.py \
    "$BOOK" "$DL_PICKLE" "$SFENS_CY" \
    --use-cython \
    --visited-nodes-limit 0 \
    --first_board_sfen_output "$FIRST_CY" \
    "${ROOT_SFENS_CMD[@]}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$LOG_CY"
END_CY=$(python -c "import time; print(time.time())")
TIME_CY=$(python -c "print(f'{$END_CY - $START_CY:.2f}')")
echo ""

# --- 結果比較 ---
echo "============================================================"
echo "結果比較"
echo "============================================================"

echo ""
echo "--- 実行時間 ---"
echo "  Python : ${TIME_PY}s"
echo "  Cython : ${TIME_CY}s"
SPEEDUP=$(python -c "py=$TIME_PY; cy=$TIME_CY; print(f'{py/cy:.1f}x' if cy > 0 else 'N/A')")
echo "  Speedup: ${SPEEDUP}"

echo ""
echo "--- sfens 出力 diff ---"
if diff -q "$SFENS_PY" "$SFENS_CY" > /dev/null 2>&1; then
    LINES_PY=$(wc -l < "$SFENS_PY" | tr -d ' ')
    echo "  MATCH (${LINES_PY} lines)"
else
    echo "  DIFFERS:"
    diff --unified=0 "$SFENS_PY" "$SFENS_CY" | head -30
fi

echo ""
echo "--- first_board_sfens diff ---"
if diff -q "$FIRST_PY" "$FIRST_CY" > /dev/null 2>&1; then
    LINES_FB=$(wc -l < "$FIRST_PY" | tr -d ' ')
    echo "  MATCH (${LINES_FB} lines)"
else
    echo "  DIFFERS:"
    diff --unified=0 "$FIRST_PY" "$FIRST_CY" | head -30
fi

echo ""
echo "--- Total search (ログから抽出) ---"
TOTAL_PY=$(grep 'Total search:' "$LOG_PY" || echo "N/A")
TOTAL_CY=$(grep 'Total search:' "$LOG_CY" || echo "N/A")
echo "  Python : ${TOTAL_PY}"
echo "  Cython : ${TOTAL_CY}"

echo ""
echo "============================================================"
echo "完了"
echo "============================================================"
