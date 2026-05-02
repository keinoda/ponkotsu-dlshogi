#!/bin/bash

BOOKDIR=""
DEFAULT_THREADS="$(nproc)"
THREADS="$DEFAULT_THREADS"

# オプション解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --book-dir)
            BOOKDIR="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --book-dir DIR [--threads N]"
            echo "Options:"
            echo "  --book-dir DIR      Path to book directory"
            echo "  --threads N         Number of engine threads (default: $DEFAULT_THREADS)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 引数チェック
if [ -z "$BOOKDIR" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 --book-dir DIR [--threads N]"
    exit 1
fi

if ! [[ "$THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: --threads must be a positive integer"
    exit 1
fi

# スクリプトディレクトリを計算
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pythonスクリプトへのパスを定義
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
HIDE_BOOK_EVAL_SCRIPT="$SCRIPTS_DIR/hide_book_eval.py"

# 元のディレクトリを保存
ORIGINAL_DIR="$(pwd)"

# 終了時に元のディレクトリに戻す
trap "cd '$ORIGINAL_DIR'" EXIT

# BOOKDIR下で作業
cd "$BOOKDIR" || exit 1

# パス置換の関数
setup_startup_file() {
    local template_file=$1
    sed -i \
      -e "s|@BOOKDIR@|$BOOKDIR|g" \
      -e "s|@THREADS@|$THREADS|g" \
      "$template_file"
}

# shrink処理
cp "$SCRIPT_DIR/startup_petashock_shrink.txt" startup.txt
setup_startup_file startup.txt
./YaneuraOu-by-gcc-no-search

# hide処理
python "$HIDE_BOOK_EVAL_SCRIPT" "$BOOKDIR/test_book_petashocktest_book_petashock_shrink.db" "$BOOKDIR/test_book_petashocktest_book_petashock_shrink.db"
