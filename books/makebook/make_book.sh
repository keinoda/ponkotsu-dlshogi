#!/bin/bash

# デフォルト値
N=1
EVALDIR=""
BOOKDIR=""
MODEL=""

# オプション解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --iterations)
            N="$2"
            shift 2
            ;;
        --eval-dir)
            EVALDIR="$2"
            shift 2
            ;;
        --book-dir)
            BOOKDIR="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --iterations N --eval-dir DIR --book-dir DIR --model PATH"
            echo "Options:"
            echo "  --iterations N      Number of iterations (default: 1)"
            echo "  --eval-dir DIR      Path to evaluation directory"
            echo "  --book-dir DIR      Path to book directory"
            echo "  --model PATH        Path to ONNX model file"
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
if [ -z "$EVALDIR" ] || [ -z "$BOOKDIR" ] || [ -z "$MODEL" ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 --iterations N --eval-dir DIR --book-dir DIR --model PATH"
    exit 1
fi

# スクリプトディレクトリを計算
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pythonスクリプトへのパスを定義
SCRIPTS_DIR="$SCRIPT_DIR/scripts"
GET_SEARCH_BOARD_SCRIPT="$SCRIPTS_DIR/get_search_board_multipv.py"
EVAL_SFENS_SCRIPT="$SCRIPTS_DIR/eval_sfens.py"
MCTS_SCRIPT="$SCRIPTS_DIR/mcts_on_book_dl.py"

# 元のディレクトリを保存
ORIGINAL_DIR="$(pwd)"

# 終了時に元のディレクトリに戻す
trap "cd '$ORIGINAL_DIR'" EXIT

# BOOKDIR下で作業
cd "$BOOKDIR" || exit 1

# C++拡張のビルドチェック
MCTS_CPP_SO=$(find "$SCRIPTS_DIR" -name "mcts_cpp.cpython-*.so" 2>/dev/null | head -1)
if [ -z "$MCTS_CPP_SO" ]; then
    echo "Building mcts_cpp extension..."
    python -c "import pybind11" >/dev/null 2>&1 || python -m pip install pybind11
    (cd "$SCRIPTS_DIR" && python setup_cpp.py build_ext --inplace)
fi

# パス置換の関数
setup_startup_file() {
    local template_file=$1
    sed -i \
      -e "s|@EVALDIR@|$EVALDIR|g" \
      -e "s|@BOOKDIR@|$BOOKDIR|g" \
      "$template_file"
}

for ((i=1; i<=$N; i++)); do
    # think処理
    cp "$SCRIPT_DIR/startup_think.txt" startup.txt
    setup_startup_file startup.txt
    ./YaneuraOu-by-gcc
    
    # merge_delta処理
    cp "$SCRIPT_DIR/startup_merge_delta.txt" startup.txt
    setup_startup_file startup.txt
    ./YaneuraOu-by-gcc-no-search
    
    # petashock処理
    cp "$SCRIPT_DIR/startup_petashock.txt" startup.txt
    setup_startup_file startup.txt
    ./YaneuraOu-by-gcc-no-search
    
    # multipv取得
    python "$GET_SEARCH_BOARD_SCRIPT" test_book_petashock.db first_board_sfens.txt test_multipv.sfens
    
    # think_multipv処理
    cp "$SCRIPT_DIR/startup_think_multipv.txt" startup.txt
    setup_startup_file startup.txt
    ./YaneuraOu-by-gcc

    # merge_multipv_delta処理
    cp "$SCRIPT_DIR/startup_merge_multipv_delta.txt" startup.txt
    setup_startup_file startup.txt
    ./YaneuraOu-by-gcc-no-search
    
    # merge処理
    cp "$SCRIPT_DIR/startup_merge.txt" startup.txt
    setup_startup_file startup.txt
    ./YaneuraOu-by-gcc-no-search
    
    # petashock処理
    cp "$SCRIPT_DIR/startup_petashock.txt" startup.txt
    setup_startup_file startup.txt
    ./YaneuraOu-by-gcc-no-search
    
    # eval処理
    python "$EVAL_SFENS_SCRIPT" "$MODEL" out_book/test_book.db eval_dl --batch_size 1024 --device cuda
    
    # mcts処理
    python "$MCTS_SCRIPT" test_book_petashock.db eval_dl test.sfens --root_sfens root_sfens.txt csa_root.sfens --use-cpp
done
