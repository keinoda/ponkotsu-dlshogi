#!/bin/sh

# 第1引数を save_dir として受け取る
save_dir="$1"

# 未指定なら使い方を表示して終了
if [ -z "$save_dir" ]; then
  echo "Usage: $0 <save_dir>"
  exit 1
fi

python download_wcsc_kifu.py "$save_dir"
python get_wcsc_kifu.py https://www.computer-shogi.org/live/wcsc36/lower.html https://www.computer-shogi.org/live/wcsc36/list.txt "$save_dir"
