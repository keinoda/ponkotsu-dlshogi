#!/bin/sh

exec_dir=$1
cd $2

yesterday=$(date -d "yesterday" +%Y%m%d)

wget http://wdoor.c.u-tokyo.ac.jp/shogi/x/kifuarchive-daily/wdoor$yesterday.7z
7z x wdoor$yesterday.7z
rm wdoor$yesterday.7z

python $exec_dir/csa_to_sfens.py $yesterday add.sfens --filter_score 300
rm -rf $yesterday
