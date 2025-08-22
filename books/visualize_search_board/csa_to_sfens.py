import argparse
import cshogi
from cshogi import CSA, BLACK_WIN, WHITE_WIN
import glob
import os

args = argparse.ArgumentParser()
args.add_argument('dir')
args.add_argument('sfens')
args.add_argument('--filter_rating', type=int, default=3700)
args.add_argument('--filter_score', type=int, default=500)
args = args.parse_args()

csa_file_list = glob.glob(os.path.join(args.dir, '**', '*.csa'), recursive=True)

boards = dict()
for csa_file in csa_file_list:
    parser = CSA.Parser()
    parser.parse_csa_file(csa_file)

    # 負けた側のレーティングでフィルタリングする
    if parser.win == BLACK_WIN:
        rating = parser.ratings[1]
    elif parser.win == WHITE_WIN:
        rating = parser.ratings[0]
    else:
        rating = max(parser.ratings)
    if rating < args.filter_rating:
        continue

    board = cshogi.Board()
    for move, score in zip(parser.moves, parser.scores):
        board.push(move)
        boards[board.zobrist_hash()] = f"sfen {board.sfen()}\n"
        if abs(score) > args.filter_score:
            break

sfens = list(boards.values())
with open(args.sfens, "w") as f:
    f.writelines(sfens)
