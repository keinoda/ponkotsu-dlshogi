import argparse
import os

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('in_book')
    args.add_argument('out_book')
    args = args.parse_args()

    if not os.path.exists(args.in_book):
        raise FileNotFoundError(f"Input book file '{args.in_book}' does not exist.")

    with open(args.in_book, 'r') as f:
        in_books = f.readlines()

    out_books = []
    cnt = 0
    for line in in_books:
        if line.startswith('sfen') or line.startswith('#'):
            cnt = 0
            out_books.append(line)
        else:
            next_move_info = line.split()
            if cnt == 0:
                eval = next_move_info[2]
                next_move_info[2] = '0'
                out_books.append(' '.join(next_move_info) + '\n')
            elif next_move_info[2] == eval:
                next_move_info[2] = '0'
                out_books.append(' '.join(next_move_info) + '\n')
            cnt += 1

    with open(args.out_book, 'w') as f:
        f.writelines(out_books)
