import argparse

args = argparse.ArgumentParser()
args.add_argument('sfens')
args.add_argument('add_sfens')
args = args.parse_args()

with open(args.sfens, "r") as f:
    sfens = f.readlines()

if "sfen" not in sfens[-1]:
    sfens = sfens[:-1]

with open(args.add_sfens, "r") as f:
    add_sfens = f.readlines()

if add_sfens[-1] == "\n":
    add_sfens = add_sfens[:-1]

add_sfens = [f"sfen {s}" for s in add_sfens]

sfens += add_sfens

with open(args.sfens, "w") as f:
    f.writelines(sfens)

with open(args.add_sfens, "w") as f:
    f.writelines([])
