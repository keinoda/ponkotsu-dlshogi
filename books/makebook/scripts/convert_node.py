import pickle
import argparse
import sys
from pathlib import Path
import numpy as np
import tqdm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "makebook" / "scripts"))
from mcts_on_book_dl import Node, softmax_temperature_with_normalization

args = argparse.ArgumentParser()
args.add_argument('input')
args.add_argument('output')
args = args.parse_args()

with open(args.input, 'rb') as f:
    data = pickle.load(f)

out = dict()
for key, value in tqdm.tqdm(data.items()):
    out[key] = Node()
    out[key].board = value.sfen # 良くはないが一旦boardにsfenを入れる
    out[key].child_move = None
    if value.legal_moves is not None:
        if len(value.legal_moves) > 0:
            out[key].child_move = value.legal_moves
            out[key].child_move_count = np.zeros(len(value.legal_moves), dtype=np.float32)
            out[key].child_score_sum = np.zeros(len(value.legal_moves), dtype=np.float32)
            out[key].child_policy = softmax_temperature_with_normalization(value.policy_logits, 1.76)
    out[key].value = value.value

print("Converting {} nodes".format(len(out)))
with open(args.output, 'wb') as f:
    pickle.dump(out, f, protocol=5)
