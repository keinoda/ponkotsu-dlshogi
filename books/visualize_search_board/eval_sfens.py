import argparse
from eval import *
import os
import pickle
import tqdm

# sfen文字列のリストに対し推論を行う
def eval_sfens(session, sfens, batch_size, out=None):
    x1 = np.empty((batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32)
    x2 = np.empty((batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32)
    # ダミー推論をして推論を速くする
    eval(session, x1, x2)

    if out is None:
        out = dict()
    eval_board_list = []
    for sfen in sfens:
        board = cshogi.Board(sfen=sfen)
        key = board.zobrist_hash()
        if key not in out:
            eval_board_list.append(board)
        elif out[key].policy_logits is None:
            eval_board_list.append(board)

    for i in tqdm.tqdm(range(0, len(eval_board_list), batch_size)):
        for j in range(batch_size):
            if i + j < len(eval_board_list):
                make_input_features(eval_board_list[i + j], x1[j], x2[j])
            else:
                make_input_features(cshogi.Board(), x1[j], x2[j])
        policy_logits, values = eval(session, x1, x2)
        for j in range(batch_size):
            if i + j < len(eval_board_list):
                node = EvalNode()
                node.sfen = eval_board_list[i + j].sfen()
                node.policy_logits = make_logits(eval_board_list[i + j], policy_logits[j])
                node.value = values[j][0]
                node.legal_moves = list(eval_board_list[i + j].legal_moves)
                out[eval_board_list[i + j].zobrist_hash()] = node
    return out

# sfen文字列のリストに対し推論を行う(合法手のpolicyは計算しない)
# 末端ノード用
def eval_sfens_without_policy(session, sfens, batch_size, out=None):
    x1 = np.empty((batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32)
    x2 = np.empty((batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32)
    # ダミー推論をして推論を速くする
    eval(session, x1, x2)

    if out is None:
        out = dict()
    eval_board_list = [cshogi.Board(sfen=sfen) for sfen in sfens if cshogi.Board(sfen=sfen).zobrist_hash() not in out]

    for i in tqdm.tqdm(range(0, len(eval_board_list), batch_size)):
        for j in range(batch_size):
            if i + j < len(eval_board_list):
                make_input_features(eval_board_list[i + j], x1[j], x2[j])
            else:
                make_input_features(cshogi.Board(), x1[j], x2[j])
        _, values = eval(session, x1, x2)
        for j in range(batch_size):
            if i + j < len(eval_board_list):
                node = EvalNode()
                node.sfen = eval_board_list[i + j].sfen()
                node.value = values[j][0]
                out[eval_board_list[i + j].zobrist_hash()] = node
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("book")
    parser.add_argument("pickle")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tensorrt", action="store_true")
    args = parser.parse_args()

    session = onnxruntime.InferenceSession(args.model, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] if args.tensorrt else ['CUDAExecutionProvider', 'CPUExecutionProvider'])
    batch_size = args.batch_size

    out = None
    if os.path.exists(args.pickle):
        with open(args.pickle, "rb") as f:
            out = pickle.load(f)

    eval_board_list = []
    with open(args.book, "r") as f:
        lines = f.readlines()
        eval_sfen_list = [" ".join(line.split()[1:]).replace("\n", "") for line in lines if "sfen" in line]
        eval_sfen_list += [cshogi.rotate_sfen(sfen) for sfen in eval_sfen_list]

    out = eval_sfens(session, eval_sfen_list, batch_size, out)

    # 登録されている全局面の合法手の評価値を計算する
    sfens = []
    for node in out.values():
        for move in node.legal_moves:
            board = cshogi.Board(sfen=node.sfen)
            board.push_usi(cshogi.to_usi(move).decode())
            if board.zobrist_hash() not in out:
                sfens.append(board.sfen())

    out = eval_sfens_without_policy(session, sfens, batch_size, out)

    with open(args.pickle, "wb") as f:
        pickle.dump(out, f)
