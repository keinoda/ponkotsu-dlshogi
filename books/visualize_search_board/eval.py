import argparse
import cshogi
from cshogi.dlshogi import make_input_features, make_move_label, FEATURES1_NUM, FEATURES2_NUM
import numpy as np
import onnxruntime
import pickle

class EvalNode:
    def __init__(self):
        self.sfen = None
        self.policy_logits = None
        self.value = None
        self.legal_moves = None

    def board(self):
        if self.sfen is None:
            return None
        return cshogi.Board(sfen=self.sfen)

def eval(session, x1, x2):
    io_binding = session.io_binding()
    io_binding.bind_cpu_input('input1', x1)
    io_binding.bind_cpu_input('input2', x2)
    io_binding.bind_output('output_policy')
    io_binding.bind_output('output_value')
    session.run_with_iobinding(io_binding)
    return io_binding.copy_outputs_to_cpu()


def softmax_temperature_with_normalize(logits, temperature):
    logits /= temperature

    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    sum_exp_logits = np.sum(exp_logits)
    return exp_logits / sum_exp_logits


def make_logits(board, policy):
    legal_moves = list(board.legal_moves)
    legal_move_probabilities = np.empty(len(legal_moves), dtype=np.float32)
    for i, move in enumerate(legal_moves):
        move_label = make_move_label(move, board.turn)
        legal_move_probabilities[i]  = policy[move_label]
    return legal_move_probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("pickle")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--tensorrt", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.76)
    args = parser.parse_args()

    session = onnxruntime.InferenceSession(args.model, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'] if args.tensorrt else ['CUDAExecutionProvider', 'CPUExecutionProvider'])

    batch_size = args.batch_size
    temperature = args.temperature
    x1 = np.empty((batch_size, FEATURES1_NUM, 9, 9), dtype=np.float32)
    x2 = np.empty((batch_size, FEATURES2_NUM, 9, 9), dtype=np.float32)


    board = cshogi.Board()
    make_input_features(board, x1[0], x2[0])
    policy, values = eval(session, x1, x2)
    logits = make_logits(board, policy[0])

    node = EvalNode()
    node.sfen = board.sfen()
    node.policy_logits = logits
    node.value = values[0][0]

    out = dict()
    out[board.zobrist_hash()] = node

    with open(args.pickle, "wb") as f:
        pickle.dump(out, f, protocol=5)

    with open(args.pickle, "rb") as f:
        data = pickle.load(f)

    for node in data.values():
        print("sfen:", node.sfen)
        print("value:", node.value)
        print("policy logits:", node.policy_logits)
        print()
