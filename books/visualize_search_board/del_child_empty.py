import argparse
import pickle

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('input')
    args.add_argument('output')
    args = args.parse_args()

    with open(args.input, 'rb') as f:
        data = pickle.load(f)

    keys_to_delete = []
    for key, node in data.items():
        if node.legal_moves is None:
            keys_to_delete.append(key)

    for key in keys_to_delete:
        del data[key]

    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
