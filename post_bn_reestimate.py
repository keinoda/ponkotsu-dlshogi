import argparse
import os
import time

import numpy as np
import torch
from torch.optim.swa_utils import update_bn

from dlshogi import serializers
from dlshogi.data_loader import Hcpe3DataLoader
from dlshogi.network.policy_value_network import policy_value_network


def limited_hcpe_loader(data, batchsize, device, max_batches, total_batches):
    count = 0
    start = time.time()
    for x1, x2, _t1, _t2, _value in Hcpe3DataLoader(data, batchsize, device):
        yield {"x1": x1, "x2": x2}
        count += 1
        elapsed = time.time() - start
        if total_batches is not None:
            pct = 100.0 * count / total_batches
            print(
                f"\rBN re-estimation: {count}/{total_batches} batches ({pct:5.1f}%) elapsed {elapsed:,.1f}s",
                end="",
                flush=True,
            )
        else:
            print(
                f"\rBN re-estimation: {count} batches elapsed {elapsed:,.1f}s",
                end="",
                flush=True,
            )
        if max_batches is not None and max_batches > 0 and count >= max_batches:
            break
    print()


def main():
    parser = argparse.ArgumentParser(description="Run SWA BN re-estimation as post process")
    parser.add_argument(
        "train_data",
        nargs="*",
        help="hcpe3 training data files (optional when --cache exists)",
    )
    parser.add_argument(
        "--model",
        "--checkpoint",
        dest="model_path",
        required=True,
        help="dlshogi npz model path",
    )
    parser.add_argument("--network", required=True, help="dlshogi network name")
    parser.add_argument("--output_model", required=True, help="output npz model path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, use -1 for CPU")
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="maximum number of batches for BN re-estimation (omit or 0 for full pass)",
    )
    parser.add_argument("--use_average", action="store_true")
    parser.add_argument("--use_evalfix", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--patch", type=str)
    parser.add_argument("--cache", type=str)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["float16", "bfloat16"], default="float16")
    args = parser.parse_args()

    if not args.train_data and not args.cache:
        parser.error("Specify either train_data files or --cache.")
    if not args.train_data and args.cache and not os.path.isfile(args.cache):
        parser.error(f"cache file not found: {args.cache}")
    if args.max_batches is not None and args.max_batches < 0:
        parser.error("--max_batches must be >= 0 when specified.")

    if args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        device_type_str = "cuda"
    else:
        device = torch.device("cpu")
        device_type_str = "cpu"

    model = policy_value_network(args.network)
    model.cpu()

    serializers.load_npz(args.model_path, model)
    print(f"Loaded weights from npz model: {args.model_path}")

    model.to(device)

    train_len, _actual_len = Hcpe3DataLoader.load_files(
        args.train_data,
        args.use_average,
        args.use_evalfix,
        args.temperature,
        args.patch,
        args.cache,
    )
    train_data = np.arange(train_len, dtype=np.uint64)

    if args.max_batches is not None and args.max_batches > 0:
        total_batches = min((train_len + args.batchsize - 1) // args.batchsize, args.max_batches)
    else:
        total_batches = (train_len + args.batchsize - 1) // args.batchsize

    print(f"Starting BN re-estimation: total_batches={total_batches}, batchsize={args.batchsize}")

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    forward_ = model.forward
    model.forward = lambda x: forward_(**x)
    try:
        with torch.autocast(device_type_str, enabled=args.use_amp, dtype=amp_dtype):
            update_bn(
                limited_hcpe_loader(train_data, args.batchsize, device, args.max_batches, total_batches),
                model,
            )
    finally:
        del model.forward

    serializers.save_npz(args.output_model, model)
    print(f"Saved: {args.output_model}")


if __name__ == "__main__":
    main()
