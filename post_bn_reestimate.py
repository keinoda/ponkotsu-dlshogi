import argparse

import numpy as np
import torch
from torch.optim.swa_utils import AveragedModel, update_bn

from dlshogi import serializers
from dlshogi.data_loader import Hcpe3DataLoader
from dlshogi.network.policy_value_network import policy_value_network


def limited_hcpe_loader(data, batchsize, device, max_batches):
    count = 0
    for x1, x2, _t1, _t2, _value in Hcpe3DataLoader(data, batchsize, device):
        yield {"x1": x1, "x2": x2}
        count += 1
        if max_batches > 0 and count >= max_batches:
            break


def main():
    parser = argparse.ArgumentParser(description="Run SWA BN re-estimation as post process")
    parser.add_argument("train_data", nargs="+", help="hcpe3 training data files")
    parser.add_argument("--checkpoint", required=True, help="checkpoint path")
    parser.add_argument("--network", required=True, help="dlshogi network name")
    parser.add_argument("--output_model", required=True, help="output npz model path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, use -1 for CPU")
    parser.add_argument("--batchsize", type=int, default=1024)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means full pass")
    parser.add_argument("--use_average", action="store_true")
    parser.add_argument("--use_evalfix", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--patch", type=str)
    parser.add_argument("--cache", type=str)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--amp_dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument(
        "--allow_missing_swa_model",
        action="store_true",
        help="fallback to model weights when checkpoint has no swa_model",
    )
    args = parser.parse_args()

    if args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        device_type_str = "cuda"
    else:
        device = torch.device("cpu")
        device_type_str = "cpu"

    model = policy_value_network(args.network)
    model.cpu()
    swa_model = AveragedModel(model)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "swa_model" in checkpoint:
        swa_model.load_state_dict(checkpoint["swa_model"])
    elif args.allow_missing_swa_model and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print("WARN: swa_model not found. Falling back to model weights.")
    else:
        raise RuntimeError("checkpoint does not contain swa_model")

    swa_model.to(device)

    train_len, _actual_len = Hcpe3DataLoader.load_files(
        args.train_data,
        args.use_average,
        args.use_evalfix,
        args.temperature,
        args.patch,
        args.cache,
    )
    train_data = np.arange(train_len, dtype=np.uint64)

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    forward_ = swa_model.forward
    swa_model.forward = lambda x: forward_(**x)
    try:
        with torch.autocast(device_type_str, enabled=args.use_amp, dtype=amp_dtype):
            update_bn(
                limited_hcpe_loader(train_data, args.batchsize, device, args.max_batches),
                swa_model,
            )
    finally:
        del swa_model.forward

    serializers.save_npz(args.output_model, swa_model.module)
    print(f"Saved: {args.output_model}")


if __name__ == "__main__":
    main()
