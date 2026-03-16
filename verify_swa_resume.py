import argparse

import torch
from torch.optim.swa_utils import AveragedModel, update_bn

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi.network.policy_value_network import policy_value_network


def synthetic_loader(batchsize, device, batches):
    for _ in range(batches):
        x1 = torch.randn(batchsize, FEATURES1_NUM, 9, 9, device=device)
        x2 = torch.randn(batchsize, FEATURES2_NUM, 9, 9, device=device)
        yield {"x1": x1, "x2": x2}


def main():
    parser = argparse.ArgumentParser(description="Verify SWA resume/update_bn path")
    parser.add_argument("--resume", required=True, help="checkpoint path")
    parser.add_argument("--network", required=True, help="network type used for the checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, use -1 for CPU")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
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

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    swa_model.to(device)

    if "swa_model" not in checkpoint:
        raise RuntimeError("checkpoint does not contain swa_model")
    swa_model.load_state_dict(checkpoint["swa_model"])

    parameter = next(swa_model.parameters())
    print(f"swa param device={parameter.device}, dtype={parameter.dtype}")
    print(f"target device={device}")
    assert parameter.device == device, f"swa_model is on {parameter.device}, expected {device}"

    amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
    forward_ = swa_model.forward
    swa_model.forward = lambda x: forward_(**x)

    try:
        with torch.autocast(device_type_str, enabled=args.use_amp, dtype=amp_dtype):
            update_bn(synthetic_loader(args.batchsize, device, args.batches), swa_model)
    finally:
        del swa_model.forward

    print("OK: update_bn completed without device/dtype mismatch")


if __name__ == "__main__":
    main()