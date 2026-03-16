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
    parser.add_argument(
        "--network",
        help="dlshogi network name (e.g. resnet35x512_fcl512)",
    )
    parser.add_argument(
        "network_name",
        nargs="?",
        help="dlshogi network name as positional arg (e.g. resnet35x512_fcl512)",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID, use -1 for CPU")
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--batches", type=int, default=2)
    parser.add_argument(
        "--allow_missing_swa_model",
        action="store_true",
        help="If checkpoint has no swa_model, use model weights as SWA fallback",
    )
    parser.add_argument(
        "--apply_swa_device_fix",
        action="store_true",
        help="Apply fix by moving swa_model to target device after model.to(device)",
    )
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
    )
    args = parser.parse_args()

    network = args.network or args.network_name or "resnet35x512_fcl512"

    if args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        device_type_str = "cuda"
    else:
        device = torch.device("cpu")
        device_type_str = "cpu"

    print(f"network={network}")

    # Emulate dlshogi resume order:
    # model is created on CPU, swa_model is cloned from that CPU model,
    # then only model is moved to CUDA unless the fix is applied.
    model = policy_value_network(network)
    model.cpu()
    swa_model = AveragedModel(model)

    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    if args.apply_swa_device_fix:
        swa_model.to(device)
        print("mode=patched")
    else:
        print("mode=unpatched")

    if "swa_model" not in checkpoint:
        if not args.allow_missing_swa_model:
            raise RuntimeError(
                "checkpoint does not contain swa_model. "
                "Use a checkpoint created with --use_swa (typically epoch >= SWA start), "
                "or run with --allow_missing_swa_model to fall back to model weights."
            )
        print("WARN: checkpoint has no swa_model; using initial AveragedModel weights")
    else:
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