"""Load the authors' ImageNet-1k DeiT-Tiny/16 in this repository's tensor implementation."""

import argparse
import hashlib
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model import DeiTTiny

WEIGHT_URL = "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth"
WEIGHT_SHA256 = "a1311bcf4f24e3c95adaa75535db67bc4412d95535b98f7c1dfd1164dda41c97"
WEIGHT_PATH = ROOT / ".cache" / WEIGHT_URL.rsplit("/", 1)[-1]


def normalize_image(images):
    """Convert a float [N, 1 or 3, H, W] batch in [0, 1] to normalized RGB."""
    if images.ndim != 4 or images.shape[1] not in (1, 3) or not images.is_floating_point():
        raise ValueError("Expected floating-point grayscale or RGB images in NCHW layout")
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (images - mean) / std


def official_state(download=False):
    if not WEIGHT_PATH.exists():
        if not download:
            raise FileNotFoundError(f"Run python experiments/imagenet_deit.py --download for {WEIGHT_PATH.name}")
        WEIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(WEIGHT_URL, str(WEIGHT_PATH), hash_prefix=WEIGHT_SHA256)
    with WEIGHT_PATH.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != WEIGHT_SHA256:
            raise ValueError(f"Unexpected DeiT checkpoint checksum: {WEIGHT_PATH}")
    return torch.load(WEIGHT_PATH, map_location="cpu", weights_only=True)["model"]


def converted_state(official):
    converted = {}
    for key, value in official.items():
        if ".attn.qkv." in key:
            block, parameter = key.split(".attn.qkv.")
            for name, projection in zip(("query", "key", "value"), value.chunk(3)):
                for head, weights in enumerate(projection.chunk(3)):
                    converted[f"{block}.attention.heads.{head}.{name}.{parameter}"] = weights
        else:
            key = key.replace("pos_embed", "position").replace("head.", "classifier.")
            key = key.replace("patch_embed.proj.", "patch_projection.")
            key = key.replace(".attn.proj.", ".attention.projection.")
            converted[key] = value.flatten(1) if key == "patch_projection.weight" else value
    return converted


def load_deit(condition, *, seed=42, download=False):
    """Return a CPU model in evaluation mode. Random weights use the original initializer distributions."""
    if condition not in {"imagenet", "random"}:
        raise ValueError(f"Unknown condition {condition}")
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.default_generator.manual_seed(seed)
        model = DeiTTiny(num_classes=1000)
        if condition == "imagenet":
            model.load_state_dict(converted_state(official_state(download)), strict=True)
        else:
            # Original PatchEmbed uses the default Conv2d initializer on 3 x 16 x 16 inputs.
            limit = model.patch_projection.weight.shape[1] ** -0.5
            model.patch_projection.weight.uniform_(-limit, limit)
            model.patch_projection.bias.uniform_(-limit, limit)
    return model.eval()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch the original ImageNet checkpoint if missing")
    options = parser.parse_args()
    load_deit("imagenet", download=options.download)
    print(f"Verified {WEIGHT_PATH.name}, SHA-256 {WEIGHT_SHA256}")
