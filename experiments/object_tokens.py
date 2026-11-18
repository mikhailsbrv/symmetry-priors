"""Extract spatially matched DeiT and pixel descriptors for the paired COIL-20 views.

Each .npy is [object, view, side, shift, feature]. Frame side and shifts are
listed below; view index j corresponds to COIL file view 3*j, object index i
to object number i+1. Patch vectors flatten 4 x 4 x 192 in row-major order.
"""

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from imagenet_deit import load_deit, normalize_image
from scale_features import ROOT, place
from transfer_objects import CACHE as SOURCE_CACHE, COIL_SHA, COIL_URL, checked_file


CACHE = ROOT / ".cache" / "object_tokens"
SIDES = (64, 96, 128)
SHIFTS = ((-8, -8), (8, 8), (-8, 8), (8, -8))
VIEWS = tuple(range(0, 72, 3))
BLOCKS = (1, 3, 6, 9, 12)
LAYERS = [f"B{i}" for i in BLOCKS] + ["LN"]
SHAPE = (20, len(VIEWS), len(SIDES), len(SHIFTS))


def make_frames(archive, obj, views):
    frames = np.empty((len(views), *SHAPE[2:], 224, 224), dtype=np.uint8)
    for v, view in enumerate(views):
        with archive.open(f"coil-20-proc/obj{obj}__{view}.png") as fp:
            image = Image.open(fp).convert("L")
        assert image.size == (128, 128)
        for h, side in enumerate(SIDES):
            tile = image.resize((side, side), Image.Resampling.LANCZOS)
            frames[v, h] = place(np.asarray(tile), SHIFTS, 224)
    return frames


def pixels(frames):
    output = {name: np.empty((*frames.shape[:3], 256), dtype=np.float32)
              for name in ("fixed", "aligned")}
    for h, side in enumerate(SIDES):
        for pos, (dx, dy) in enumerate(SHIFTS):
            for name, window in (("fixed", 64), ("aligned", side)):
                x0, y0 = 112 + dx - window // 2, 112 + dy - window // 2
                region = frames[:, h, pos, y0:y0 + window, x0:x0 + window]
                grid = region.reshape(-1, 16, window // 16, 16, window // 16)
                output[name][:, h, pos] = grid.mean((2, 4), dtype=np.float32).reshape(-1, 256) / 255
    return output


def sampling_grids(device):
    # Use cell centres in continuous canvas-edge coordinates. For a 14-token
    # grid align_corners=False converts x to token index x/16 - 0.5.
    coord = (torch.arange(4, device=device, dtype=torch.float32) + 0.5) / 4 - 0.5
    gy, gx = torch.meshgrid(coord, coord, indexing="ij")
    offsets = torch.stack((gx, gy), dim=-1)
    centers = 112 + torch.tensor(SHIFTS, device=device, dtype=torch.float32)
    widths = torch.tensor(SIDES, device=device, dtype=torch.float32)[:, None, None, None, None]
    centers = centers[None, :, None, None, :]
    fixed = (centers + 64 * offsets).expand(len(SIDES), -1, -1, -1, -1)
    aligned = centers + widths * offsets
    return (torch.cat((fixed, aligned), dim=3) / 112 - 1).reshape(-1, 4, 8, 2)


def descriptors(module, frames, device, grid):
    nviews = len(frames)
    gray = torch.from_numpy(frames.reshape(-1, 224, 224)).to(device, torch.float32)[:, None] / 255
    batch = normalize_image(gray)
    batch_grid = grid.repeat(nviews, 1, 1, 1)
    output = {}

    def hook(layer):
        def read(_, inputs, tokens):
            assert tokens.shape[1:] == (197, 192)
            output[layer + "_cls"] = tokens[:, 0]
            field = tokens[:, 1:].transpose(1, 2).reshape(-1, 192, 14, 14)
            tiles = F.grid_sample(field, batch_grid, mode="bilinear", align_corners=False)
            for name, block in zip(("fixed", "aligned"), tiles.split(4, dim=-1)):
                output[layer + "_" + name] = block.permute(0, 2, 3, 1).reshape(len(tokens), -1)
        return read

    taps = [module.blocks[i - 1] for i in BLOCKS] + [module.norm]
    hooks = [tap.register_forward_hook(hook(name)) for name, tap in zip(LAYERS, taps)]
    try:
        module(batch)
    finally:
        for handle in hooks:
            handle.remove()
    return {name: value.reshape(nviews, *SHAPE[2:], -1).to(device="cpu", dtype=torch.float16).numpy()
            for name, value in output.items()}


@torch.inference_mode()
def run(condition, source):
    directory = CACHE / condition
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "ready.json"
    is_pixel = condition == "pixels"
    if is_pixel:
        dims = {"fixed": 256, "aligned": 256}
    else:
        device = torch.device("cuda")
        model = load_deit(condition).to(device).eval()
        grid = sampling_grids(device)
        dims = {f"{layer}_{name}": dim for layer in LAYERS
                for name, dim in (("cls", 192), ("fixed", 3072), ("aligned", 3072))}
    # The readout only opens a directory carrying a completion marker.
    marker.unlink(missing_ok=True)
    arrays = {name: np.lib.format.open_memmap(directory / (name + ".partial.npy"), mode="w+",
                                               dtype=np.float16, shape=(*SHAPE, dim))
              for name, dim in dims.items()}
    with zipfile.ZipFile(source) as archive:
        for obj in range(1, 21):
            for start in range(0, len(VIEWS), 2):
                frames = make_frames(archive, obj, VIEWS[start:start + 2])
                output = (pixels(frames) if is_pixel else
                          descriptors(model, frames, device, grid))
                for name, values in output.items():
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite values in {condition}/{name}, object {obj}")
                    arrays[name][obj - 1, start:start + len(frames)] = values
                for name in output:
                    if name.endswith("aligned"):
                        other = name.replace("aligned", "fixed")
                        np.testing.assert_array_equal(output[name][:, 0], output[other][:, 0])
            if obj % 5 == 0:
                print(f"{condition}: {obj}/20 objects", flush=True)
    for array in arrays.values():
        array.flush()
    del array
    arrays.clear()
    for name in dims:
        os.replace(directory / (name + ".partial.npy"), directory / (name + ".npy"))
    meta = dict(condition=condition, coil_sha256=COIL_SHA, sides=SIDES, shifts=SHIFTS, views=VIEWS,
                objects=list(range(1, 21)), dtype="float16", feature_dims=dims)
    temp_marker = directory / "ready.tmp"
    temp_marker.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    temp_marker.replace(marker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conditions", nargs="*", default=["pixels", "random", "imagenet"])
    parser.add_argument("--download", action="store_true", help="Download the original COIL-20 archive")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if any(name not in ("pixels", "random", "imagenet") for name in args.conditions):
        parser.error("conditions must be pixels, random or imagenet")
    source = checked_file(SOURCE_CACHE / "coil-20-proc.zip", COIL_SHA, COIL_URL, args.download)
    for name in args.conditions:
        run(name, source)
