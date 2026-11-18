"""Input-only square seam detection, known-frame control, and a check of CLS attention mass."""

import argparse
import csv
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from area_prior import foreground, forward_with_prior, resize_box, tensor
from imagenet_deit import ROOT, load_deit
from prior_data import batches
from scale_operator import write_rows


def line_gradients(frame, axis):
    """Gradient at the boundary before each pixel, averaged across RGB channels."""
    arr = frame.astype(np.float32)
    diff = np.abs(np.diff(arr, axis=axis, prepend=np.take(arr, [0], axis=axis))).mean(axis=2)
    near = sum(np.roll(diff, k, axis=axis) for k in (-3, -2, -1, 1, 2, 3)) / 6
    def integral(a):
        return np.pad(a.cumsum(axis=0), ((1, 0), (0, 0)))
    # First dimension runs along the candidate edge in both orientations.
    if axis == 0:
        diff, near = diff.T, near.T
    return integral(diff), integral(near)


def seam_square(frame):
    """Find a square with four sharp, spatially localized edges; abstain on weak evidence."""
    vertical, horizontal = line_gradients(frame, 1), line_gradients(frame, 0)
    best_score, best_box, best_ratio, best_edge = 0.0, None, 0.0, 0.0
    for s in range(48, 161, 8):
        sides, ratios = [], []
        for (raw, control), transpose in ((vertical, False), (horizontal, True)):
            length = 224 - s - 7
            sums = raw[4+s:4+s+length] - raw[4:4+length]
            neighbors = control[4+s:4+s+length] - control[4:4+length]
            for start in (4, 4+s):
                edge = sums[:, start:start+length] / s
                ref = neighbors[:, start:start+length] / s
                ratio = (edge + 5) / (ref + 5)
                ratios.append(ratio.T if transpose else ratio)
                sides.append(edge.T if transpose else edge)
        score = np.minimum.reduce(ratios)
        y, x = np.unravel_index(score.argmax(), score.shape)
        if score[y, x] > best_score:
            best_score = float(score[y, x])
            best_ratio = float(np.mean([r[y, x] for r in ratios]))
            best_edge = float(np.mean([g[y, x] for g in sides]))
            best_box = (int(x)+4, int(y)+4, int(x)+4+s, int(y)+4+s)
    if best_score >= 2 and best_ratio >= 3 and best_edge >= 12:
        return best_box
    return None


def estimated_box(frame):
    _, box, bg = foreground(frame)
    if box is not None:
        return box, bg, "uniform_edge"
    box = seam_square(frame)
    return box, bg, "square_edge" if box is not None else "full"


def renderer_box(name, view):
    if view == "clean":
        return None
    if view == "small64":
        return (80, 80, 144, 144)
    image_id = int(name.removesuffix(".JPEG").rsplit("_", 1)[1])
    dx, dy = np.random.default_rng([42, image_id, 0]).integers(-32, 33, size=2)
    if dx == 0 and dy == 0:
        dx = 1
    side = {"small96_shift": 96, "small64_texture": 64}[view]
    x, y = int(112 - side//2 + dx), int(112 - side//2 + dy)
    return x, y, x + side, y + side


def box_mask(box):
    if box is None:
        return np.zeros((14, 14), dtype=bool)
    x, y, X, Y = box
    start = np.arange(0, 224, 16)
    width = np.maximum(0, np.minimum(start + 16, X) - np.maximum(start, x))
    height = np.maximum(0, np.minimum(start + 16, Y) - np.maximum(start, y))
    return (height[:, None] * width[None, :]) >= 32


def attention_mass(model, x, masks):
    """Measure probability of attending to the actual foreground in native and prior networks."""
    mask = torch.as_tensor(np.stack(masks), device=x.device, dtype=x.dtype).flatten(1)
    bias = -torch.log(mask.mean(1, keepdim=True).clamp_min(1 / 196)) * mask
    samples = {}
    def head_forward(head, layer, index, method, tokens):
        q, k, v = head.query(tokens), head.key(tokens), head.value(tokens)
        scores = (q @ k.transpose(-2, -1)) * head.scale
        if method == "late_area":
            scores[:, 0, 1:] += bias
        attention = scores.softmax(-1)
        samples[layer, index, method] = ((attention[:, 0, 1:] * mask).sum(-1)).tolist()
        return attention @ v
    for method in ("native", "late_area"):
        with ExitStack() as changes:
            for layer in (10, 11):
                for h, head in enumerate(model.blocks[layer-1].attention.heads):
                    fn = partial(head_forward, head, layer, h, method)
                    changes.enter_context(patch.object(head, "forward", fn))
            model(x)
    return samples


def target_square(row):
    """Smallest integer square containing the visible target rectangle within the 224px frame."""
    x, y = (int(np.floor(float(row[k]))) for k in ("bx1", "by1"))
    X, Y = (int(np.ceil(float(row[k]))) for k in ("bx2", "by2"))
    if X <= x or Y <= y:
        return None
    side = max(X - x, Y - y)
    left = min(224 - side, max(0, (x + X - side) // 2))
    top = min(224 - side, max(0, (y + Y - side) // 2))
    assert 0 <= left <= x < X <= left + side <= 224
    assert 0 <= top <= y < Y <= top + side <= 224
    return left, top, left + side, top + side


@torch.inference_mode()
def run_natural(download, out):
    """Diagnostic only: look up a ground-truth box using the known class of each query."""
    with (out / "natural_sizes.csv").open(encoding="utf-8", newline="") as stream:
        annotations = {(r["dataset"], r["image"]): r for r in csv.DictReader(stream)}
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_deit("imagenet", download=download).to(device)
    rows = []
    for dataset in ("imagenette", "imagewoof"):
        for names, targets, views in batches(dataset, download=download):
            frames = views["clean"]
            gt = [annotations[dataset, name] for name in names]
            boxes = [target_square(r) for r in gt]
            raw = tensor(frames, device)
            crops = [resize_box(f, box, (0, 0, 0)) for f, box in zip(frames, boxes)]
            masks = torch.from_numpy(np.stack([box_mask(b) for b in boxes])).to(device)
            scores = {"native": model(raw), "gt_square_late_area": forward_with_prior(model, raw, masks),
                      "gt_square_resize": model(tensor(crops, device))}
            labels = torch.tensor(targets, device=device)
            for method, logits in scores.items():
                assert torch.isfinite(logits).all()
                prob = logits.softmax(-1).gather(1, labels[:, None]).flatten().tolist()
                for name, label, box, r, top, p in zip(names, targets, boxes, gt, logits.argmax(-1).tolist(), prob):
                    assert label == int(r["target"])
                    rows.append(dict(dataset=dataset, image=name, target=label, method=method,
                                     top1=top, p_true=p, area_band=r["area_band"], visibility=r["visibility"],
                                     bbox=" ".join(map(str, box)) if box else ""))
    assert len(rows) == 2 * 500 * 3
    write_rows(out / "natural_geometry.csv", rows)


@torch.inference_mode()
def run(download, out):
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_deit("imagenet", download=download).to(device)
    rows, masses = [], []
    for dataset in ("imagenette", "imagewoof"):
        count = 0
        for names, targets, views in batches(dataset, download=download):
            for view, frames in views.items():
                auto = [estimated_box(f) for f in frames]
                oracle = [renderer_box(name, view) for name in names]
                if view == "small64_texture":
                    for i, (x, y, X, Y) in enumerate(oracle):
                        np.testing.assert_array_equal(frames[i, y:Y, x:X], views["small64"][i, 80:144, 80:144])
                if dataset == "imagewoof" and view == "small64":
                    ms = attention_mass(model, tensor(frames, device), [foreground(f)[0] for f in frames])
                    for (layer, h, method), values in ms.items():
                        for name, value in zip(names, values):
                            masses.append(dict(image=name, block=layer, head=h, method=method, fg_mass=value))
                raw = tensor(frames, device)
                labels = torch.tensor(targets, device=device)
                for finder, boxes in (("seam", [a[0] for a in auto]), ("oracle", oracle)):
                    resized = [resize_box(f, box, a[1]) for f, box, a in zip(frames, boxes, auto)]
                    mask = torch.from_numpy(np.stack([box_mask(box) for box in boxes])).to(device)
                    scores = {"resize": model(tensor(resized, device)), "late_area": forward_with_prior(model, raw, mask)}
                    for rule, logits in scores.items():
                        prob = logits.softmax(-1).gather(1, labels[:, None]).flatten().tolist()
                        for i, (top, p) in enumerate(zip(logits.argmax(-1).tolist(), prob)):
                            label = auto[i][2] if finder == "seam" else ("renderer" if boxes[i] else "full")
                            rows.append(dict(dataset=dataset, image=names[i], target=targets[i], view=view,
                                             method=f"{finder}_{rule}", top1=top, p_true=p, box_source=label,
                                             bbox=" ".join(map(str, boxes[i])) if boxes[i] is not None else ""))
            count += len(names)
            if count % 100 == 0:
                print(dataset, count, flush=True)
    assert len(rows) == 2 * 500 * 4 * 4 and len(masses) == 500 * 2 * 3 * 2
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "geometric_predictions.csv", rows)
    write_rows(out / "attention_mass.csv", masses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/scale_prior")
    parser.add_argument("--natural-only", action="store_true",
                        help="Use the target class's annotated box from natural_sizes.csv on clean images.")
    args = parser.parse_args()
    if args.natural_only:
        run_natural(args.download, args.output)
    else:
        run(args.download, args.output)
