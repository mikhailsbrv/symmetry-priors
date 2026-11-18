"""Test a foreground-area prior for the CLS attention row of a frozen DeiT-Tiny."""

import argparse
from contextlib import ExitStack
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from imagenet_deit import ROOT, load_deit, normalize_image
from scale_operator import write_rows


def foreground(frame):
    """A uniform image boundary is evidence for a padded frame. All distances are RGB L-infinity."""
    edge = np.concatenate((frame[0], frame[-1], frame[1:-1, 0], frame[1:-1, -1]))
    bg = np.median(edge, axis=0)
    edge_near = np.max(np.abs(edge - bg), axis=-1) <= 12
    mask = np.zeros((14, 14), dtype=bool)
    if edge_near.mean() < 0.9:
        return mask, None, bg
    different = np.max(np.abs(frame - bg), axis=-1) > 12
    occupied = different.reshape(14, 16, 14, 16).sum((1, 3)) >= 32
    if not 0 < occupied.mean() <= 0.6:
        return mask, None, bg
    active_pixels = different & occupied.repeat(16, 0).repeat(16, 1)
    ys, xs = np.where(active_pixels)
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    if (box[2] - box[0]) * (box[3] - box[1]) > 0.8 * 224**2:
        return mask, None, bg
    return occupied, box, bg


def resize_box(frame, box, bg):
    if box is None:
        return frame
    crop = Image.fromarray(frame).crop(box)
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side), tuple(np.rint(bg).astype(int)))
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    return np.asarray(canvas.resize((224, 224), Image.Resampling.BICUBIC))


def biased_head(head, bias, x):
    q, k, v = head.query(x), head.key(x), head.value(x)
    scores = (q @ k.transpose(-2, -1)) * head.scale
    scores[:, 0, 1:] += bias
    return scores.softmax(dim=-1) @ v


def forward_with_prior(model, images, mask, blocks=(10, 11)):
    """One model forward. images are ImageNet-normalized; mask is [batch, 14, 14]."""
    mask = mask.reshape(-1, 196).to(device=images.device, dtype=images.dtype)
    # The clamp only defines empty-mask rows, for which the final bias is zero.
    bias = -torch.log(mask.mean(1, keepdim=True).clamp_min(1 / 196)) * mask
    with ExitStack() as changes:
        for number in blocks:
            for head in model.blocks[number - 1].attention.heads:
                changes.enter_context(patch.object(head, "forward", partial(biased_head, head, bias)))
        return model(images)


def tensor(frames, device):
    values = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).to(device, torch.float32)
    return normalize_image(values / 255)


def verify(model, device):
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    frame[80:144, 80:144] = (230, 0, 0)
    mask, box, _ = foreground(frame)
    assert mask.sum() == 16 and box == (80, 80, 144, 144)
    gray = np.full_like(frame, 30)
    assert foreground(gray)[1] is None
    x = tensor([frame, gray], device)
    empty = torch.zeros((2, 14, 14), device=device)
    torch.testing.assert_close(forward_with_prior(model, x, empty), model(x), rtol=0, atol=0)
    head = model.blocks[9].attention.heads[0]
    tokens = torch.linspace(-1, 1, 2 * 197 * 192, device=device).reshape(2, 197, 192)
    bias = torch.from_numpy(np.stack([mask, mask])).to(device, torch.float32).reshape(2, -1)
    native, altered = head(tokens), biased_head(head, bias, tokens)
    assert torch.equal(native[:, 1:], altered[:, 1:]) and not torch.equal(native[:, 0], altered[:, 0])


def evaluate(model, frames, device):
    geom = [foreground(frame) for frame in frames]
    masks, boxes, backgrounds = zip(*geom)
    x = tensor(frames, device)
    base = model(x)
    scores = {"native": base}
    active = any(box is not None for box in boxes)
    masks = torch.from_numpy(np.stack(masks)).to(device)
    for name, blocks, mask in (("late_area", (10, 11), masks),
                               ("early_area", (2, 3), masks),
                               ("late_shifted_mask", (10, 11), torch.roll(masks, (7, 7), (1, 2)))):
        scores[name] = forward_with_prior(model, x, mask, blocks) if active else base
    resized = [resize_box(frame, box, bg) for frame, box, bg in zip(frames, boxes, backgrounds)]
    scores["bbox_resize"] = model(tensor(resized, device)) if active else base
    return scores, geom


@torch.inference_mode()
def run(datasets, download, output):
    from prior_data import batches

    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_deit("imagenet", download=download).to(device).eval()
    verify(model, device)
    rows = []
    for dataset in datasets:
        count = 0
        for names, targets, views in batches(dataset, download=download):
            labels = torch.as_tensor(targets, device=device)
            for view, frames in views.items():
                scores, geom = evaluate(model, frames, device)
                for method, logits in scores.items():
                    assert torch.isfinite(logits).all()
                    prob = logits.softmax(-1).gather(1, labels[:, None]).flatten().tolist()
                    for name, label, top, p_true, (mask, box, _) in zip(
                            names, targets, logits.argmax(-1).tolist(), prob, geom):
                        rows.append(dict(dataset=dataset, image=name, target=int(label), view=view,
                                         method=method, top1=top, p_true=p_true,
                                         active_patches=int(mask.sum()),
                                         bbox=" ".join(map(str, box)) if box is not None else ""))
            count += len(names)
            if count % 100 == 0:
                print(dataset, count, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_rows(output, rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("imagenette", "imagewoof"),
                        default=["imagenette", "imagewoof"])
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results/scale_prior/prior_predictions.csv")
    args = parser.parse_args()
    run(args.datasets, args.download, args.output)
