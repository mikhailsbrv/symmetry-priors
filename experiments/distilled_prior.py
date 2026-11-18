"""Distill a cross-scale correction into one CLS attention readout of frozen DeiT."""

import argparse
from collections import defaultdict
import hashlib
import tarfile
from pathlib import PurePosixPath

import numpy as np
import torch
from torch.nn import functional as F

from imagenet_deit import ROOT, WEIGHT_SHA256, load_deit, normalize_image
from photo_swaps import ARCHIVE, CLASSES, SHA256, URL, render
from prior_data import IMAGEWOOF, WOOF_SHA, WOOF_URL, batches, render_views
from scale_operator import write_rows
from transfer_objects import checked_file


DEPTHS = (3, 11)
TRAIN_PER_CLASS = 75
RIDGE = 0.1
LOCAL_ADAPTER = ROOT / ".cache/scale_prior/imagenette_b11.pt"


def training_set(download, dataset="imagenette"):
    if dataset == "imagenette":
        classes, path, sha, url = CLASSES, ARCHIVE, SHA256, URL
    else:
        assert dataset == "imagewoof"
        classes = IMAGEWOOF
        path, sha, url = ROOT / ".cache/imagewoof2-160.tgz", WOOF_SHA, WOOF_URL
    source = checked_file(path, sha, url, download)
    groups = defaultdict(list)
    with tarfile.open(source, "r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            wnid = path.parent.name
            if (wnid in classes and member.isfile() and path.parent.parent.name == "train"
                    and path.name.startswith(wnid + "_") and path.suffix == ".JPEG"):
                with archive.extractfile(member) as stream:
                    groups[wnid].append((path.name, stream.read()))
    # Stable selection without consulting predictions. Original validation images use a different prefix.
    chosen = []
    for wnid in classes:
        group = sorted(groups[wnid], key=lambda entry: hashlib.sha256(entry[0].encode()).digest())
        assert len(group) >= TRAIN_PER_CLASS
        chosen.append(group[:TRAIN_PER_CLASS])
    return list(zip(*chosen))


def pixels(batch, device):
    return torch.from_numpy(batch).permute(0, 3, 1, 2).to(device, torch.float32) / 255


def patch_weights(x):
    # Per-patch RGB gradient energy. Unit sum, including constant images.
    dx = F.pad((x[..., 1:] - x[..., :-1]).abs(), (0, 1))
    dy = F.pad((x[:, :, 1:] - x[:, :, :-1]).abs(), (0, 0, 0, 1))
    energy = F.avg_pool2d((dx + dy).mean(1, keepdim=True), 16).flatten(1) + 1e-6
    return energy / energy.sum(1, keepdim=True)


def forward(model, x, depths, correction=None):
    """Capture training pairs, or insert the fitted correction within a single forward."""
    weights = patch_weights(x)
    hooks, features, head_cls, summaries = [], {}, {}, {}

    def save_patches(depth):
        def read(_, args, tokens):
            summaries[depth] = (weights[..., None] * tokens[:, 1:]).sum(dim=1)
        return read

    def process_cls(depth):
        def replace(_, args):
            tokens = args[0]
            h = tokens[:, 0]
            design = torch.cat((h, summaries[depth]), dim=1)
            features[depth], head_cls[depth] = design, h
            if correction is None:
                return None
            mean_x, scale_x, mean_y, coef = correction
            delta = mean_y if coef is None else mean_y + ((design - mean_x) / scale_x) @ coef
            modified = tokens.clone()
            modified[:, 0] = h + delta
            return (modified,)
        return replace

    assert correction is None or len(depths) == 1
    for d in depths:
        block = model.blocks[d - 1]
        hooks.append(block.norm1.register_forward_hook(save_patches(d)))
        hooks.append(block.attention.projection.register_forward_pre_hook(process_cls(d)))
    try:
        logits = model(normalize_image(x))
    finally:
        for handle in hooks:
            handle.remove()
    return logits, features, head_cls


def fit(model, device, examples, depths):
    designs, responses = defaultdict(list), defaultdict(list)
    for batch_id, batch in enumerate(examples):
        variants = [render_views(name, payload) for name, payload in batch]
        teacher = np.stack([render(payload)[4] for _, payload in batch])  # Same 64×64 information.
        _, _, target = forward(model, pixels(teacher, device), depths)
        for name in ("clean", "small64", "small64_texture"):
            x = pixels(np.stack([v[name] for v in variants]), device)
            _, design, h = forward(model, x, depths)
            for d in depths:
                delta = torch.zeros_like(h[d]) if name == "clean" else target[d] - h[d]
                designs[d].append(design[d].cpu().numpy())
                responses[d].append(delta.cpu().numpy())
        if (batch_id + 1) % 25 == 0:
            print(f"Fit pairs from {(batch_id+1)*10} training photos", flush=True)
    fitted = {}
    for d in depths:
        x, y = (np.concatenate(values[d]).astype(np.float64) for values in (designs, responses))
        assert x.shape == (3 * TRAIN_PER_CLASS * len(CLASSES), 384) and y.shape == (len(x), 192)
        mean_x, mean_y, scale_x = x.mean(axis=0), y.mean(axis=0), x.std(axis=0).clip(1e-6)
        x = (x - mean_x) / scale_x
        coef = np.linalg.solve(x.T @ x + len(x) * RIDGE * np.eye(x.shape[1]), x.T @ (y - mean_y))
        fitted[d] = tuple(torch.as_tensor(a, device=device, dtype=torch.float32)
                          for a in (mean_x, scale_x, mean_y, coef))
    return fitted, {name for batch in examples for name, _ in batch}


def load_adapter(device, path=LOCAL_ADAPTER):
    """Load coefficients saved by the primary fit, for the original ImageNet DeiT."""
    state = torch.load(path, map_location=device, weights_only=True)
    if state["base_sha256"] != WEIGHT_SHA256 or state["block"] != 11:
        raise ValueError("Unexpected backbone or block in scale adapter")
    return tuple(state["coefficients"])


@torch.inference_mode()
def predict_with_adapter(model, rgb, adapter):
    """Return 1000 logits in one pass; rgb is float NCHW in [0,1] on the model device."""
    return forward(model, rgb, (11,), adapter)[0]


@torch.inference_mode()
def run(download, crossover):
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_deit("imagenet", download=download).to(device)
    original = training_set(download)
    variants = {}
    if crossover:
        dogs = training_set(download, "imagewoof")
        train_names = set()
        for split in (0, 1):
            # Each fit uses five classes per domain, 75 distinct training images per class.
            joint = [list(a[split::2]) + list(b[split::2]) for a, b in zip(original, dogs)]
            fitted, used = fit(model, device, joint, (11,))
            assert train_names.isdisjoint(used)
            train_names.update(used)
            variants[f"split_{split}"] = (11, fitted[11])
    else:
        fitted, train_names = fit(model, device, original, DEPTHS)
        LOCAL_ADAPTER.parent.mkdir(parents=True, exist_ok=True)
        temporary = LOCAL_ADAPTER.with_suffix(".part")
        torch.save(dict(base_sha256=WEIGHT_SHA256, block=11, ridge=RIDGE,
                        coefficients=[v.cpu() for v in fitted[11]]), temporary)
        temporary.replace(LOCAL_ADAPTER)
        fitted[11] = load_adapter(device)
        variants = {"native": (None, None), "mean_b11": (11, (*fitted[11][:3], None)),
                    "ridge_b3": (3, fitted[3]), "ridge_b11": (11, fitted[11])}
    rows = []
    for dataset in ("imagenette", "imagewoof"):
        for number, (names, targets, views) in enumerate(batches(dataset, download)):
            assert train_names.isdisjoint(names)
            label = torch.tensor(targets, device=device)
            for view, frames in views.items():
                x = pixels(frames, device)
                for method, (d, coeff) in variants.items():
                    if d is None:
                        logits = model(normalize_image(x))
                    else:
                        logits = forward(model, x, [d], coeff)[0]
                    if not torch.isfinite(logits).all():
                        raise ValueError("Non-finite logits")
                    probabilities = logits.softmax(dim=-1)
                    probs = probabilities.gather(1, label[:, None]).flatten().tolist()
                    # Evaluator-only diagnostic: candidates are the five labels from the true
                    # half of this domain. A uniform preference for that half cancels here.
                    candidate_labels = torch.stack([label[i % 2::2] for i in range(len(names))])
                    restricted = logits.gather(1, candidate_labels)
                    within = candidate_labels.gather(1, restricted.argmax(-1, keepdim=True)).flatten().tolist()
                    group_mass = probabilities.gather(1, candidate_labels).sum(1).tolist()
                    top1 = logits.argmax(-1).tolist()
                    for image, y, p, prediction, top5set, mass in zip(names, targets, probs, top1, within, group_mass):
                        row = dict(dataset=dataset, image=image, target=y, view=view, method=method,
                                   top1=prediction, p_true=p, within_group_top1=top5set, p_group=mass)
                        if crossover:
                            row["fit_saw_class"] = int(targets.index(y) % 2 == int(method[-1]))
                        rows.append(row)
            if (number + 1) % 25 == 0:
                print(f"{dataset}: {(number+1)*10}/500 photos", flush=True)
    directory = ROOT / "results/scale_prior"
    directory.mkdir(parents=True, exist_ok=True)
    write_rows(directory / ("crossover_predictions.csv" if crossover else "distilled_predictions.csv"), rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch missing verified datasets and weights")
    parser.add_argument("--class-crossover", action="store_true", help="Cross two disjoint sets of 5+5 fit classes")
    args = parser.parse_args()
    run(args.download, args.class_crossover)
