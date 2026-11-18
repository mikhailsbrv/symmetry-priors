"""Test late-head interventions under scale, blur and contrast at calibrated accuracies."""

import argparse
from collections import Counter

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch

from attention_grafts import BLOCKS, collect, graft
from imagenet_deit import ROOT, load_deit, normalize_image
from photo_swaps import CLASSES, CONDITIONS, render, validation_images
from scale_operator import write_rows

CALIBRATION_PER_CLASS = 15
# Ordered from mild to severe. On a calibration tie choose the milder treatment.
GRID = {"blur": (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48),
        "contrast": (1, .8, .6, .4, .3, .2, .15, .1, .075, .05, .035, .025, .015, .008, 0)}


def extract_views(batch):
    indices = [CONDITIONS.index(name) for name in ("small64_black", "from64_to224")]
    frames = np.stack([render(payload)[indices] for _, payload in batch])
    return frames[:, 0], frames[:, 1]


def tensor(frames, device):
    x = torch.from_numpy(frames).to(device, torch.float32).permute(0, 3, 1, 2) / 255
    return normalize_image(x)


def degraded(canonical, condition, severity):
    frames = []
    for rgb in canonical:
        image = Image.fromarray(rgb)
        if condition == "blur":
            image = image.filter(ImageFilter.GaussianBlur(severity))
        elif condition == "contrast":
            image = ImageEnhance.Contrast(image).enhance(severity)
        else:
            raise ValueError(condition)
        frames.append(np.asarray(image))
    return np.stack(frames)


def calibrate(batches, model, labels):
    counts = Counter()
    for batch in batches:
        small, canonical = extract_views(batch)
        counts["scale", 64] += (model(tensor(small, labels.device)).argmax(1) == labels).sum().item()
        for condition, values in GRID.items():
            for severity in values:
                frames = degraded(canonical, condition, severity)
                hits = (model(tensor(frames, labels.device)).argmax(1) == labels).sum().item()
                counts[condition, severity] += hits
    selected = {"scale": 64}
    target_hits = counts["scale", 64]
    for condition, values in GRID.items():
        selected[condition] = min(values, key=lambda v: abs(counts[condition, v] - target_hits))
    total = len(batches) * len(labels)
    rows = [dict(condition=c, severity=v, correct=n, total=total, selected=int(selected[c] == v))
            for (c, v), n in counts.items()]
    print(f"Calibration target {target_hits}/{total}; selected {selected}", flush=True)
    return selected, rows


def score_rows(logits, names, labels, condition, severity, block, head, source, donor_indices):
    targets = labels.tolist()
    probs = logits.softmax(1).gather(1, labels[:, None]).flatten().tolist()
    return [dict(image=name, target=targets[i], condition=condition, severity=severity,
                 block=block, head=head, source=source, donor_image=names[j], donor_target=targets[j],
                 top1=pred, p_true=p)
            for i, (name, j, pred, p) in enumerate(zip(names, donor_indices, logits.argmax(1).tolist(), probs))]


@torch.inference_mode()
def run(download, output):
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_deit("imagenet", download=download).to(device)
    batches = validation_images(download)
    labels = torch.tensor(list(CLASSES.values()), device=device)
    selected, calibration = calibrate(batches[:CALIBRATION_PER_CLASS], model, labels)
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "specificity_calibration.csv", calibration)
    results = []
    same = np.arange(len(labels))
    for rank, batch in enumerate(batches[CALIBRATION_PER_CLASS:], CALIBRATION_PER_CLASS):
        names = [name for name, _ in batch]
        small, canonical = extract_views(batch)
        clean, _, heads = collect(model, tensor(canonical, device))
        results.extend(score_rows(clean, names, labels, "canonical", 0, 0, "none", "baseline", same))
        other = np.roll(same, 1 + rank % (len(same) - 1))
        for condition, value in selected.items():
            frames = small if condition == "scale" else degraded(canonical, condition, value)
            base, inputs, _ = collect(model, tensor(frames, device))
            results.extend(score_rows(base, names, labels, condition, value, 0, "none", "baseline", same))
            for block in BLOCKS:
                for source, order in (("same", same), ("other", other)):
                    donors = {h: heads[block, h][torch.as_tensor(order, device=device)] for h in range(3)}
                    for head in (("0", "1", "2", "all") if source == "same" else ("all",)):
                        values = donors if head == "all" else {int(head): donors[int(head)]}
                        scores = graft(model, block, inputs[block], values)
                        results.extend(score_rows(scores, names, labels, condition, value, block,
                                                  head, source, order))
        if (rank + 1) % 10 == 0:
            print(f"Evaluation {rank + 1 - CALIBRATION_PER_CLASS}/{len(batches) - CALIBRATION_PER_CLASS}",
                  flush=True)
    write_rows(output / "specificity_predictions.csv", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    run(parser.parse_args().download, ROOT / "results/scale_prior")
