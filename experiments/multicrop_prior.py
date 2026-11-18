"""Compare output averaging and a late-CLS prior on six views of one input frame."""

import argparse

import numpy as np
from PIL import Image
import torch

from attention_grafts import collect, graft
from imagenet_deit import ROOT, load_deit, normalize_image
from prior_data import batches
from scale_operator import write_rows


# Top-left corners of 128-pixel windows on the 224-pixel input canvas.
WINDOWS = ((0, 0), (96, 0), (0, 96), (96, 96), (48, 48))


def crop_views(frames):
    windows = []
    for left, top in WINDOWS:
        tiles = frames[:, top:top + 128, left:left + 128]
        windows.append(np.stack([np.asarray(Image.fromarray(tile).resize(
            (224, 224), Image.Resampling.BICUBIC)) for tile in tiles]))
    return np.stack(windows)


def to_tensor(frames, device):
    x = torch.from_numpy(frames).permute(0, 3, 1, 2).to(device, torch.float32) / 255
    return normalize_image(x)


@torch.inference_mode()
def classify(model, frames, device):
    logits, inputs, base_heads = collect(model, to_tensor(frames, device))
    native = torch.softmax(logits, dim=-1)
    probabilities, head_outputs = [], []
    windows = crop_views(frames)
    for pair in range(0, len(WINDOWS), 2):
        group = windows[pair:pair + 2]
        x = to_tensor(group.reshape(-1, 224, 224, 3), device)
        crop_logits, _, features = collect(model, x)
        probabilities.append(crop_logits.softmax(dim=-1).reshape(len(group), len(frames), -1))
        heads = torch.stack([features[11, i] for i in range(3)], dim=1)
        head_outputs.append(heads.reshape(len(group), len(frames), 3, 64))
    probabilities = torch.cat(probabilities)
    head_outputs = torch.cat(head_outputs)
    entropy = -torch.special.xlogy(probabilities, probabilities).sum(dim=-1)
    assert torch.isfinite(entropy).all()
    selected = entropy.argmin(dim=0)
    images = torch.arange(len(frames), device=device)
    selected_heads = head_outputs[selected, images]
    mixed_heads = {h: .5 * (base_heads[11, h] + selected_heads[:, h]) for h in range(3)}
    mixed_logits = graft(model, 11, inputs[11], mixed_heads)
    predictions = {"native": native, "uniform_six": (native + probabilities.sum(0)) / 6,
                   "entropy_mix": .5 * (native + probabilities[selected, images]),
                   "head_mix_b11": mixed_logits.softmax(dim=-1)}
    return predictions, selected.tolist()


def run(download):
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_deit("imagenet", download=download).to(device)
    rows = []
    for dataset in ("imagenette", "imagewoof"):
        for index, (names, targets, views) in enumerate(batches(dataset, download)):
            labels = torch.tensor(targets, device=device)
            for view, frames in views.items():
                scores, selected = classify(model, frames, device)
                for method, probabilities in scores.items():
                    top = probabilities.argmax(dim=-1).tolist()
                    p_true = probabilities.gather(1, labels[:, None]).squeeze(1).tolist()
                    for i, name in enumerate(names):
                        rows.append(dict(dataset=dataset, image=name, target=targets[i], view=view,
                                         method=method, selected_window=selected[i], top1=top[i],
                                         p_true=p_true[i]))
            if (index + 1) % 10 == 0:
                print(dataset, (index + 1) * len(names), "photos", flush=True)
    assert len(rows) == 2 * 500 * 4 * 4
    output = ROOT / "results/scale_prior"
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "multicrop_predictions.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    run(parser.parse_args().download)
