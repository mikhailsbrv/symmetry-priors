"""Paired scale interventions on the original ImageNet validation images in Imagenette."""

import argparse
import io
import tarfile
from pathlib import PurePosixPath

import numpy as np
from PIL import Image
import torch

from imagenet_deit import ROOT, load_deit, normalize_image
from transfer_objects import checked_file
from scale_operator import write_rows

ARCHIVE = ROOT / ".cache/imagenette2-160.tgz"
URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
SHA256 = "64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5"
# Original ImageNet-1k class indices, in the order of Imagenette's WordNet IDs.
CLASSES = {"n01440764": 0, "n02102040": 217, "n02979186": 482, "n03000684": 491,
           "n03028079": 497, "n03394916": 566, "n03417042": 569, "n03425413": 571,
           "n03445777": 574, "n03888257": 701}
DEPTHS = (0, 3, 6, 9, 10, 11, 12)
CONDITIONS = ("reference224", "small64_black", "small64_mean", "from64_to128", "from64_to224")


def validation_images(download):
    archive = checked_file(ARCHIVE, SHA256, URL, download)
    groups = {wnid: [] for wnid in CLASSES}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            path = PurePosixPath(member.name)
            if member.isfile() and path.match("*/ILSVRC2012_val_*.JPEG") and path.parent.name in groups:
                with tar.extractfile(member) as stream:
                    groups[path.parent.name].append((path.name, stream.read()))
    for items in groups.values():
        items.sort()
        assert len(items) == 50
    assert len({name for items in groups.values() for name, _ in items}) == 500
    return list(zip(*groups.values()))


def render(payload):
    # Same central square in all five conditions, including the RGB context in the crop.
    with Image.open(io.BytesIO(payload)) as source:
        source = source.convert("RGB")
        side = min(source.size)
        x, y = (source.width - side) // 2, (source.height - side) // 2
        crop = source.crop((x, y, x + side, y + side))
    low = crop.resize((64, 64), Image.Resampling.BICUBIC)
    def paste(tile, fill=0):
        canvas = Image.new("RGB", (224, 224), fill)
        offset = (224 - tile.width) // 2
        canvas.paste(tile, (offset, offset))
        return canvas
    frames = [crop.resize((224, 224), Image.Resampling.BICUBIC), paste(low),
              paste(low, (124, 116, 104)), paste(low.resize((128, 128), Image.Resampling.BICUBIC)),
              low.resize((224, 224), Image.Resampling.BICUBIC)]
    return np.stack([np.asarray(frame) for frame in frames])


def collect(model, pixels):
    states = {}
    # Inputs to the first residual block and outputs of subsequent blocks.
    hooks = [model.blocks[0].register_forward_pre_hook(lambda mod, inp: states.update({0: inp[0]}))]
    for depth in DEPTHS[1:]:
        hooks.append(model.blocks[depth - 1].register_forward_hook(
            lambda mod, inp, out, d=depth: states.update({d: out})))
    try:
        logits = model(pixels)
    finally:
        for hook in hooks:
            hook.remove()
    return logits, states


def continue_from(model, depth, cls, patches):
    tokens = torch.cat((cls[:, :1], patches[:, 1:]), dim=1)
    for block in list(model.blocks)[depth:]:
        tokens = block(tokens)
    return model.classifier(model.norm(tokens[:, 0]))


def predictions(logits, labels):
    prob = logits.softmax(-1)
    return zip(logits.argmax(-1).tolist(), prob.gather(1, labels[:, None]).flatten().tolist())


@torch.inference_mode()
def run(download, output):
    torch.set_num_threads(2)
    batches = validation_images(download)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_deit("imagenet", download=download).to(device)
    labels = torch.tensor(list(CLASSES.values()), device=device)
    readout, swaps = [], []
    for batch_id, batch in enumerate(batches):
        names = [name for name, _ in batch]
        # Each batch contains one image from each class. Rotate the donor by 1–9 classes.
        donor = np.roll(np.arange(len(batch)), 1 + batch_id % (len(batch) - 1))
        assert np.all(donor != np.arange(len(batch)))
        donor_idx = torch.tensor(donor, device=device)
        pixels = np.stack([render(payload) for _, payload in batch])
        logits_by_view, captures = {}, {}
        for i, condition in enumerate(CONDITIONS):
            x = torch.from_numpy(pixels[:, i].copy()).permute(0, 3, 1, 2).to(device, torch.float32) / 255
            x = normalize_image(x)
            if condition in ("small64_black", "from64_to224"):
                logits, captures[condition] = collect(model, x)
            else:
                logits = model(x)
            logits_by_view[condition] = logits
            for name, label, (top, p) in zip(names, CLASSES.values(), predictions(logits, labels)):
                readout.append(dict(image=name, target=label, view=condition, top1=top, p_true=p))
        small, large = captures["small64_black"], captures["from64_to224"]
        for depth in DEPTHS:
            conditions = {"same_cls": (large[depth], small[depth]),
                          "same_patches": (small[depth], large[depth]),
                          "other_cls": (large[depth][donor_idx], small[depth]),
                          "other_patches": (small[depth], large[depth][donor_idx])}
            for part, (cls, patches) in conditions.items():
                logits = continue_from(model, depth, cls, patches)
                if depth in (0, 12):
                    # At depth 0 all CLS tokens match. After B12 only CLS affects the head.
                    uses_large = (depth == 0 and part.endswith("patches")) or (depth == 12 and part.endswith("cls"))
                    expected = logits_by_view["from64_to224" if uses_large else "small64_black"]
                    if uses_large and part.startswith("other"):
                        expected = expected[donor_idx]
                    torch.testing.assert_close(logits, expected, rtol=1e-4, atol=3e-5)
                for j, (top, p) in enumerate(predictions(logits, labels)):
                    source = donor[j] if part.startswith("other") else j
                    swaps.append(dict(image=names[j], target=int(labels[j]), depth=depth, graft=part,
                                      donor_image=names[source], donor_target=int(labels[source]), top1=top, p_true=p))
        if (batch_id + 1) % 10 == 0:
            print(f"{(batch_id + 1)*10}/500 photos", flush=True)
    assert len(readout) == 500 * len(CONDITIONS) and len(swaps) == 500 * len(DEPTHS) * 4
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "imagenette_predictions.csv", readout)
    write_rows(output / "imagenette_swaps.csv", swaps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch verified weights and data if missing")
    args = parser.parse_args()
    run(args.download, ROOT / "results/object_scale")
