"""Render matched glyphs and collect frozen DeiT CLS features for the scale-transfer probe."""

import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, __version__ as pillow_version
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from model import DeiTTiny

HEIGHTS = (64, 80, 96, 112, 128)
SHIFT_SEED = 17112026


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def templates(letters, fonts):
    glyphs = {}
    for font_id, path in enumerate(fonts):
        font = ImageFont.truetype(str(path), 1024)
        for letter_id, letter in enumerate(letters):
            left, top, right, bottom = font.getbbox(letter)
            glyph = Image.new("L", (right - left, bottom - top))
            ImageDraw.Draw(glyph).text((-left, -top), letter, font=font, fill=255)
            glyph = glyph.crop(glyph.getbbox())
            for scale_id, height in enumerate(HEIGHTS):
                width = round(glyph.width * height / glyph.height)
                resized = glyph.resize((width, height), Image.Resampling.LANCZOS)
                glyphs[letter_id, font_id, scale_id] = np.asarray(resized)
    return glyphs


def matched_shifts(glyphs, pools, canvas):
    max_width = max(im.shape[1] for im in glyphs.values())
    xmax = min(22, (canvas - max_width) // 2)
    ymax = min(40, (canvas - max(HEIGHTS)) // 2)
    safe = {(x, y) for x in range(-xmax, xmax + 1) for y in range(-ymax, ymax + 1)}
    small_train, large_train = [set(map(tuple, p["train"].tolist())) for p in pools]
    both_unseen = sorted(safe - small_train - large_train)
    small_unseen = sorted((safe - small_train) & large_train)
    rng = np.random.default_rng(SHIFT_SEED)
    take = min(len(both_unseen), 24)
    common = [both_unseen[i] for i in rng.choice(len(both_unseen), take, replace=False)]
    common += [small_unseen[i] for i in rng.choice(len(small_unseen), 24 - take, replace=False)]
    rng.shuffle(common)
    return np.asarray(common, dtype=int)


def place(glyph, shifts, canvas):
    height, width = glyph.shape
    images = np.zeros((len(shifts), canvas, canvas), dtype=np.uint8)
    for image, (dx, dy) in zip(images, shifts):
        left = (canvas - width) // 2 + dx
        top = (canvas - height) // 2 + dy
        assert 0 <= left <= canvas - width and 0 <= top <= canvas - height
        image[top:top + height, left:left + width] = glyph
    return images


@torch.inference_mode()
def extract(checkpoint: Path | None, device, output: Path, dataset_path: Path):
    """Cache activations [layer, letter, font, height, position, feature] as float32."""
    torch.manual_seed(42)
    torch.set_num_threads(2)
    data = torch.load(dataset_path, map_location="cpu", weights_only=True)
    letters, canvas = data["letters"], data["canvas"]
    font_dir = Path(os.environ.get("SCALE_FONT_DIR", "C:/Windows/Fonts"))
    fonts = [font_dir / "times.ttf", font_dir / "arial.ttf"]
    glyphs = templates(letters, fonts)
    shifts = matched_shifts(glyphs, data["shift_pools"], canvas)
    for letter_id in range(len(letters)):
        for height_id, height in enumerate(data["heights"]):
            centered = place(glyphs[letter_id, 0, HEIGHTS.index(height)], [(0, 0)], canvas)[0]
            np.testing.assert_array_equal(centered, data["images"][letter_id, height_id, 0].numpy())

    model = DeiTTiny(num_classes=len(letters))
    info = {"seed": 42, "checkpoint": None}
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert state["letters"] == letters and state["val_target_met"]
        model.load_state_dict(state["state_dict"])
        info.update(checkpoint=checkpoint.name, checkpoint_sha256=digest(checkpoint),
                    epoch=state["epoch"], donors=state["donors"])
        del state
    model.to(device).eval()
    locations = [*model.blocks, model.norm]
    captured = []
    hooks = [module.register_forward_hook(
        lambda _, args, out: captured.append(out[:, 0].detach())) for module in locations]
    shape = (len(letters), len(fonts), len(HEIGHTS), len(shifts))
    features = np.empty((len(locations), *shape, 192), dtype=np.float32)
    logits = np.empty((*shape, len(letters)), dtype=np.float32)
    ink_mass = np.empty(shape[:3], dtype=np.float32)
    occupied = np.empty(shape, dtype=np.int16)
    try:
        for (c, f, h), glyph in glyphs.items():
            batch = place(glyph, shifts, canvas)
            ink_mass[c, f, h] = glyph.sum() / 255
            tiles = batch.reshape(len(shifts), canvas // 16, 16, canvas // 16, 16)
            occupied[c, f, h] = tiles.any(axis=(2, 4)).sum(axis=(1, 2))
            images = torch.from_numpy(batch).to(device=device, dtype=torch.float32) / 255
            captured.clear()
            logits[c, f, h] = model(images[:, None].expand(-1, 3, -1, -1)).cpu().numpy()
            features[:, c, f, h] = torch.stack(captured).cpu().numpy()
    finally:
        for hook in hooks:
            hook.remove()
    assert np.isfinite(features).all() and np.isfinite(logits).all()
    overlap = {}
    for height, pool in zip(data["heights"], data["shift_pools"]):
        train_shifts = set(map(tuple, pool["train"].tolist()))
        overlap[str(height)] = [sum(tuple(p) in train_shifts for p in half)
                                for half in np.split(shifts, 2)]
    info.update(font_sha256={p.name: digest(p) for p in fonts}, dataset_sha256=digest(dataset_path),
                model_sha256=digest(ROOT / "model.py"), shift_seed=SHIFT_SEED,
                position_overlap_fit_test_with_backbone_train=overlap,
                device=str(device), torch_version=str(torch.__version__), pillow_version=pillow_version)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, features=features, logits=logits, ink_mass=ink_mass, occupied_patches=occupied,
                 letters=list(letters), fonts=["Times New Roman", "Arial"], heights=HEIGHTS, shifts=shifts,
                 layers=[f"B{i}" for i in range(1, 13)] + ["LN"], provenance=json.dumps(info))
    temporary.replace(output)
    print(output.name, features.shape, "train position overlap (fit/test)", overlap, flush=True)


def classification_by_height(cache_dir, output):
    """Top-1 on the 12 held-out positions. All networks use big_10's donor partition."""
    rows = []
    with np.load(cache_dir / "big_10.npz", allow_pickle=False) as source:
        donors = list(json.loads(source["provenance"].item())["donors"])
    for run in ("initial", "big_00", "big_10"):
        with np.load(cache_dir / f"{run}.npz", allow_pickle=False) as data:
            letters = data["letters"]
            pred = data["logits"][:, :, :, 12:].argmax(-1)
            correct = pred == np.arange(len(letters))[:, None, None, None]
            donor_mask = np.isin(letters, donors)
            for font_id, font in enumerate(data["fonts"]):
                for group, mask in [("donors", donor_mask), ("held_out", ~donor_mask)]:
                    for height_id, height in enumerate(data["heights"]):
                        subset = correct[mask, font_id, height_id]
                        rows.append(dict(model=run, font=font, group=group, height=height,
                                         correct=int(subset.sum()), count=subset.size, accuracy=subset.mean()))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_final_layer(cache_path, output):
    """Publish final CLS and logits so the primary probe can be refit without a checkpoint."""
    with np.load(cache_path, allow_pickle=False) as cache:
        assert cache["layers"][-1] == "LN"
        assert json.loads(cache["provenance"].item())["checkpoint"] == "big_10.pt"
        arrays = {k: cache[k] for k in ("logits", "heights", "shifts", "letters", "fonts", "provenance",
                                         "ink_mass", "occupied_patches")}
        arrays["features"] = cache["features"][-1].copy()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


if __name__ == "__main__":
    runs = sys.argv[1:] or ["initial", "big_00", "big_10"]
    cache_dir = ROOT / ".cache" / "scale_operator"
    for name in runs:
        ckpt = None if name == "initial" else ROOT / "results" / "letters_scale_val100" / f"{name}.pt"
        extract(ckpt, "cuda", cache_dir / f"{name}.npz", ROOT / "data" / "letters.pt")
        if name == "big_10":
            export_final_layer(cache_dir / "big_10.npz", ROOT / "data/scale_operator/big_10_ln.npz")
    if all((cache_dir / f"{run}.npz").exists() for run in ("initial", "big_00", "big_10")):
        classification_by_height(cache_dir, ROOT / "results/scale_operator/classification_by_height.csv")
