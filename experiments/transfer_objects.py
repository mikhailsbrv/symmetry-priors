"""Apply a scale operator learned on printed letters to real COIL-20 and MNIST images."""

import argparse
import gzip
import json
import urllib.request
import zipfile
from pathlib import Path

from scale_operator import (DONORS, fit_operators, fit_footprint, footprint_inputs, predict_footprint,
                            squared_errors, skill, write_rows)
from scale_features import HEIGHTS, ROOT, DeiTTiny, digest, place
import numpy as np
from PIL import Image
import torch

CACHE = ROOT / ".cache" / "scale_operator"
RESULTS = ROOT / "results" / "scale_operator"
COIL_URL = "https://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
MNIST_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
COIL_SHA = "517c5594820eb40066ba0ff6842e7f09392bf7fde849bf9cb9c28445b0f29e88"
MNIST_FILES = {
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}


def checked_file(path, sha, url, download):
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} is missing. Pass --download to obtain {url}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, temp)
        if digest(temp) != sha:
            raise ValueError(f"Checksum mismatch for {url}")
        temp.replace(path)
    if digest(path) != sha:
        raise ValueError(f"Checksum mismatch for {path}")
    return path


def external_images(download):
    """The full square source frame at both sizes preserves shape and camera angle."""
    coil, labels = [], []
    with zipfile.ZipFile(checked_file(CACHE / "coil-20-proc.zip", COIL_SHA, COIL_URL, download)) as z:
        for obj in range(1, 21):
            for view in range(0, 72, 9):
                with z.open(f"coil-20-proc/obj{obj}__{view}.png") as file:
                    image = Image.open(file).convert("L")
                assert image.size == (128, 128)
                coil.append(image)
                labels.append(obj)
    yield "COIL-20", coil, np.array(labels)

    arrays = []
    for name, sha in MNIST_FILES.items():
        path = checked_file(CACHE / "mnist" / name, sha, MNIST_URL + name, download)
        arrays.append(gzip.decompress(path.read_bytes()))
    pixels, targets = arrays
    assert tuple(np.frombuffer(pixels[:16], dtype=">u4")) == (2051, 10000, 28, 28)
    assert tuple(np.frombuffer(targets[:8], dtype=">u4")) == (2049, 10000)
    images = np.frombuffer(pixels, dtype=np.uint8, offset=16).reshape(10000, 28, 28)
    digits = np.frombuffer(targets, dtype=np.uint8, offset=8)
    rng = np.random.default_rng(17011)
    ids = np.concatenate([rng.choice(np.flatnonzero(digits == c), 20, replace=False) for c in range(10)])
    yield "MNIST", [Image.fromarray(images[i]) for i in ids], digits[ids]


@torch.inference_mode()
def extract(model, images, shifts, device):
    captured, chunks, footprints = [], [], []
    handles = [layer.register_forward_hook(lambda _, args, out: captured.append(out[:, 0]))
               for layer in [*model.blocks, model.norm]]
    try:
        for start in range(0, len(images), 4):
            batch = []
            for image in images[start:start + 4]:
                for side in (64, 128):
                    tile = np.asarray(image.resize((side, side), Image.Resampling.LANCZOS))
                    canvases = place(tile, shifts, 224)
                    batch.append(canvases)
                    if side == 64:
                        ink = canvases.sum(axis=(1, 2)) / 255.
                        patches = canvases.reshape(len(shifts), 14, 16, 14, 16).any(axis=(2, 4)).sum(axis=(1, 2))
                        footprints.append(np.stack((ink, patches), axis=-1))
            pixels = torch.from_numpy(np.concatenate(batch)).to(device, torch.float32).div_(255)
            captured.clear()
            model(pixels[:, None].expand(-1, 3, -1, -1))
            chunks.append(torch.stack(captured).cpu().numpy())
    finally:
        for handle in handles:
            handle.remove()
    levels = np.concatenate(chunks, axis=1).reshape(13, len(images), 2, len(shifts), 192)
    return levels, np.stack(footprints)


def score_layer(model, layer, dataset, features, groups, operators, footprint_prediction=None):
    """Center across source images at each matched position. Preserve image and object groups."""
    x, y = features[:, 0], features[:, 1]
    errors = {name: squared_errors(x, y, fits, 64, 128) for name, fits in operators.items()}
    if footprint_prediction is not None:
        residual = footprint_prediction - y
        centered = residual - residual.mean(axis=0, keepdims=True)
        errors["footprint"] = (np.square(residual).sum(axis=(1, 2)), np.square(centered).sum(axis=(1, 2)))
    identity, identity_center = errors["identity"]
    drift, _ = errors["drift"]
    rows = []
    subsets = {"all": np.ones(len(groups), dtype=bool)}
    if layer == "LN":
        subsets.update({str(c): groups == c for c in np.unique(groups)})
    for method, (loss, center) in errors.items():
        for group, mask in subsets.items():
            base = identity[mask].sum()
            rows.append(dict(model=model, layer=layer, dataset=dataset, group=group,
                             method=method, n_images=mask.sum(), n_positions=x.shape[1],
                             identity_sse=base, sse=loss[mask].sum(),
                             identity_centered_sse=identity_center[mask].sum(), centered_sse=center[mask].sum(),
                             skill=skill(loss[mask].sum(), base),
                             centered_skill=skill(center[mask].sum(), identity_center[mask].sum()),
                             gain_over_drift=(drift[mask].sum() - loss[mask].sum()) / base))
    return rows


def main(download, calibration_only):
    torch.set_num_threads(2)
    datasets = list(external_images(download))
    rows, calibration_rows = [], []
    for name in ("initial", "big_00", "big_10"):
        with np.load(CACHE / f"{name}.npz", allow_pickle=False) as cache:
            letters = cache["features"]
            shifts = cache["shifts"][12:16]
            assert len(shifts) == 4 and np.array_equal(cache["heights"], [64, 80, 96, 112, 128])
            info = json.loads(str(cache["provenance"].item()))
            donor_ids = [cache["letters"].tolist().index(c) for c in DONORS]
            donor_geometry = footprint_inputs(cache["ink_mass"], cache["occupied_patches"],
                                              donor_ids, 0, 64, slice(0, 12))
        assert info["model_sha256"] == digest(ROOT / "model.py") and info["seed"] == 42
        torch.manual_seed(42)
        model = DeiTTiny(num_classes=26)
        if name != "initial":
            ckpt = ROOT / "results" / "letters_scale_val100" / f"{name}.pt"
            assert info["checkpoint_sha256"] == digest(ckpt)
            model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"])
        device = torch.device("cuda")
        model.to(device).eval()
        operators = [] if calibration_only else [fit_operators(layer) for layer in letters]
        calibration = {}
        train = letters[-1, donor_ids, 0, :, :12]
        for end in (96, 128):
            fit = fit_operators(letters[-1], pairs=((64, end),))
            footprint = fit_footprint(train[:, 0], train[:, HEIGHTS.index(end)], donor_geometry, 64, end)
            calibration[end] = (fit, footprint)
        del letters
        public_features = {"shifts": shifts, "heights": np.array([64, 128])}
        for dataset, images, groups in datasets:
            levels, geometry = extract(model, images, shifts, device)
            assert np.isfinite(levels).all()
            if name == "big_10":
                prefix = "coil" if dataset == "COIL-20" else "mnist"
                public_features[prefix + "_features"] = levels[-1].copy()
                public_features[prefix + "_geometry"] = geometry
                public_features[prefix + "_groups"] = groups
            for idx, ops in enumerate(operators):
                layer = f"B{idx+1}" if idx < 12 else "LN"
                rows.extend(score_layer(name, layer, dataset, levels[idx], groups, ops))
            for end, (ops, footprint) in calibration.items():
                geometric_prediction = predict_footprint(footprint, levels[-1, :, 0], geometry, 64, 128)
                scores = score_layer(name, "LN", dataset, levels[-1], groups, ops, geometric_prediction)
                for row in scores:
                    calibration_rows.append(dict(fit_h1=64, fit_h2=end, n_fit_pairs=120,
                                                 test_h1=64, test_h2=128, **row))
            print(name, dataset, len(images), "images scored", flush=True)
        if name == "big_10":
            feature_path = ROOT / "data" / "scale_operator" / "external_big_10_ln.npz"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(feature_path, **public_features)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if rows:
        write_rows(RESULTS / "external.csv", rows)
    write_rows(RESULTS / "external_calibration.csv", calibration_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download missing public evaluation data")
    parser.add_argument("--calibration-only", action="store_true", help="Rerun only the matched calibration fits for all three backbones")
    args = parser.parse_args()
    main(args.download, args.calibration_only)
