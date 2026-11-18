"""Recognize held-out COIL-20 views with cosine prototypes across image sizes."""

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SIZES = (64, 96, 128)
LAYERS = ("B1", "B3", "B6", "B9", "B12", "LN")
CONDITIONS = ("pixels", "random", "imagenet")


def unit(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    if not np.isfinite(norm).all() or np.any(norm == 0):
        raise ValueError("Invalid descriptor for cosine readout")
    return x / norm


def recognize(features, spatial_mean=False):
    """Counts indexed by gallery size, query size and object; two complementary folds."""
    assert features.shape[:4] == (20, 24, 3, 4)
    hits = np.zeros((3, 3, 20), dtype=int)
    ties = np.zeros_like(hits)
    truth = np.repeat(np.arange(20), 24)
    for fold in (0, 1):
        # One in two of the sampled angles, so each fold tests 12 views per object.
        gallery = np.asarray(features[:, fold::2, :, :2], dtype=np.float64)
        queries = np.asarray(features[:, 1 - fold::2, :, 2:], dtype=np.float64)
        if spatial_mean:
            # Same sixteen sampled tokens as for the flattened readout, without spatial order.
            assert gallery.shape[-1] == 16 * 192
            gallery = gallery.reshape(*gallery.shape[:-1], 16, 192).mean(axis=-2)
            queries = queries.reshape(*queries.shape[:-1], 16, 192).mean(axis=-2)
        prototypes = unit(unit(gallery).mean(axis=(1, 3)))
        for j in range(3):
            queries_j = unit(queries[:, :, j].reshape(480, -1))
            for i in range(3):
                similarity = queries_j @ prototypes[:, i].T
                # NumPy resolves an exact tie in favor of the lowest object index.
                prediction = similarity.argmax(axis=-1)
                tied = (similarity == similarity.max(axis=-1, keepdims=True)).sum(axis=-1) > 1
                hits[i, j] += (prediction == truth).reshape(20, 24).sum(axis=1)
                ties[i, j] += tied.reshape(20, 24).sum(axis=1)
    return hits, ties


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def contrasts(results):
    """Paired object bootstrap, conditional on the observed gallery, views and competitors."""
    samples = np.random.default_rng(42).integers(0, 20, size=(10000, 20))
    rows = []

    def add(condition, layer, comparison, delta):
        object_gain = delta / 48
        lo, hi = np.quantile(object_gain[samples].mean(axis=1), [0.025, 0.975])
        rows.append(dict(condition=condition, layer=layer, gallery_size=64, query_size=128,
                         comparison=comparison, gain=object_gain.mean(), lower_95=lo, upper_95=hi,
                         objects_better=int((delta > 0).sum()), objects_tied=int((delta == 0).sum()),
                         objects_worse=int((delta < 0).sum())))

    for condition in CONDITIONS:
        for layer in (("input",) if condition == "pixels" else LAYERS):
            aligned = results[condition, layer, "aligned"][0, 2]
            fixed = results[condition, layer, "fixed"][0, 2]
            add(condition, layer, "aligned_minus_fixed", aligned - fixed)
            if condition != "pixels":
                cls = results[condition, layer, "cls"][0, 2]
                add(condition, layer, "aligned_minus_cls", aligned - cls)
                aligned_mean = results[condition, layer, "aligned_mean"][0, 2]
                fixed_mean = results[condition, layer, "fixed_mean"][0, 2]
                add(condition, layer, "aligned_mean_minus_fixed_mean", aligned_mean - fixed_mean)
                add(condition, layer, "aligned_minus_aligned_mean", aligned - aligned_mean)
    for layer in LAYERS:
        for descriptor in ("cls", "fixed", "aligned"):
            diff = results["imagenet", layer, descriptor][0, 2] - results["random", layer, descriptor][0, 2]
            add("imagenet_minus_random", layer, descriptor, diff)
        gain = {m: results[m, layer, "aligned"][0, 2] - results[m, layer, "fixed"][0, 2]
                for m in ("imagenet", "random")}
        add("imagenet_minus_random", layer, "aligned_minus_fixed", gain["imagenet"] - gain["random"])
    return rows


def run(cache, output):
    scores, by_object, results = [], [], {}
    for condition in CONDITIONS:
        directory = cache / condition
        marker = directory / "ready.json"
        if not marker.is_file():
            raise FileNotFoundError(f"Incomplete feature cache at {directory}; run object_tokens.py first")
        layout = json.loads(marker.read_text(encoding="utf-8"))
        assert layout["condition"] == condition and tuple(layout["sides"]) == SIZES
        assert layout["views"] == list(range(0, 72, 3)) and layout["objects"] == list(range(1, 21))
        assert layout["shifts"] == [[-8, -8], [8, 8], [-8, 8], [8, -8]]
        layers = ("input",) if condition == "pixels" else LAYERS
        descriptors = (("fixed", "aligned") if condition == "pixels" else
                       ("cls", "fixed", "aligned", "fixed_mean", "aligned_mean"))
        for layer in layers:
            for descriptor in descriptors:
                window = descriptor.removesuffix("_mean")
                filename = window if condition == "pixels" else f"{layer}_{window}"
                features = np.load(directory / f"{filename}.npy", mmap_mode="r", allow_pickle=False)
                hits, ties = recognize(features, spatial_mean=descriptor.endswith("_mean"))
                results[condition, layer, descriptor] = hits
                for i, gallery_size in enumerate(SIZES):
                    for j, query_size in enumerate(SIZES):
                        key = dict(condition=condition, layer=layer, descriptor=descriptor,
                                   gallery_size=gallery_size, query_size=query_size)
                        scores.append(dict(**key, correct=hits[i, j].sum(), total=960,
                                           accuracy=hits[i, j].sum()/960, top_ties=ties[i, j].sum()))
                        for obj, count in enumerate(hits[i, j]):
                            by_object.append(dict(**key, object_id=obj + 1, correct=count, total=48,
                                                  top_ties=ties[i, j, obj]))
            # At size 64 the two windows coincide on both gallery and query.
            np.testing.assert_array_equal(results[condition, layer, "aligned"][0, 0],
                                          results[condition, layer, "fixed"][0, 0])
            if condition != "pixels":
                np.testing.assert_array_equal(results[condition, layer, "aligned_mean"][0, 0],
                                              results[condition, layer, "fixed_mean"][0, 0])
        print(f"{condition}: scored {len(layers)} layer(s)", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "scores.csv", scores)
    write_rows(output / "by_object.csv", by_object)
    write_rows(output / "contrasts.csv", contrasts(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/object_tokens")
    parser.add_argument("--output", type=Path, default=ROOT / "results/object_scale")
    args = parser.parse_args()
    run(args.cache, args.output)
