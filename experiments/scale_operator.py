"""Predict held-out cross-scale CLS features of frozen DeiT models."""

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
from string import ascii_uppercase

for env_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[env_var] = "1"

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DONORS = "ABCDEFGIJK"
HELD_OUT = [i for i, letter in enumerate(ascii_uppercase) if letter not in DONORS]
HEIGHTS = (64, 80, 96, 112, 128)
TRAIN_PAIRS = ((64, 80), (80, 96), (64, 96))
TEST_PAIRS = ((64, 112), (64, 128), (80, 128))
RANK = 4
RIDGE = 0.01


@dataclass
class Operator:
    """Affine velocity v(x) = bias + ((x - origin) @ readout) @ basis.T."""

    origin: np.ndarray
    bias: np.ndarray
    readout: np.ndarray
    basis: np.ndarray


def predict(op, x, h1, h2):
    x = np.asarray(x, dtype=np.float64)
    velocity = op.bias + ((x - op.origin) @ op.readout) @ op.basis.T
    return x + np.log(h2 / h1) * velocity


def footprint_inputs(ink_mass, occupied_patches, class_ids, font, height, shifts):
    """Return input-scale foreground mass and number of occupied patches."""
    index = HEIGHTS.index(height)
    occupied = occupied_patches[class_ids, font, index, shifts]
    mass = np.broadcast_to(ink_mass[class_ids, font, index, None], occupied.shape)
    return np.stack([mass, occupied], axis=-1).astype(np.float64)


def fit_footprint(x, y, predictors, h1, h2):
    """Ridge regression of the change in CLS on two source-image descriptors."""
    predictors = np.asarray(predictors, dtype=np.float64).reshape(-1, 2)
    mean = predictors.mean(axis=0)
    std = predictors.std(axis=0)
    scale = np.where(std > 0, std, 1)
    design = (predictors - mean) / scale
    velocity = ((np.asarray(y, dtype=np.float64) - x) / np.log(h2 / h1)).reshape(len(design), -1)
    bias = velocity.mean(axis=0)
    penalty = max(RIDGE * np.square(design).sum(), np.finfo(float).tiny)
    weights = np.linalg.solve(design.T @ design + penalty * np.eye(2), design.T @ (velocity - bias))
    return mean, scale, bias, weights


def predict_footprint(fitted, x, predictors, h1, h2):
    mean, scale, bias, weights = fitted
    velocity = bias + ((predictors - mean) / scale) @ weights
    return x + np.log(h2 / h1) * velocity


def fit_operators(features, pairs=TRAIN_PAIRS):
    """Fit the supplied height pairs on 10 Times New Roman classes and 12 positions."""
    donor_ids = [ascii_uppercase.index(c) for c in DONORS]
    train = features[donor_ids, 0, :, :12].astype(np.float64)
    start = np.stack([train[:, HEIGHTS.index(a)] for a, _ in pairs])
    end = np.stack([train[:, HEIGHTS.index(b)] for _, b in pairs])
    step = np.array([np.log(b / a) for a, b in pairs])[:, None, None, None]
    velocity = (end - start) / step
    # Multiplying the velocity residual by step gives the activation residual.
    weights = np.broadcast_to(step, start.shape[:-1] + (1,)).reshape(-1, 1)
    dim = start.shape[-1]
    x = start.reshape(-1, dim)
    v = velocity.reshape(-1, dim)
    origin = np.average(x, axis=0, weights=weights[:, 0] ** 2)
    bias = np.average(v, axis=0, weights=weights[:, 0] ** 2)
    x = (x - origin) * weights
    y = (v - bias) * weights
    _, singular, vt = np.linalg.svd(y, full_matrices=False)
    threshold = np.finfo(float).eps * max(y.shape) * singular[0]
    basis = vt[:min(RANK, (singular > threshold).sum())].T
    # Restrict shear readout to directions orthogonal to every possible velocity.
    directions = np.column_stack((basis, bias))
    q, singular, _ = np.linalg.svd(directions, full_matrices=False)
    threshold = np.finfo(float).eps * max(directions.shape) * singular[0]
    q = q[:, singular > threshold]
    xs = x - (x @ q) @ q.T
    penalty = max(RIDGE * np.square(x).sum(), np.finfo(float).tiny)
    eye = np.eye(dim)
    ordinary = np.linalg.solve(x.T @ x + penalty * eye, x.T)
    shear = np.linalg.solve(xs.T @ xs + penalty * eye, xs.T)
    # Explicit projection also removes finite-precision leakage into q.
    shear -= q @ (q.T @ shear)
    empty = np.zeros((dim, 0))
    ops = {
        "identity": [Operator(origin, np.zeros(dim), empty, empty)],
        "drift": [Operator(origin, bias, empty, empty)],
        "shear4": [Operator(origin, bias, shear @ (y @ basis), basis)],
        "residual4": [Operator(origin, bias, ordinary @ (y @ basis), basis)],
        "ridge": [Operator(origin, bias, ordinary @ y, eye)],
        "shuffled_shear4": [],
    }
    # Each cyclic derangement preserves the donor mean, position and size pair.
    # Use the mean loss of nine fitted controls instead of ensembling predictions.
    for shift in range(1, len(DONORS)):
        shuffled = np.roll(velocity, shift, axis=1).reshape(-1, dim)
        coefficients = shear @ (((shuffled - bias) * weights) @ basis)
        ops["shuffled_shear4"].append(Operator(origin, bias, coefficients, basis))
    return ops


def squared_errors(x, y, ops, h1, h2):
    """Per-class SSE and SSE after subtracting the across-class mean at each shift."""
    losses = []
    centered_losses = []
    for op in ops:
        residual = predict(op, x, h1, h2) - y
        losses.append(np.square(residual).sum(axis=(1, 2)))
        centered = residual - residual.mean(axis=0, keepdims=True)
        centered_losses.append(np.square(centered).sum(axis=(1, 2)))
    return np.mean(losses, axis=0), np.mean(centered_losses, axis=0)


def skill(loss, identity_loss):
    return 1 - np.asarray(loss) / np.where(identity_loss > 0, identity_loss, np.nan)


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def calibration_control(model, values, ink_mass, occupied_patches):
    """Hold the evaluation images and fit budget fixed; change only the observed target height."""
    rows = []
    donor_ids = [ascii_uppercase.index(c) for c in DONORS]
    train = values[donor_ids, 0, :, :12]
    geom_train = footprint_inputs(ink_mass, occupied_patches, donor_ids, 0, 64, slice(0, 12))
    for fit_height in (96, 128):
        ops = fit_operators(values, pairs=((64, fit_height),))
        footprint = fit_footprint(train[:, 0], train[:, HEIGHTS.index(fit_height)],
                                  geom_train, 64, fit_height)
        for font_index, font in enumerate(("Times New Roman", "Arial")):
            test = values[HELD_OUT, font_index, :, 12:].astype(np.float64)
            x, y = test[:, HEIGHTS.index(64)], test[:, HEIGHTS.index(128)]
            id_loss, id_center = squared_errors(x, y, ops["identity"], 64, 128)
            geom_test = footprint_inputs(ink_mass, occupied_patches, HELD_OUT, font_index,
                                         64, slice(12, None))
            residual = predict_footprint(footprint, x, geom_test, 64, 128) - y
            centered_residual = residual - residual.mean(axis=0, keepdims=True)
            errors = {name: squared_errors(x, y, estimators, 64, 128)
                      for name, estimators in ops.items()}
            errors["footprint"] = (np.square(residual).sum(axis=(1, 2)),
                                   np.square(centered_residual).sum(axis=(1, 2)))
            for method, (loss, center) in errors.items():
                rows.append(dict(model=model, layer="LN", fit_h1=64, fit_h2=fit_height,
                                 n_fit_pairs=len(DONORS)*12, test_font=font, test_h1=64,
                                 test_h2=128, n_test_pairs=len(HELD_OUT)*12, method=method,
                                 skill=skill(loss.sum(), id_loss.sum()),
                                 centered_skill=skill(center.sum(), id_center.sum()),
                                 letters_improved=(loss < id_loss * (1-1e-10)).sum()))
    return rows


def run(cache, output):
    output.mkdir(parents=True, exist_ok=True)
    all_results, per_class, controls, calibration = [], [], [], []
    for model in ("initial", "big_00", "big_10"):
        with np.load(cache / f"{model}.npz", allow_pickle=False) as data:
            features = data["features"]
            assert features.shape == (13, 26, 2, 5, 24, 192)
            assert tuple(data["heights"]) == HEIGHTS
            assert len(np.unique(data["shifts"], axis=0)) == 24
            assert "".join(data["letters"].tolist()) == ascii_uppercase
            layers = data["layers"].astype(str)
            ink_mass = data["ink_mass"]
            occupied_patches = data["occupied_patches"]
        for layer, values in zip(layers, features):
            operators = fit_operators(values)
            if layer == "LN":
                calibration.extend(calibration_control(model, values, ink_mass, occupied_patches))
                # Factor class transfer and size extrapolation, using new probe positions throughout.
                donors = [ascii_uppercase.index(letter) for letter in DONORS]
                for group, ids in (("fit_classes", donors), ("held_out_classes", HELD_OUT)):
                    subset = values[ids, 0, :, 12:].astype(np.float64)
                    for h2 in (96, 128):
                        x, y = subset[:, 0], subset[:, HEIGHTS.index(h2)]
                        id_loss, id_center = squared_errors(x, y, operators["identity"], 64, h2)
                        for name, estimators in operators.items():
                            loss, center = squared_errors(x, y, estimators, 64, h2)
                            controls.append(dict(model=model, layer=layer, letter_group=group,
                                                 n_letters=len(ids), font="Times New Roman", h1=64, h2=h2,
                                                 method=name, skill=skill(loss.sum(), id_loss.sum()),
                                                 centered_skill=skill(center.sum(), id_center.sum())))
            for font_index, font in enumerate(("Times New Roman", "Arial")):
                test = values[HELD_OUT, font_index, :, 12:].astype(np.float64)
                for h1, h2 in TEST_PAIRS:
                    x, y = test[:, HEIGHTS.index(h1)], test[:, HEIGHTS.index(h2)]
                    id_loss, id_center = squared_errors(x, y, operators["identity"], h1, h2)
                    context = dict(model=model, layer=layer, font=font, h1=h1, h2=h2)
                    class_rows = [dict(**context, letter=ascii_uppercase[i], identity_sse=d,
                                       identity_centered_sse=c)
                                  for i, d, c in zip(HELD_OUT, id_loss, id_center)]
                    for name, estimators in operators.items():
                        loss, center = squared_errors(x, y, estimators, h1, h2)
                        per_letter = skill(loss, id_loss)
                        all_results.append(dict(**context, method=name, n_letters=len(x),
                                                shifts_per_letter=x.shape[1],
                                                skill=skill(loss.sum(), id_loss.sum()),
                                                centered_skill=skill(center.sum(), id_center.sum()),
                                                median_letter_skill=np.median(per_letter),
                                                letters_improved=(per_letter > 1e-10).sum()))
                        for row, s, c in zip(class_rows, per_letter, skill(center, id_center)):
                            row[name + "_skill"] = s
                            row[name + "_centered_skill"] = c
                    per_class.extend(class_rows)
        print(f"Scored {model}", flush=True)
    write_rows(output / "operator_scores.csv", all_results)
    write_rows(output / "operator_by_letter.csv", per_class)
    write_rows(output / "operator_controls.csv", controls)
    write_rows(output / "operator_calibration.csv", calibration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache" / "scale_operator")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "scale_operator")
    args = parser.parse_args()
    run(args.cache, args.output)
