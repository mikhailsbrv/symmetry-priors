"""Plot the practical priors, late-head interventions and class crossover from their CSVs."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_figure(directory=Path(__file__).resolve().parents[1] / "results/scale_prior"):
    directory = Path(directory)

    def read(name):
        with (directory / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    prior, distilled, geometric = [read(f"{s}_predictions.csv") for s in ("prior", "distilled", "geometric")]
    methods = ("native", "late_area", "seam_late_area", "ridge_b11", "bbox_resize", "seam_resize")
    views = ("clean", "small64", "small96_shift", "small64_texture")
    counts = np.zeros((len(views), len(methods)))
    for j, (method, data) in enumerate(zip(methods, (prior, prior, geometric, distilled, prior, geometric))):
        for i, view in enumerate(views):
            subset = [r for r in data if (r["dataset"], r["method"], r["view"]) == ("imagewoof", method, view)]
            assert len({r["image"] for r in subset}) == len(subset) == 500
            counts[i, j] = 100 * np.mean([r["top1"] == r["target"] for r in subset])

    # Paired photograph bootstrap, stratified by the ten fixed Imagenette categories.
    specificity = read("specificity_predictions.csv")
    changes, intervals = [], []
    for condition in ("scale", "blur", "contrast"):
        items = [r for r in specificity if r["condition"] == condition]
        base = {r["image"]: (r["target"], int(r["top1"] == r["target"])) for r in items
                if r["source"] == "baseline"}
        graft = {r["image"]: int(r["top1"] == r["target"]) for r in items
                 if (r["block"], r["head"], r["source"]) == ("11", "all", "same")}
        assert base.keys() == graft.keys() and len(graft) == 350
        labels = sorted({label for label, _ in base.values()})
        delta = np.array([[graft[im] - base[im][1] for im in sorted(base) if base[im][0] == label]
                          for label in labels])
        assert delta.shape == (10, 35)
        rng = np.random.default_rng(42)
        boot = delta[np.arange(10)[:, None, None], rng.integers(0, 35, (10, 10000, 35))].mean((0, 2))
        changes.append(100 * delta.mean())
        intervals.append(100 * np.quantile(boot, [.025, .975]))
    changes, intervals = np.array(changes), np.array(intervals)

    crossover = read("crossover_predictions.csv")
    cross = []
    for dataset in ("imagenette", "imagewoof"):
        small = [r for r in crossover if r["dataset"] == dataset and r["view"] == "small64"]
        original = [r for r in prior if (r["dataset"], r["view"], r["method"]) == (dataset, "small64", "native")]
        groups = [original, *([r for r in small if r["fit_saw_class"] == s] for s in ("0", "1"))]
        photos = {r["image"] for r in original}
        assert len(photos) == 500 and all(len(group) == 500 and {r["image"] for r in group} == photos
                                         for group in groups)
        cross.append([100 * np.mean([r["top1"] == r["target"] for r in group]) for group in groups])

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(14.6, 4.1), gridspec_kw={"width_ratios": [1.88, .98, 1.05]})
    fig.subplots_adjust(left=.087, right=.993, top=.72, bottom=.26, wspace=.5)
    navy, teal, copper = "#264b70", "#147466", "#ac582b"
    titles = [("А   Приор на новых классах", "Imagewoof · top-1, %"),
              ("Б   Подстановка в CLS", "Imagenette · прирост top-1, п.п."),
              ("В   Перенос между классами", "Растр 64 px · top-1, %")]
    for ax, (title, subtitle) in zip((a, b, c), titles):
        ax.text(0, 1.30, title, transform=ax.transAxes, fontsize=12, fontweight="bold", ha="left")
        ax.text(0, 1.13, subtitle, transform=ax.transAxes, fontsize=9.2, color="#48505a", ha="left")
        ax.tick_params(length=0, labelsize=8.8)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.spines["bottom"].set_color("#c7cbcf")

    a.imshow(counts, cmap="Blues", vmin=0, vmax=83, aspect="auto")
    a.set_yticks(range(4), ["Исходный\nкадр", "64 px\nчёрный фон", "96 px\nсо сдвигом", "64 px\nтекстура"])
    a.set_xticks(range(6), ["DeiT", "Внимание\nграница", "Внимание\n+ шов", "Ridge\nB11", "Увелич.\nграница", "Увелич.\n+ шов"])
    a.tick_params(axis="x", pad=6)
    a.spines["bottom"].set_visible(False)
    for (i, j), value in np.ndenumerate(counts):
        a.text(j, i, f"{value:.1f}".replace(".", ","), ha="center", va="center", fontsize=10.4,
               color="white" if value > 45 else "#132d46")

    y = np.arange(3)
    b.axvline(0, color="#b8bec5", linewidth=.9)
    b.errorbar(changes, y, xerr=np.stack([changes - intervals[:, 0], intervals[:, 1] - changes]),
               fmt="o", color=teal, ecolor=teal, capsize=4, ms=6)
    for yi, estimate in enumerate(changes):
        b.text(estimate, yi - .18, f"+{estimate:.1f}".replace(".", ","), ha="center", va="bottom",
               fontsize=9, color=teal)
    b.set(yticks=y, yticklabels=["Масштаб", "Размытие", "Контраст"], ylim=(2.55, -.55),
          xlim=(-.5, 24), xticks=(0, 5, 10, 15, 20))
    b.grid(axis="x", color="#e7e9eb")
    b.set_axisbelow(True)
    b.set_xlabel("Парный 95% интервал", fontsize=9)

    x = np.arange(2)
    for j, (label, color) in enumerate([("Исходная DeiT", navy), ("Класс вне подгонки", "#98a5b5"),
                                        ("Класс в подгонке", copper)]):
        bars = c.bar(x + (j - 1) * .24, np.array(cross)[:, j], width=.22, label=label, color=color)
        c.bar_label(bars, labels=[f"{v:.1f}".replace(".", ",") for v in np.array(cross)[:, j]],
                    fontsize=8.4, padding=2)
    c.set(xticks=x, xticklabels=["Imagenette", "Imagewoof"], ylim=(0, 46), yticks=[0, 10, 20, 30, 40])
    c.grid(axis="y", color="#e5e8eb")
    c.set_axisbelow(True)
    c.legend(loc="upper left", bbox_to_anchor=(-.02, -.12), frameon=False, fontsize=8.5, handlelength=1.1)

    fig.savefig(directory / "comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    return fig


if __name__ == "__main__":
    plt.close(make_figure())
