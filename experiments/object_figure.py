"""Render the COIL-20 and Imagenette scale experiments from their published CSV files."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def make_figure(directory=Path(__file__).resolve().parents[1] / "results/object_scale"):
    directory = Path(directory)
    with (directory / "scores.csv").open(encoding="utf-8", newline="") as file:
        coil = {(r["condition"], r["descriptor"], r["layer"]): 100 * int(r["correct"]) / int(r["total"])
                for r in csv.DictReader(file) if (r["gallery_size"], r["query_size"]) == ("64", "128")}

    views = ["reference224", "small64_black", "small64_mean", "from64_to128", "from64_to224"]
    with (directory / "imagenette_predictions.csv").open(encoding="utf-8", newline="") as file:
        predictions = {(r["view"], r["image"]): r["top1"] == r["target"] for r in csv.DictReader(file)}
    photos = sorted(image for view, image in predictions if view == "reference224")
    assert len(photos) == 500 and len(predictions) == len(views) * len(photos)
    accuracy = {v: 100 * np.mean([predictions[v, image] for image in photos]) for v in views}
    with (directory / "imagenette_swaps.csv").open(encoding="utf-8", newline="") as file:
        swaps = {(r["graft"], int(r["depth"]), r["image"]): r["top1"] == r["target"]
                 for r in csv.DictReader(file) if r["graft"] in ("same_cls", "same_patches")}

    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(13.6, 4.7),
                                              gridspec_kw={"width_ratios": [1.15, 0.94, 1.13]})
    fig.subplots_adjust(left=.055, right=.99, bottom=.30, top=.83, wspace=.28)
    blue, green, ochre, gray = "#285b9a", "#09786a", "#a15013", "#666b71"
    titles = [("А   COIL-20", "Узнавание предметов при 64 → 128 px"),
              ("Б   Imagenette", "Точность среди 1000 классов"),
              ("В   Подстановка токенов", "Фото 64 px получает токены от версии 64 → 224")]
    for ax, (title, subtitle) in zip((left, middle, right), titles):
        ax.text(0, 1.195, title, transform=ax.transAxes, fontsize=14, weight="bold", ha="left")
        ax.text(0, 1.075, subtitle, transform=ax.transAxes, fontsize=10, ha="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#aeb3b8")
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, color="#aeb3b8")

    layers = ("B1", "B3", "B6", "B9", "B12", "LN")
    tracks = [("imagenet", "cls", "CLS", blue, "o"), ("imagenet", "aligned", "Патчи по рамке", green, "s"),
              ("random", "cls", "CLS, случайные веса", gray, None),
              ("imagenet", "fixed", "Патчи в окне 64", ochre, "^")]
    for network, descriptor, label, color, marker in tracks:
        values = [coil[network, descriptor, layer] for layer in layers]
        left.plot(range(len(layers)), values, label=label, color=color, marker=marker, ms=4.5,
                  lw=1.8, ls="--" if network == "random" else "-")
    left.set(xticks=range(len(layers)), xticklabels=layers, ylim=(0, 100), xlabel="Слой",
             ylabel="Точность, %", yticks=range(0, 101, 20))
    left.grid(axis="y", color="#e2e5e7")
    left.legend(loc="upper left", bbox_to_anchor=(-.03, -.24), ncol=2, frameon=False,
                fontsize=8.8, columnspacing=.9, handlelength=1.65)

    names = ("224 из исходника", "64, чёрный фон", "64, средний фон", "64 → 128", "64 → 224")
    colors = ("#9aaba8", "#aebcca", "#aebcca", "#68a194", green)
    for y, (view, name, color) in enumerate(zip(views, names, colors)):
        value = accuracy[view]
        middle.barh(y + .13, value, color=color, height=.35)
        middle.text(0, y - .20, name, va="center", fontsize=9.2)
        middle.text(value + 1.3, y + .13, f"{value:.1f}".replace(".", ","), va="center", fontsize=9)
    middle.set(xlim=(0, 84), ylim=(4.52, -.50), xticks=[0, 20, 40, 60, 80], yticks=[],
               xlabel="Правильный top-1, %")
    middle.spines["left"].set_visible(False)
    middle.grid(axis="x", color="#e2e5e7")

    depths = sorted({depth for graft, depth, image in swaps})
    right.axhline(accuracy["small64_black"], color=gray, linestyle=":", linewidth=1.4, label="Исходные 64 px")
    right.axhline(accuracy["from64_to224"], color=gray, linestyle="--", linewidth=1.4, label="Версия 64 → 224")
    for graft, label, color, marker in [("same_cls", "Только CLS", blue, "o"),
                                        ("same_patches", "Только патчи", green, "s")]:
        values = [100 * np.mean([swaps[graft, depth, image] for image in photos]) for depth in depths]
        right.plot(depths, values, color=color, marker=marker, lw=2, ms=5, label=label)
    right.set(xticks=depths, ylim=(0, 70), yticks=[0, 20, 40, 60], xlabel="После скольких блоков",
              ylabel="Правильный top-1, %")
    right.grid(axis="y", color="#e2e5e7")
    handles, labels = right.get_legend_handles_labels()
    order = [2, 3, 0, 1]
    right.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper left",
                 bbox_to_anchor=(-.03, -.24), ncol=2, frameon=False, fontsize=8.8, columnspacing=1.0,
                 handlelength=1.8)
    fig.savefig(directory / "object_scale.png", dpi=140, bbox_inches="tight", facecolor="white")
    return fig


if __name__ == "__main__":
    plt.close(make_figure())
