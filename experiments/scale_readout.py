"""Does reversing the fitted change of scale restore the frozen letter classifier?"""
from string import ascii_uppercase

from scale_operator import DONORS, HEIGHTS, ROOT, fit_operators, predict, write_rows
from scale_features import place, templates
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from model import DeiTTiny


def canonicalize(frame):
    """Measure the visible ink, resize it to height 64 and preserve its centre."""
    image = Image.fromarray(frame)
    left, top, right, bottom = image.getbbox()
    width = max(1, round((right - left) * 64 / (bottom - top)))
    glyph = image.crop((left, top, right, bottom)).resize((width, 64), Image.Resampling.LANCZOS)
    result = Image.new("L", image.size)
    result.paste(glyph, ((left + right - width) // 2, (top + bottom - 64) // 2))
    return np.asarray(result)


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    font_dir = Path(os.getenv("SCALE_FONT_DIR", "C:/Windows/Fonts"))
    glyphs = templates(ascii_uppercase, [font_dir / "times.ttf", font_dir / "arial.ttf"])
    results = []
    for name in ("big_00", "big_10"):
        with np.load(ROOT / ".cache" / "scale_operator" / f"{name}.npz") as data:
            features, logits, shifts = data["features"][-1], data["logits"], data["shifts"][12:]
        operators = fit_operators(features)
        checkpoint = ROOT / "results" / "letters_scale_val100" / f"{name}.pt"
        model = DeiTTiny(num_classes=26).to(device).eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"])
        weight = model.classifier.weight.cpu().numpy().astype(np.float64)
        bias = model.classifier.bias.cpu().numpy()
        for f, font in enumerate(("Times New Roman", "Arial")):
            for height in (64, 128):
                h = HEIGHTS.index(height)
                z = features[:, f, h, 12:]
                original = logits[:, f, h, 12:]
                np.testing.assert_allclose(z @ weight.T + bias, original, atol=2e-5, rtol=2e-5)
                corrected = {}
                for method in ("drift", "shear4"):
                    small = predict(operators[method][0], z, height, 64)
                    corrected[method] = small @ weight.T + bias
                pixel_logits = []
                for c in range(26):
                    frames = place(glyphs[c, f, h], shifts, 224)
                    batch = np.stack([canonicalize(frame) for frame in frames])
                    x = torch.from_numpy(batch).to(device, torch.float32) / 255
                    pixel_logits.append(model(x[:, None].expand(-1, 3, -1, -1)).cpu().numpy())
                for method, scores in {"original": original, **corrected,
                                       "bbox_to_64": np.stack(pixel_logits)}.items():
                    for c, letter in enumerate(ascii_uppercase):
                        other = np.delete(scores[c], c, axis=-1).max(axis=-1)
                        margin = scores[c, :, c] - other
                        results.append(dict(model=name, font=font, height=height, method=method,
                                            letter=letter, probe_fit=letter in DONORS, n=len(shifts),
                                            correct=int((scores[c].argmax(-1) == c).sum()),
                                            mean_margin=float(margin.mean())))
    directory = ROOT / "results" / "scale_operator"
    directory.mkdir(parents=True, exist_ok=True)
    write_rows(directory / "readout.csv", results)


if __name__ == "__main__":
    run()
