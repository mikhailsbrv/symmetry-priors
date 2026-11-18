"""Measure annotated objects in unmodified central crops for evaluation only.

The public publisher URL is below. For this run its servers timed out and the
annotated archive came from the byte-identical public mirror
https://huggingface.co/datasets/satabios/imagenet_val/resolve/main/ILSVRC2012_bbox_val_v3.tgz
with the MD5 published at https://image-net.org/challenges/LSVRC/2012/2012-downloads.php.
"""

import argparse
import hashlib
import io
import shutil
import tarfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

from PIL import Image

from imagenet_deit import ROOT
from prior_data import IMAGENETTE, IMAGEWOOF, raw_batches
from scale_operator import write_rows


ARCHIVE = ROOT / ".cache/ILSVRC2012_bbox_val_v3.tgz"
URL = "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_bbox_val_v3.tgz"
MIRROR = "https://huggingface.co/datasets/satabios/imagenet_val/resolve/main/ILSVRC2012_bbox_val_v3.tgz"
MD5 = "f4cd18b5ea29fe6bbea62ec9c20d80f0"
SHA256 = "994500634aefcd26e188b5d0cb10e062e954abef4edaa263dfbdae9c0f785257"


def annotations(path, download):
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Supply the ILSVRC2012 validation bounding-box archive at {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".part")
        for url in (URL, MIRROR):
            try:
                with urllib.request.urlopen(url, timeout=20) as response, temp.open("wb") as stream:
                    shutil.copyfileobj(response, stream)
                break
            except (TimeoutError, urllib.error.URLError):
                if url == MIRROR:
                    raise
        archive = temp.read_bytes()
        if hashlib.md5(archive).hexdigest() != MD5 or hashlib.sha256(archive).hexdigest() != SHA256:
            raise ValueError("ILSVRC validation annotation checksum mismatch")
        temp.replace(path)
    with path.open("rb") as f:
        if hashlib.file_digest(f, "md5").hexdigest() != MD5:
            raise ValueError("ILSVRC validation annotation checksum mismatch")
    return path


def measure(annotation, image_size, wnid):
    w, h = image_size
    orig_w = int(annotation.findtext("size/width"))
    orig_h = int(annotation.findtext("size/height"))
    assert abs((w / h) / (orig_w / orig_h) - 1) < .01
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    boxes = []
    for obj in annotation.findall("object"):
        if obj.findtext("name") != wnid:
            continue
        box = obj.find("bndbox")
        # ILSVRC validation minima can be zero; maxima reach width-1 / height-1.
        # Convert inclusive pixel indices to half-open edges before resizing.
        x1 = float(box.findtext("xmin")) * w / orig_w
        y1 = float(box.findtext("ymin")) * h / orig_h
        x2 = (float(box.findtext("xmax")) + 1) * w / orig_w
        y2 = (float(box.findtext("ymax")) + 1) * h / orig_h
        assert x2 > x1 and y2 > y1
        width = max(0.0, min(left + side, x2) - max(left, x1))
        height = max(0.0, min(top + side, y2) - max(top, y1))
        visible = width * height
        bounds = ((max(left, x1) - left) * 224 / side, (max(top, y1) - top) * 224 / side,
                  (min(left + side, x2) - left) * 224 / side, (min(top + side, y2) - top) * 224 / side)
        boxes.append((visible / side**2, visible / ((x2 - x1) * (y2 - y1)),
                      bounds if visible > 0 else (0, 0, 0, 0)))
    if not boxes:
        return (0, "", "", "missing", (0, 0, 0, 0))
    area, visibility, bounds = max(boxes)
    assert 0 <= area <= 1.000001 and 0 <= visibility <= 1.000001
    band = "small" if area <= .25 else "medium" if area <= .50 else "large"
    return len(boxes), area, visibility, band, bounds


def run(path, download):
    files = {}
    for dataset, mapping in (("imagenette", IMAGENETTE), ("imagewoof", IMAGEWOOF)):
        _, groups = raw_batches(dataset, download)
        for group in groups:
            for (wnid, target), (name, payload) in zip(mapping.items(), group, strict=True):
                with Image.open(io.BytesIO(payload)) as image:
                    files[name] = dataset, wnid, target, image.size
    rows = []
    with tarfile.open(annotations(path, download), "r|gz") as tar:
        for member in tar:
            file = PurePosixPath(member.name)
            name = file.stem + ".JPEG"
            if not member.isfile() or file.suffix.lower() != ".xml" or name not in files:
                continue
            dataset, wnid, target, size = files[name]
            with tar.extractfile(member) as stream:
                annotation = ET.parse(stream).getroot()
            n, area, visibility, band, bounds = measure(annotation, size, wnid)
            rows.append(dict(dataset=dataset, image=name, target=target, target_boxes=n,
                             area_frac=area, visibility=visibility, area_band=band,
                             bx1=bounds[0], by1=bounds[1], bx2=bounds[2], by2=bounds[3]))
    assert len(rows) == len(files) == 1000 and {r["image"] for r in rows} == set(files)
    output = ROOT / "results/scale_prior"
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "natural_sizes.csv", sorted(rows, key=lambda r: (r["dataset"], r["image"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--download", action="store_true")
    options = parser.parse_args()
    run(options.archive, options.download)
