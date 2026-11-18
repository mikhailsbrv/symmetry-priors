"""Paired single-image scale tests from the original ILSVRC validation split."""

import hashlib
import io
import tarfile
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

from imagenet_deit import ROOT
from photo_swaps import CLASSES as IMAGENETTE, validation_images
from transfer_objects import checked_file


# ImageNet-1k indices follow timm/data/_info/imagenet_synsets.txt.
IMAGEWOOF = {"n02086240": 155, "n02087394": 159, "n02088364": 162,
             "n02089973": 167, "n02093754": 182, "n02096294": 193,
             "n02099601": 207, "n02105641": 229, "n02111889": 258, "n02115641": 273}
WOOF_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz"
WOOF_SHA = "b5ffa16037e07f60882434f55b7814a3d44483f2a484129f251604bc0d0f8172"


def raw_batches(dataset, download=False):
    """Each batch contains one original validation JPEG per class, ordered by WNID."""
    if dataset == "imagenette":
        return list(IMAGENETTE.values()), validation_images(download)
    if dataset != "imagewoof":
        raise ValueError(f"Unknown dataset: {dataset}")
    archive = checked_file(ROOT / ".cache/imagewoof2-160.tgz", WOOF_SHA, WOOF_URL, download)
    groups = {wnid: [] for wnid in IMAGEWOOF}
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
    return list(IMAGEWOOF.values()), list(zip(*groups.values()))


def render_views(name, payload):
    with Image.open(io.BytesIO(payload)) as source:
        source = source.convert("RGB")
        side = min(source.size)
        left, top = (source.width - side) // 2, (source.height - side) // 2
        crop = source.crop((left, top, left + side, top + side))
    clean = np.asarray(crop.resize((224, 224), Image.Resampling.BICUBIC))
    tile = np.asarray(crop.resize((64, 64), Image.Resampling.BICUBIC))
    small = np.zeros_like(clean)
    small[80:144, 80:144] = tile

    # Positions and texture are keyed by image ID, independently of class or pixels.
    if name.startswith("ILSVRC2012_val_"):
        image_id = int(name.removesuffix(".JPEG").rsplit("_", 1)[1])
    else:
        image_id = int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "little")
    placement_rng = np.random.default_rng([42, image_id, 0])
    texture_rng = np.random.default_rng([42, image_id, 1])
    dx, dy = placement_rng.integers(-32, 33, size=2)
    if dx == 0 and dy == 0:
        dx = 1
    shifted = np.empty_like(clean)
    shifted[:] = (124, 116, 104)
    shifted[64+dy:160+dy, 64+dx:160+dx] = crop.resize((96, 96), Image.Resampling.BICUBIC)

    low_freq = np.clip(texture_rng.normal(125, 22, size=(8, 8, 3)), 0, 255).astype(np.uint8)
    field = np.asarray(Image.fromarray(low_freq).resize((224, 224), Image.Resampling.BICUBIC))
    noise = texture_rng.normal(0, 6, size=field.shape)
    textured = np.clip(np.rint(field + noise), 0, 255).astype(np.uint8)
    textured[80+dy:144+dy, 80+dx:144+dx] = tile
    return {"clean": clean, "small64": small, "small96_shift": shifted,
            "small64_texture": textured}


def batches(dataset, download=False):
    """Yield names, ImageNet-1k labels and four equally ordered uint8 NHWC batches."""
    targets, grouped = raw_batches(dataset, download)
    for batch in grouped:
        names = [name for name, _ in batch]
        views = [render_views(name, payload) for name, payload in batch]
        frames = {view: np.stack([image[view] for image in views]) for view in views[0]}
        yield names, targets, frames
