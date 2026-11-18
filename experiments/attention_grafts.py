"""Replace individual attention-head outputs at the CLS position in blocks 10 and 11."""

import argparse

import numpy as np
import torch

from imagenet_deit import ROOT, load_deit, normalize_image
from photo_swaps import CLASSES, CONDITIONS, render, validation_images
from scale_operator import write_rows

BLOCKS = (10, 11)


def collect(model, images):
    inputs, heads, hooks = {}, {}, []
    for number in BLOCKS:
        block = model.blocks[number - 1]
        hooks.append(block.register_forward_pre_hook(
            lambda _, args, n=number: inputs.update({n: args[0]})))
        for h, head in enumerate(block.attention.heads):
            hooks.append(head.register_forward_hook(
                lambda _, args, out, n=number, h=h: heads.update({(n, h): out[:, 0]})))
    try:
        logits = model(images)
    finally:
        for handle in hooks:
            handle.remove()
    return logits, inputs, heads


def graft(model, number, tokens, replacements, direct=False):
    block = model.blocks[number - 1]
    hooks = []
    def replace_cls(tensor, cls):
        patched = tensor.clone()
        patched[:, 0] = cls
        return patched

    if direct:
        # Independent implementation of replacing all heads just before projection.
        concat = torch.cat([replacements[h] for h in range(3)], dim=-1)
        hooks.append(block.attention.projection.register_forward_pre_hook(
            lambda _, args: (replace_cls(args[0], concat),)))
    else:
        for h, donor in replacements.items():
            hooks.append(block.attention.heads[h].register_forward_hook(
                lambda _, args, out, donor=donor: replace_cls(out, donor)))
    try:
        for tail in list(model.blocks)[number - 1:]:
            tokens = tail(tokens)
        return model.classifier(model.norm(tokens[:, 0]))
    finally:
        for handle in hooks:
            handle.remove()


@torch.inference_mode()
def run(download, output):
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_deit("imagenet", download=download).to(device)
    targets = list(CLASSES.values())
    labels = torch.tensor(targets, device=device)
    view_ids = [CONDITIONS.index(view) for view in ("small64_black", "from64_to224")]
    rows = []
    for batch_id, batch in enumerate(validation_images(download)):
        names = [name for name, _ in batch]
        views = np.stack([render(payload)[view_ids] for _, payload in batch])
        pixels = torch.from_numpy(views).to(device, torch.float32).div_(255)
        small, large = [normalize_image(x.permute(0, 3, 1, 2)) for x in pixels.unbind(1)]
        logits, inputs, _ = collect(model, small)
        _, _, heads = collect(model, large)
        order = np.arange(len(batch))
        other = np.roll(order, 1 + batch_id % (len(batch) - 1))
        for number in BLOCKS:
            baseline = graft(model, number, inputs[number], {})
            torch.testing.assert_close(baseline, logits, rtol=1e-4, atol=3e-5)
            for source, ids in (("same", order), ("other", other)):
                indices = torch.tensor(ids, device=device)
                donors = {h: heads[number, h][indices] for h in range(3)}
                for head, values in [(str(h), {h: donors[h]}) for h in range(3)] + [("all", donors)]:
                    scores = graft(model, number, inputs[number], values)
                    if head == "all":
                        expected = graft(model, number, inputs[number], values, direct=True)
                        torch.testing.assert_close(scores, expected, rtol=1e-4, atol=3e-5)
                    prob = scores.softmax(-1)
                    top = scores.argmax(-1).tolist()
                    p_true = prob.gather(1, labels[:, None]).flatten().tolist()
                    p_donor = prob.gather(1, labels[indices, None]).flatten().tolist()
                    for j, k in enumerate(ids):
                        rows.append(dict(image=names[j], target=targets[j], block=number, head=head,
                                         source=source, donor_image=names[k], donor_target=targets[k],
                                         top1=top[j], p_true=p_true[j], p_donor=p_donor[j]))
        if (batch_id + 1) % 10 == 0:
            print(f"{(batch_id + 1) * len(batch)}/500 photos", flush=True)
    assert len(rows) == 500 * len(BLOCKS) * 4 * 2
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "attention_grafts.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch verified weights and data if missing")
    run(parser.parse_args().download, ROOT / "results/object_scale")
