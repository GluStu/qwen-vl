"""Caption generation + scoring on a held-out JSONL (see `data.coco`)."""

import json
from collections import OrderedDict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from qwenvl_siglip.constants import CAPTION_PROMPT, IMG_STR
from qwenvl_siglip.data.transforms import build_transform
from qwenvl_siglip.evaluation.metrics import compute_metrics


class _ImageDataset(Dataset):
    def __init__(self, keys, root):
        self.keys, self.root, self.tf = keys, Path(root), build_transform()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        return self.keys[i], self.tf(Image.open(self.root / self.keys[i]).convert("RGB"))


def _collate(batch):
    return [b[0] for b in batch], torch.stack([b[1] for b in batch])


@torch.no_grad()
def _generate(model, loader, prompt, shuffle, beams, max_new_tokens, device):
    tok = model.tokenizer
    enc = tok(IMG_STR + prompt + "\n", add_special_tokens=False, return_tensors="pt")
    preds: dict[str, str] = {}
    for names, pix in loader:
        pix = pix.to(device)
        if shuffle and pix.size(0) > 1:
            pix = pix.roll(1, dims=0)  # sample i is captioned from image i-1
        n = pix.size(0)
        ids = enc["input_ids"].to(device).expand(n, -1).contiguous()
        att = enc["attention_mask"].to(device).expand(n, -1).contiguous()
        with torch.autocast(device, dtype=torch.bfloat16):
            out = model.generate(
                pix, ids, att,
                max_new_tokens=max_new_tokens, do_sample=False, num_beams=beams,
                eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            )  # fmt: skip
        for name, text in zip(names, tok.batch_decode(out, skip_special_tokens=True), strict=True):
            preds[name] = text.strip().split("\n")[0]
        print(f"\r{len(preds)}/{len(loader.dataset)}", end="", flush=True)
    print()
    return preds


def evaluate_captions(
    model,
    data: str,
    image_root: str,
    prompt: str = CAPTION_PROMPT,
    bs: int = 8,
    beams: int = 1,
    max_new_tokens: int = 40,
    limit: int = 0,
    ablate: bool = False,
    device: str = "cuda",
) -> dict:
    """Returns {"metrics", "predictions"[, "metrics_shuffled_images"]}."""
    refs: OrderedDict[str, list[str]] = OrderedDict()
    with open(data) as f:
        for line in f:
            r = json.loads(line)
            refs.setdefault(r["image"], []).append(r["answer"])
    keys = list(refs)[: limit or None]
    loader = DataLoader(_ImageDataset(keys, image_root), batch_size=bs, num_workers=2, collate_fn=_collate)

    preds = _generate(model, loader, prompt, False, beams, max_new_tokens, device)
    result = {
        "metrics": compute_metrics(preds, refs),
        "predictions": {k: {"pred": preds[k], "refs": refs[k]} for k in preds},
    }
    if ablate:
        shuffled = _generate(model, loader, prompt, True, beams, max_new_tokens, device)
        result["metrics_shuffled_images"] = compute_metrics(shuffled, refs)
    return result
