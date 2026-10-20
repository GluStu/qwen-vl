"""Supervised dataset and collator.

Each JSON / JSONL record is ONE of::

    {"image": "a.jpg", "prompt": "...", "answer": "..."}
    {"image": "a.jpg", "conversations": [{"from": "human", "value": "<image>\\nQ"},
                                         {"from": "gpt", "value": "A"}, ...]}  # LLaVA format

The loss is computed on answers (+ EOS) only; image tokens and prompts are masked with -100.
"""

import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from qwenvl_siglip.constants import IMG_STR
from qwenvl_siglip.data.transforms import build_transform


def _pairs(rec: dict) -> list[tuple[str, str]]:
    if "conversations" in rec:
        conv = rec["conversations"]
        return [
            (conv[i]["value"].replace("<image>", "").strip(), conv[i + 1]["value"])
            for i in range(0, len(conv) - 1, 2)
        ]
    return [(rec["prompt"], rec["answer"])]


def read_records(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        return json.load(f)


class VLDataset(Dataset):
    def __init__(self, path, image_root, tokenizer, max_len=2048, image_size=384):
        self.recs = read_records(path)
        self.root = Path(image_root)
        self.tok = tokenizer
        self.max_len = max_len
        self.tf = build_transform(image_size)

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int) -> dict:
        for _ in range(10):  # skip broken images instead of crashing a long run
            try:
                return self._load(self.recs[i])
            except Exception as e:
                print(f"[data] skipping sample {i}: {e}")
                i = random.randrange(len(self.recs))
        raise RuntimeError("too many broken samples")

    def _load(self, rec: dict) -> dict:
        pix = self.tf(Image.open(self.root / rec["image"]).convert("RGB"))
        ids: list[int] = []
        labels: list[int] = []

        def enc(s: str) -> list[int]:
            return self.tok(s, add_special_tokens=False)["input_ids"]

        for k, (prompt, answer) in enumerate(_pairs(rec)):
            p = enc((IMG_STR if k == 0 else "") + prompt + "\n")
            a = enc(answer) + [self.tok.eos_token_id]
            ids += p + a
            labels += [-100] * len(p) + a
        return {
            "pixel_values": pix,
            "input_ids": torch.tensor(ids[: self.max_len]),
            "labels": torch.tensor(labels[: self.max_len]),
        }


class Collator:
    """Right-pads a batch of variable-length samples."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        longest = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), longest), self.pad_id)
        labels = torch.full((len(batch), longest), -100)
        mask = torch.zeros((len(batch), longest), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n], labels[i, :n], mask[i, :n] = b["input_ids"], b["labels"], 1
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "input_ids": ids,
            "labels": labels,
            "attention_mask": mask,
        }
