"""Download COCO 2017 val (~1 GB) and build caption JSONLs for stage-1 training."""

import json
import random
import urllib.request
import zipfile
from pathlib import Path

from qwenvl_siglip.constants import CAPTION_PROMPT

FILES = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",  # ~780 MB
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",  # ~240 MB  # noqa: E501
}  # fmt: skip


def _download(url: str, dst: Path) -> None:
    def hook(blocks, block_size, total):
        print(f"\r  {blocks * block_size / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)

    print("downloading", url)
    urllib.request.urlretrieve(url, dst, hook)
    print()


def prepare_coco(root: str = "data/coco", heldout_images: int = 200, seed: int = 0) -> None:
    """Writes ``train_captions.jsonl`` and ``heldout_captions.jsonl`` under `root`."""
    root_p = Path(root)
    root_p.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        archive, marker = root_p / name, root_p / f".{name}.done"
        if not archive.exists():
            _download(url, archive)
        if not marker.exists():
            print("extracting", name)
            only = None if name == "val2017.zip" else ["annotations/captions_val2017.json"]
            with zipfile.ZipFile(archive) as z:
                z.extractall(root_p, members=only)
            marker.touch()

    ann = json.loads((root_p / "annotations/captions_val2017.json").read_text())
    files = {im["id"]: f"val2017/{im['file_name']}" for im in ann["images"]}
    caps: dict[int, list[str]] = {}
    for a in ann["annotations"]:
        caps.setdefault(a["image_id"], []).append(a["caption"].strip())

    ids = sorted(files)
    random.Random(seed).shuffle(ids)
    held, train = ids[:heldout_images], ids[heldout_images:]

    def dump(path: Path, image_ids: list[int]) -> None:
        n = 0
        with open(path, "w") as f:
            for i in image_ids:
                for c in caps[i]:
                    rec = {"image": files[i], "prompt": CAPTION_PROMPT, "answer": c}
                    f.write(json.dumps(rec) + "\n")
                    n += 1
        print(f"{path}: {n} samples from {len(image_ids)} images")

    dump(root_p / "train_captions.jsonl", train)
    dump(root_p / "heldout_captions.jsonl", sorted(held))
