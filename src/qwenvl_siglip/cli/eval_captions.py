"""``qvl-eval-captions``: BLEU / ROUGE-L / CIDEr on a held-out caption set."""

import argparse
import json
from pathlib import Path

from qwenvl_siglip.constants import CAPTION_PROMPT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt", default="trainable.pt")
    ap.add_argument("--data", required=True, help="held-out JSONL from qvl-prepare-coco")
    ap.add_argument("--image_root", required=True)
    ap.add_argument("--prompt", default=CAPTION_PROMPT)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--beams", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick test)")
    ap.add_argument("--ablate", action="store_true", help="also score with shuffled images")
    ap.add_argument("--out", default=None, help="result JSON (default: <run_dir>/eval_captions.json)")
    a = ap.parse_args()

    from qwenvl_siglip.evaluation.captioning import evaluate_captions
    from qwenvl_siglip.modeling import QwenVLStyle

    model = QwenVLStyle.from_run_dir(a.run_dir, a.ckpt)
    result = evaluate_captions(
        model, a.data, a.image_root, a.prompt, a.bs, a.beams, a.max_new_tokens, a.limit, a.ablate
    )

    m = result["metrics"]
    print("\n=== metrics ===")
    for k, v in m.items():
        print(f"{k:22s} {v:.4f}")
    if a.ablate:
        s = result["metrics_shuffled_images"]
        print("\n=== shuffled-image ablation ===")
        for k in ("BLEU-4", "ROUGE-L", "CIDEr"):
            print(f"{k:22s} {s[k]:.4f}   (drop {m[k] - s[k]:+.4f})")
        if m["CIDEr"] - s["CIDEr"] < 0.1 * max(m["CIDEr"], 1e-6):
            print("WARNING: scores barely change with wrong images -> the model ignores the image.")
    print("\n=== samples ===")
    for name, rec in list(result["predictions"].items())[:8]:
        print(f"{name}\n  pred: {rec['pred']}\n  ref : {rec['refs'][0]}")

    out = Path(a.out or Path(a.run_dir) / "eval_captions.json")
    out.write_text(json.dumps(result, indent=1))
    print("saved ->", out)


if __name__ == "__main__":
    main()
