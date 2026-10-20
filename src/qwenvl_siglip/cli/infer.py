"""``qvl-infer``: caption / query a single image with a trained run directory."""

import argparse

from qwenvl_siglip.constants import CAPTION_PROMPT, IMG_STR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True, help="directory with args.json + trainable.pt")
    ap.add_argument("--ckpt", default="trainable.pt")
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default=CAPTION_PROMPT)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    a = ap.parse_args()

    import torch
    from PIL import Image

    from qwenvl_siglip.data.boxes import parse_boxes
    from qwenvl_siglip.data.transforms import build_transform
    from qwenvl_siglip.modeling import QwenVLStyle

    model = QwenVLStyle.from_run_dir(a.run_dir, a.ckpt)
    img = Image.open(a.image).convert("RGB")
    pix = build_transform()(img).unsqueeze(0).to("cuda")
    enc = model.tokenizer(IMG_STR + a.prompt + "\n", add_special_tokens=False, return_tensors="pt")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            pix, enc["input_ids"].cuda(), enc["attention_mask"].cuda(),
            max_new_tokens=a.max_new_tokens, do_sample=False,
            eos_token_id=model.tokenizer.eos_token_id, pad_token_id=model.tokenizer.pad_token_id,
        )  # fmt: skip
    text = model.tokenizer.decode(out[0], skip_special_tokens=False)
    print(text)
    boxes = parse_boxes(text, *img.size)
    if boxes:
        print("boxes (px):", boxes)


if __name__ == "__main__":
    main()
