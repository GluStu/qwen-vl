"""``qvl-train``: train one of the three Qwen-VL stages."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--llm", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--siglip", default="google/siglip2-so400m-patch14-384")
    p.add_argument("--dinov2", default=None, help="e.g. vit_large_patch14_dinov2.lvd142m")
    p.add_argument("--data", required=True, help=".json / .jsonl training file")
    p.add_argument("--image_root", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--init_ckpt", default=None, help="trainable.pt from the previous stage")
    p.add_argument("--train_vit", action="store_true", help="stage 1 only: also train the ViT(s)")
    p.add_argument("--lora", type=int, default=0, help="LoRA rank for the LLM (stage 2/3); 0 = full FT")
    p.add_argument("--grad_ckpt", action="store_true", help="gradient checkpointing (saves memory)")
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--lr", type=float, default=None, help="default depends on stage (paper values)")
    p.add_argument("--vit_lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=10)
    return p


def main() -> None:
    from qwenvl_siglip.training.trainer import train  # heavy imports only when actually training

    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
