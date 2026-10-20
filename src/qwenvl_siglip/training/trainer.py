"""Single-GPU trainer for the three Qwen-VL stages."""

import json
import time
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from qwenvl_siglip.data.dataset import Collator, VLDataset
from qwenvl_siglip.modeling.vlm import QwenVLStyle

# Peak learning rates from the paper (Table 8).
DEFAULT_LR = {1: 2e-4, 2: 5e-5, 3: 1e-5}


def build_model(args: Namespace) -> QwenVLStyle:
    """Frozen parts stay bf16 (cheap); trained parts are fp32 and run under bf16 autocast."""
    llm_full_ft = args.stage in (2, 3) and not args.lora
    vit_trained = args.train_vit if args.stage == 1 else args.stage == 2
    model = QwenVLStyle(
        args.llm,
        args.siglip,
        args.dinov2,
        llm_dtype=torch.float32 if llm_full_ft else torch.bfloat16,
        vit_dtype=torch.float32 if vit_trained else torch.bfloat16,
    )
    model.set_trainable(args.stage, args.train_vit)
    if args.grad_ckpt:
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.lora and args.stage in (2, 3):
        model.apply_lora(args.lora)
    if args.init_ckpt:
        result = model.load_state_dict(torch.load(args.init_ckpt, map_location="cpu"), strict=False)
        print(f"loaded {args.init_ckpt} | unexpected keys: {len(result.unexpected_keys)}")
    return model


def save_trainable(model: QwenVLStyle, out: Path, tag: str) -> None:
    state = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(state, out / f"trainable_{tag}.pt")
    torch.save(state, out / "trainable.pt")


def train(args: Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA GPU")
    device = "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))
    lr = args.lr or DEFAULT_LR[args.stage]

    model = build_model(args).to(device).train()
    model.tokenizer.save_pretrained(out)

    groups: dict[str, list[torch.nn.Parameter]] = {"adapter": [], "vit": [], "llm": []}
    for name, p in model.named_parameters():
        if p.requires_grad:
            key = "adapter" if name.startswith("adapter") else "vit" if name.startswith("vision") else "llm"
            groups[key].append(p)
    lrs = {"adapter": lr, "vit": args.vit_lr, "llm": lr}
    optimizer = torch.optim.AdamW(
        [{"params": v, "lr": lrs[k]} for k, v in groups.items() if v],
        betas=(0.9, 0.98), eps=1e-6, weight_decay=0.05,
    )  # fmt: skip
    print({k: f"{sum(p.numel() for p in v) / 1e6:.1f}M" for k, v in groups.items()}, "trainable params")

    dataset = VLDataset(args.data, args.image_root, model.tokenizer, args.max_len)
    loader = DataLoader(
        dataset, batch_size=args.bs, shuffle=True, num_workers=args.workers,
        collate_fn=Collator(model.tokenizer.pad_token_id), drop_last=True,
    )  # fmt: skip
    total = args.max_steps if args.max_steps > 0 else args.epochs * len(loader) // args.grad_accum
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup, total)
    clip_params = [p for g in groups.values() for p in g]

    step = micro = last_micro = 0
    running, t0 = 0.0, time.time()
    log = open(out / "log.jsonl", "a")
    while step < total:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device, dtype=torch.bfloat16):
                loss = model(**batch).loss
            (loss / args.grad_accum).backward()
            running += loss.item()
            micro += 1
            if micro <= 3:
                print(f"micro-batch {micro} ok, loss {loss.item():.3f}, {time.time() - t0:.1f}s", flush=True)
            if micro % args.grad_accum:
                continue

            torch.nn.utils.clip_grad_norm_(clip_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == 1:
                avg = running / (micro - last_micro)
                rec = {"step": step, "loss": avg, "lr": scheduler.get_last_lr()[0], "sec": time.time() - t0}
                print(f"step {step}/{total} loss {avg:.4f} lr {rec['lr']:.2e} {rec['sec']:.0f}s", flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                running, last_micro = 0.0, micro
            if step % args.save_every == 0:
                save_trainable(model, out, f"step{step}")
            if step >= total:
                break
    save_trainable(model, out, "final")
    print("done ->", out)
