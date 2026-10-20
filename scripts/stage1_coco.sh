#!/usr/bin/env bash
# Stage-1 alignment on COCO val captions (~1 GB). Adapter-only training, frozen LLM + ViTs.
set -euo pipefail

qvl-prepare-coco --root data/coco

qvl-train --stage 1 \
  --llm Qwen/Qwen2.5-1.5B \
  --siglip google/siglip2-so400m-patch14-384 \
  --dinov2 vit_large_patch14_dinov2.lvd142m \
  --data data/coco/train_captions.jsonl \
  --image_root data/coco \
  --out out/stage1_coco \
  --bs 4 --grad_accum 4 --grad_ckpt \
  --max_steps 1000 --warmup 50
