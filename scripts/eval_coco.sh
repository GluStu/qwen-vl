#!/usr/bin/env bash
set -euo pipefail
RUN_DIR="${1:-out/stage1_coco}"
qvl-eval-captions --run_dir "$RUN_DIR" \
  --data data/coco/heldout_captions.jsonl --image_root data/coco --ablate
