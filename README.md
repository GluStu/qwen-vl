# qwenvl-siglip

A compact, from-scratch re-implementation of the **[Qwen-VL](https://arxiv.org/abs/2308.12966)**
recipe with modern open components: **SigLIP 2** (optionally fused with **DINOv2**) as the vision
encoder, Qwen-VL's **position-aware cross-attention resampler** as the adapter, and a **Qwen2.5**
language model.

> **Status:** research / learning project. Stage 1 (adapter alignment) is implemented, trained and
> evaluated (COCO CIDEr 1.20, see [Results](#results)); stages 2 and 3 are implemented but have not
> been run at scale. See [Limitations](#limitations).

## Architecture

```
image (384x384) ──► SigLIP2 ViT-SO400M/14 ─┐
                                           ├─ concat (27x27 patches, 1152+1024 ch)
image (378x378) ──► DINOv2 ViT-L/14 ───────┘  (optional)
                              │
                              ▼
        Resampler: 1 cross-attention layer, 256 learned queries,
        2D sin-cos positions added to queries and keys
                              │  256 visual tokens
                              ▼
 "<img>" [256 visual embeddings] "</img>" prompt text ──► Qwen2.5 LLM ──► answer
```

Text follows the Qwen-VL interface: `<img>…</img>` wraps the visual tokens, `<ref>…</ref>` names a
region and `<box>(x1,y1),(x2,y2)</box>` gives coordinates normalised to `[0, 1000)`.

## Installation

```bash
git clone https://github.com/GluStu/qwenvl-siglip && cd qwenvl-siglip
uv sync --all-extras          # or: pip install -e ".[all]"
```

Requires Python ≥ 3.10 and an NVIDIA GPU for training and inference. Base models (Qwen2.5,
SigLIP 2, DINOv2) download automatically from the Hugging Face Hub on first use.

## Quickstart: caption an image

Download `trainable.pt` and `args.json` from the [Releases](../../releases) page into `out/stage1_coco/`, then:

```bash
qvl-infer --run_dir out/stage1_coco --image my.jpg --prompt "Generate the caption in English:"
```

## Training

The three stages follow the paper (Table 8), scaled down.

| Stage | Trains | Data (this repo's recipe) | Peak LR |
|---|---|---|---|
| 1. Alignment | adapter (`--train_vit` adds the ViTs) | image–caption pairs | 2e-4 |
| 2. Multi-task | everything (or LoRA) | captioning, VQA, grounding, OCR | 5e-5 |
| 3. Instruction tuning | LLM + adapter | chat / instruction data | 1e-5 |

**Stage 1 on COCO captions (~1 GB):**

```bash
bash scripts/stage1_coco.sh
# equivalent to:
qvl-prepare-coco --root data/coco
qvl-train --stage 1 --data data/coco/train_captions.jsonl --image_root data/coco \
          --dinov2 vit_large_patch14_dinov2.lvd142m --out out/stage1_coco \
          --bs 4 --grad_accum 4 --grad_ckpt --max_steps 1000
```

Later stages chain through checkpoints:

```bash
qvl-train --stage 2 --init_ckpt out/stage1_coco/trainable.pt --lora 64 ...
qvl-train --stage 3 --init_ckpt out/stage2/trainable.pt ...
```

Run `qvl-train --help` for all options. Training logs go to `<out>/log.jsonl`.

### Data format

JSON or JSONL; each record is either of:

```json
{"image": "a.jpg", "prompt": "Generate the caption in English:", "answer": "two dogs play in the grass"}
{"image": "a.jpg", "conversations": [{"from": "human", "value": "<image>\nWhat is this?"}, {"from": "gpt", "value": "A dog."}]}
```

Grounding targets are built with `qwenvl_siglip.data.boxes.ref_box(phrase, (x1, y1, x2, y2), w, h)`.

## Evaluation

```bash
bash scripts/eval_coco.sh out/stage1_coco
```

Reports BLEU, ROUGE-L and CIDEr on held-out COCO images, plus diagnostics. `--ablate` re-scores with
shuffled images; if scores barely drop, the model is ignoring the image.

### Results

**Stage 1 on COCO captions.** Qwen2.5-1.5B, SigLIP 2 + DINOv2, adapter-only training.
Evaluated on 200 held-out COCO val2017 images (5 references each), greedy decoding.

| Metric | Trained model | Control: images shuffled |
|---|---|---|
| CIDEr | **1.203** | 0.047 |
| BLEU-1 | 0.731 | 0.395 |
| BLEU-4 | 0.288 | 0.026 |
| ROUGE-L | 0.535 | 0.297 |
| Unique-caption ratio | 0.965 | 0.965 |
| Avg. caption length (words) | 10.1 | 10.1 |

The shuffled-image control gives each caption the wrong image. Scores collapse (CIDEr 1.20 → 0.05),
so the model is genuinely conditioning on the visual tokens and is not writing generic captions.
The unique-caption ratio of 0.965 shows it is not collapsing to a few stock phrases.

<p align="center"><img src="results/stage1_coco_loss.png" width="560" alt="Stage 1 training loss"></p>

**Training setup**

| | |
|---|---|
| Trainable | adapter only (15.5M parameters); LLM, SigLIP 2 and DINOv2 frozen |
| Data | COCO val2017 captions: 4,800 images, 24,000 captions (200 images held out) |
| Schedule | 1000 steps × effective batch 16 ≈ 16k samples (≈ 0.67 epoch); peak LR 2e-4, cosine, 50 warm-up steps |
| Loss | 3.67 (step 10) → ≈ 2.1 (steps 400–1000), plateauing from about step 400 |
| Hardware | single laptop RTX 3050 |
| Wall-clock | ≈ 43 h (156,011 s). Typical pace was ≈ 2 min per optimizer step; two stalls of about 4 h each (steps 820 and 890) are included |

**Example outputs** (first rows of the held-out set; the full set is in `results/stage1_coco_eval.json`):

| Model caption | Reference caption |
|---|---|
| A skier is doing a trick in the air. | A man flying through the air while riding skis. |
| A little girl with a red umbrella in her hand. | A little girl holds up a big blue umbrella. |
| Two zebras eating grass in a field. | Two Zebras grazing together in a grassy area. |
| A fire hydrant on a street with a sign. | A red fire hydrant on a city sidewalk. |
| Two birds are swimming in a lake. | A family of swans swimming on top of a lake. |
| A cat is looking at a refrigerator in a kitchen. | A cat stands on top of a blue electric device. |

Captions are fluent and mostly correct about the scene, but often generic and sometimes wrong in
detail (colours, object identity, fine-grained categories such as swans vs. birds).

**How to read these numbers.** The held-out images are disjoint from the training images but come
from the same distribution and annotation style, and the evaluation set is small (200 images), with
CIDEr's IDF statistics computed over those 200 images only. The scores are therefore good for comparing
runs of this repository with each other, but they are **not comparable to published benchmark numbers**
(which use larger standard test splits). Differences of a few hundredths between runs are within noise.

Raw files in [`results/`](results/): evaluation output with all predictions
(`stage1_coco_eval.json`), the training log (`stage1_coco_train_log.txt`) and the loss plot.

## Repository layout

```
src/qwenvl_siglip/
├── constants.py        # special tokens, image placeholder string
├── modeling/           # resampler.py, vision.py (SigLIP2 + DINOv2), vlm.py (full model)
├── data/               # dataset.py, transforms.py, boxes.py, coco.py
├── training/           # trainer.py (stages 1-3)
├── evaluation/         # metrics.py, captioning.py
└── cli/                # qvl-train, qvl-infer, qvl-eval-captions, qvl-prepare-coco
tests/                  # unit tests (no model downloads needed)
scripts/                # reproducible end-to-end recipes
results/                # evaluation outputs
```

## Development

```bash
uv sync --all-extras && make test     # pytest
make lint                              # ruff
```

## Limitations

- Single-GPU training loop; no multi-GPU, DeepSpeed or layer-wise LR decay.
- Fixed 384px input (729 patches); the paper's 448px stage-2 resolution is not implemented.
- Single prompt template for all stages (no ChatML); no multi-image input.
- Stage-1 adapter captions images but does not yet answer questions or ground objects; that needs stages 2 and 3.
- With LoRA, the new special-token embeddings stay frozen.

## Acknowledgements and licenses

This is an independent implementation, not affiliated with the authors. Please cite the original work:

- **Qwen-VL** — Bai et al., [arXiv:2308.12966](https://arxiv.org/abs/2308.12966)
- **SigLIP 2** — Tschannen et al., [arXiv:2502.14786](https://arxiv.org/abs/2502.14786)
- **DINOv2** — Oquab et al., [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)

Base models keep their own licenses (check Qwen2.5, SigLIP 2 and DINOv2 before redistributing
weights). COCO is not redistributed; it is downloaded from its official source under its own terms.
This code is released under the [MIT License](LICENSE).

```bibtex
@article{Qwen-VL,
  title   = {Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author  = {Bai, Jinze and Bai, Shuai and Yang, Shusheng and Wang, Shijie and Tan, Sinan and Wang, Peng and Lin, Junyang and Zhou, Chang and Zhou, Jingren},
  journal = {arXiv preprint arXiv:2308.12966},
  year    = {2023}
}
```