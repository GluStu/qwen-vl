# Results

| File | Contents |
|---|---|
| `stage1_coco_eval.json` | Output of `qvl-eval-captions --ablate` for the stage-1 COCO run: metrics, shuffled-image metrics, and every prediction with its 5 references. |
| `stage1_coco_train_log.txt` | Raw training log (loss and learning rate every 10 optimizer steps). |
| `stage1_coco_loss.png` | Loss curve plotted from the training log. |

Reproduce with `bash scripts/stage1_coco.sh` followed by `bash scripts/eval_coco.sh out/stage1_coco`.
Exact numbers can differ slightly between runs (data order, hardware, library versions).
