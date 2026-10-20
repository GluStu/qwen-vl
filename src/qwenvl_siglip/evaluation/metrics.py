"""Caption metrics (pure Python; needs `pycocoevalcap`, no Java)."""

import re


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())


def compute_metrics(preds: dict[str, str], refs: dict[str, list[str]]) -> dict[str, float]:
    """`preds`: {id: caption}; `refs`: {id: [reference captions]}."""
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.rouge.rouge import Rouge
    except ImportError as e:
        raise ImportError("pip install 'qwenvl-siglip[eval]'") from e

    res = {k: [normalize(v) or "empty"] for k, v in preds.items()}
    gts = {k: [normalize(r) for r in refs[k]] for k in preds}
    bleu, _ = Bleu(4).compute_score(gts, res, verbose=0)
    cider, _ = Cider().compute_score(gts, res)
    rouge, _ = Rouge().compute_score(gts, res)
    return {
        "BLEU-1": float(bleu[0]),
        "BLEU-4": float(bleu[3]),
        "ROUGE-L": float(rouge),
        "CIDEr": float(cider),
        "unique_caption_ratio": len({v[0] for v in res.values()}) / len(res),
        "avg_words": sum(len(v[0].split()) for v in res.values()) / len(res),
    }
