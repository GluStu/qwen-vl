"""Grounding helpers using the Qwen-VL text format: ``(x1,y1),(x2,y2)`` in [0, 1000)."""

import re

_BOX_RE = re.compile(r"\((\d+),(\d+)\),\((\d+),(\d+)\)")


def format_box(x1: float, y1: float, x2: float, y2: float, w: float, h: float) -> str:
    """Pixel box -> ``<box>(x1,y1),(x2,y2)</box>`` with coordinates normalised to [0, 999]."""

    def c(v: float, size: float) -> int:
        return min(999, max(0, int(v / size * 1000)))

    return f"<box>({c(x1, w)},{c(y1, h)}),({c(x2, w)},{c(y2, h)})</box>"


def ref_box(phrase: str, box_xyxy: tuple[float, float, float, float], w: float, h: float) -> str:
    return f"<ref>{phrase}</ref>{format_box(*box_xyxy, w, h)}"


def parse_boxes(text: str, w: float, h: float) -> list[tuple[float, float, float, float]]:
    """Extract boxes from model output and convert back to pixel coordinates."""
    out = []
    for m in _BOX_RE.finditer(text):
        a, b, c, d = map(int, m.groups())
        out.append((a / 1000 * w, b / 1000 * h, c / 1000 * w, d / 1000 * h))
    return out
