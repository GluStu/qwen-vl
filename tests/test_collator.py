import pytest

torch = pytest.importorskip("torch")
from qwenvl_siglip.data.dataset import Collator  # noqa: E402


def _sample(n):
    return {
        "pixel_values": torch.zeros(3, 8, 8),
        "input_ids": torch.arange(1, n + 1),
        "labels": torch.arange(1, n + 1),
    }


def test_collator_pads_and_masks():
    batch = Collator(pad_id=0)([_sample(3), _sample(5)])
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert batch["labels"][0, 3:].tolist() == [-100, -100]
    assert batch["pixel_values"].shape == (2, 3, 8, 8)
