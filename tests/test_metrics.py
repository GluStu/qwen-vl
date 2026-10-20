import pytest

pytest.importorskip("pycocoevalcap")
from qwenvl_siglip.evaluation.metrics import compute_metrics, normalize  # noqa: E402

REFS = {
    "a": ["A man riding a wave on a surfboard.", "a surfer rides a big wave", "man surfing the ocean",
          "person on surfboard on a wave", "a guy surfs"],
    "b": ["A cat sleeping on a couch.", "a cat is asleep on the sofa", "a sleeping cat on a couch",
          "cat naps on a sofa", "kitty asleep on couch"],
    "c": ["Two dogs play in the grass.", "dogs playing on a lawn", "two dogs running in a field",
          "a couple of dogs play outside", "puppies chase each other on grass"],
}  # fmt: skip


def test_normalize():
    assert normalize("A Cat, sleeping!") == "a cat sleeping"


def test_correct_captions_beat_shuffled_ones():
    good = {"a": "a man riding a wave on a surfboard", "b": "a cat sleeping on a couch",
            "c": "two dogs playing in the grass"}  # fmt: skip
    bad = {"a": good["b"], "b": good["c"], "c": good["a"]}
    mg, mb = compute_metrics(good, REFS), compute_metrics(bad, REFS)
    assert mg["CIDEr"] > 10 * max(mb["CIDEr"], 1e-3)
    assert mg["BLEU-4"] > mb["BLEU-4"]


def test_diversity_stat_detects_collapse():
    same = dict.fromkeys(REFS, "a picture of something")
    assert compute_metrics(same, REFS)["unique_caption_ratio"] == pytest.approx(1 / 3)
