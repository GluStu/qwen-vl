from qwenvl_siglip.data.boxes import format_box, parse_boxes, ref_box


def test_format_box_normalises_to_1000():
    assert format_box(0, 0, 320, 240, 640, 480) == "<box>(0,0),(500,500)</box>"


def test_format_box_clips_to_999():
    assert format_box(0, 0, 640, 480, 640, 480) == "<box>(0,0),(999,999)</box>"


def test_ref_box_wraps_phrase():
    s = ref_box("a cat", (64, 48, 320, 240), 640, 480)
    assert s.startswith("<ref>a cat</ref><box>")


def test_roundtrip_is_close():
    w, h, box = 640, 480, (100.0, 50.0, 400.0, 300.0)
    (back,) = parse_boxes(format_box(*box, w, h), w, h)
    assert all(abs(a - b) <= 1.0 * max(w, h) / 1000 for a, b in zip(box, back, strict=True))


def test_parse_boxes_ignores_text_without_boxes():
    assert parse_boxes("a dog on grass", 100, 100) == []
