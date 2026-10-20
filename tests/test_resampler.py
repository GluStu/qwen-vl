import pytest

torch = pytest.importorskip("torch")
from qwenvl_siglip.modeling.resampler import Resampler, sincos_2d  # noqa: E402


def test_sincos_shape_and_range():
    pos = sincos_2d(64, 16, 16)
    assert pos.shape == (256, 64)
    assert pos.abs().max() <= 1.0


def test_sincos_requires_dim_divisible_by_4():
    with pytest.raises(ValueError):
        sincos_2d(66, 4, 4)


def test_resampler_output_shape():
    m = Resampler(embed_dim=64, kv_dim=32, num_heads=4, key_grid=27)
    out = m(torch.randn(2, 27 * 27, 32))
    assert out.shape == (2, 256, 64)


def test_resampler_is_differentiable():
    m = Resampler(embed_dim=64, kv_dim=32, num_heads=4, key_grid=27)
    m(torch.randn(1, 27 * 27, 32)).sum().backward()
    assert m.query.grad is not None and m.kv_proj.weight.grad is not None
