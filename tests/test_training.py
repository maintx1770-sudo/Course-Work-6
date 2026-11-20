import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.train import compute_metrics, set_seed


def test_compute_metrics_wrong_everywhere():
    m = compute_metrics([0, 1, 2, 3], [1, 2, 3, 0])
    assert m["accuracy"] == 0.0


def test_set_seed_is_reproducible():
    import torch

    set_seed(42)
    a = torch.rand(3)
    set_seed(42)
    b = torch.rand(3)
    assert torch.equal(a, b)
