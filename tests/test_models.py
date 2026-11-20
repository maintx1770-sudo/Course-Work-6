import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.models import compute_metrics


def test_compute_metrics_perfect():
    m = compute_metrics([0, 1, 2, 3], [0, 1, 2, 3])
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0


def test_compute_metrics_keys_and_range():
    m = compute_metrics([0, 1, 2, 3], [0, 1, 1, 3])
    assert set(m) == {"accuracy", "precision_macro", "recall_macro", "f1_macro"}
    assert all(0.0 <= v <= 1.0 for v in m.values())
