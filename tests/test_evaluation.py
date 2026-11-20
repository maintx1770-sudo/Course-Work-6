import pandas as pd
import pytest

pytest.importorskip("torch")

from src.evaluate import LABEL_NAMES, get_misclassified_samples, plot_confusion_matrix


def test_plot_confusion_matrix_saves_figure(tmp_path):
    out = tmp_path / "cm.png"
    cm = plot_confusion_matrix([0, 1, 2, 3, 0], [0, 1, 2, 2, 1], "demo", out)
    assert out.exists()
    assert cm.shape == (4, 4)
    assert cm.sum() == 5


def test_get_misclassified_samples(tmp_path):
    p = tmp_path / "pred.csv"
    pd.DataFrame(
        {"text": ["a", "b", "c"], "true_label": [0, 1, 2], "pred_label": [0, 2, 2]}
    ).to_csv(p, index=False)
    errors = get_misclassified_samples(p)
    assert len(errors) == 1
    assert errors.iloc[0]["true_name"] == LABEL_NAMES[1]
    assert errors.iloc[0]["pred_name"] == LABEL_NAMES[2]
