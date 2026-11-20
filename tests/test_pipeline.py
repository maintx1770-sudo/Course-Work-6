import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.baseline import build_pipeline, count_trainable_params

TEXTS = [
    "the team won the football match",
    "stocks fell as oil prices rose",
    "scientists release new software chip",
    "president meets leaders at summit",
] * 6
LABELS = [1, 2, 3, 0] * 6


def test_baseline_pipeline_fits_and_predicts():
    pipe = build_pipeline(C=1.0).fit(TEXTS, LABELS)
    assert list(pipe.predict(["the team won the football match"])) == [1]
    assert count_trainable_params(pipe) > 0


def test_compare_merges_metric_files(tmp_path):
    pytest.importorskip("torch")
    from src.compare import compare

    row = {
        "model": "baseline", "accuracy": 0.9, "precision_macro": 0.9, "recall_macro": 0.9,
        "f1_macro": 0.9, "train_time_s": 1.0, "inference_time_s": 1.0,
        "trainable_params": 10, "model_size_mb": 1.0,
    }
    src_csv = tmp_path / "m.csv"
    pd.DataFrame([row]).to_csv(src_csv, index=False)
    out = tmp_path / "eval.csv"
    df = compare([src_csv], output=out)
    assert out.exists()
    assert "Macro F1-score" in df.columns
