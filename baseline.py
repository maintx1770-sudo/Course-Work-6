"""Baseline for Coursework Topic 6: TF-IDF + Logistic Regression on AG News.

Run from the repository root:
    python -m src.baseline

Outputs:
    results/metrics/baseline_metrics.csv
    results/metrics/baseline_test_predictions.csv
    results/checkpoints/baseline/baseline_pipeline.joblib
    results/checkpoints/baseline/config.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

from src.data import LABEL_NAMES, SEED, load_splits  # shared split: 108,000 / 12,000 / 7,600

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "metrics" / "baseline_metrics.csv"
PREDICTIONS_PATH = ROOT / "results" / "metrics" / "baseline_predictions.csv"
CKPT_DIR = ROOT / "results" / "checkpoints" / "baseline"
MODEL_PATH = CKPT_DIR / "baseline_pipeline.joblib"
CONFIG_PATH = CKPT_DIR / "config.json"

C_GRID = [0.25, 0.5, 1.0, 2.0, 5.0]  # C=1 was best at the edge of [1, 5, 10], so extend below 1


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_pipeline(C: float = 1.0) -> Pipeline:
    """TF-IDF (unigram + bigram) followed by multinomial Logistic Regression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=50_000,
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=SEED)),
        ]
    )


def count_trainable_params(pipe: Pipeline) -> int:
    """Weights + biases of the classifier. TF-IDF has no learned weights."""
    clf = pipe.named_steps["clf"]
    return int(clf.coef_.size + clf.intercept_.size)


def model_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Train / evaluate
# --------------------------------------------------------------------------- #
def train(x_train, y_train, x_val, y_val, c_grid=C_GRID):
    """Pick C on the validation set. Return (best_pipeline, best_C, fit_seconds, val_f1)."""
    best = None
    for C in c_grid:
        pipe = build_pipeline(C)
        start = time.perf_counter()
        pipe.fit(x_train, y_train)
        fit_time = time.perf_counter() - start
        val_f1 = f1_score(y_val, pipe.predict(x_val), average="macro")
        print(f"C={C:<5} val macro-F1={val_f1:.4f}  fit time={fit_time:.1f}s")
        if best is None or val_f1 > best["val_f1"]:
            best = {"pipe": pipe, "C": C, "fit_time": fit_time, "val_f1": val_f1}
    return best["pipe"], best["C"], best["fit_time"], best["val_f1"]


def evaluate(pipe: Pipeline, x_test, y_test) -> dict:
    """Test metrics. Inference time = TF-IDF transform + predict on the whole test set."""
    start = time.perf_counter()
    y_pred = pipe.predict(x_test)
    inference_time = time.perf_counter() - start

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    return {
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "inference_time_s": inference_time,
        "inference_ms_per_sample": 1000 * inference_time / len(y_test),
    }


def save_outputs(pipe, best_C, fit_time, val_f1, result, x_test, y_test, n_train=None, n_val=None) -> pd.DataFrame:
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, MODEL_PATH)

    row = {
        "model": "baseline",
        "accuracy": result["accuracy"],
        "precision_macro": result["precision_macro"],
        "recall_macro": result["recall_macro"],
        "f1_macro": result["f1_macro"],
        "train_time_s": fit_time,
        "inference_time_s": result["inference_time_s"],
        "inference_ms_per_sample": result["inference_ms_per_sample"],
        "trainable_params": count_trainable_params(pipe),
        "model_size_mb": model_size_mb(MODEL_PATH),
        "best_val_f1_macro": val_f1,
        "best_C": best_C,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": len(x_test),
        "seed": SEED,
        "device": "cpu",
    }
    metrics = pd.DataFrame([row])
    metrics.to_csv(METRICS_PATH, index=False)

    proba = pipe.predict_proba(x_test)
    y_true = np.asarray(y_test)
    y_pred = np.asarray(result["y_pred"])
    # Same columns as frozen_encoder_predictions.csv and fine_tuning_predictions.csv
    pd.DataFrame(
        {
            "text": x_test,
            "true_label": y_true,
            "pred_label": y_pred,
            "true_name": [LABEL_NAMES[i] for i in y_true],
            "pred_name": [LABEL_NAMES[i] for i in y_pred],
            "confidence": proba.max(axis=1),
            "correct": y_true == y_pred,
        }
    ).to_csv(PREDICTIONS_PATH, index=False)

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "best_C": best_C,
                "val_macro_f1": val_f1,
                "seed": SEED,
                "c_grid": C_GRID,
                "tfidf": {"ngram_range": [1, 2], "max_features": 50000, "min_df": 2, "sublinear_tf": True},
                "label_names": LABEL_NAMES,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def run(splits: dict | None = None) -> pd.DataFrame:
    splits = splits or load_splits()
    x_tr, y_tr = splits["train"]
    x_val, y_val = splits["val"]
    x_te, y_te = splits["test"]

    pipe, best_C, fit_time, val_f1 = train(x_tr, y_tr, x_val, y_val)
    result = evaluate(pipe, x_te, y_te)
    metrics = save_outputs(pipe, best_C, fit_time, val_f1, result, x_te, y_te, len(x_tr), len(x_val))

    print("\n=== Baseline (test set) ===")
    print(metrics.T.to_string(header=False))
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_te, result["y_pred"]))
    print("\n" + classification_report(y_te, result["y_pred"], target_names=LABEL_NAMES, digits=4))
    return metrics


if __name__ == "__main__":
    run()
