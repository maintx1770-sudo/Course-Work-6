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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "metrics" / "baseline_metrics.csv"
PREDICTIONS_PATH = ROOT / "results" / "metrics" / "baseline_test_predictions.csv"
CKPT_DIR = ROOT / "results" / "checkpoints" / "baseline"
MODEL_PATH = CKPT_DIR / "baseline_pipeline.joblib"
CONFIG_PATH = CKPT_DIR / "config.json"

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]  # AG News ids 0..3
SEED = 42
C_GRID = [1.0, 5.0, 10.0]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_splits(val_size: float = 0.1, seed: int = SEED) -> dict:
    """Load AG News and return {"train": (texts, labels), "val": ..., "test": ...}.

    TEMPORARY: loads directly from Hugging Face. Replace the body of this
    function with the loader from src/data.py once it is available, so that
    all models use the exact same train/val/test split.
    """
    from datasets import load_dataset

    ds = load_dataset("ag_news")
    x = list(ds["train"]["text"])
    y = list(ds["train"]["label"])
    x_tr, x_val, y_tr, y_val = train_test_split(
        x, y, test_size=val_size, stratify=y, random_state=seed
    )
    return {
        "train": (x_tr, y_tr),
        "val": (x_val, y_val),
        "test": (list(ds["test"]["text"]), list(ds["test"]["label"])),
    }


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


def save_outputs(pipe, best_C, fit_time, val_f1, result, x_test, y_test) -> pd.DataFrame:
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
    }
    metrics = pd.DataFrame([row])
    metrics.to_csv(METRICS_PATH, index=False)

    proba = pipe.predict_proba(x_test)
    pd.DataFrame(
        {
            "text": x_test,
            "true_label": y_test,
            "pred_label": result["y_pred"],
            "confidence": proba.max(axis=1),
        }
    ).to_csv(PREDICTIONS_PATH, index=False)

    CONFIG_PATH.write_text(
        json.dumps(
            {
                "best_C": best_C,
                "val_macro_f1": val_f1,
                "seed": SEED,
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
    metrics = save_outputs(pipe, best_C, fit_time, val_f1, result, x_te, y_te)

    print("\n=== Baseline (test set) ===")
    print(metrics.T.to_string(header=False))
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_te, result["y_pred"]))
    print("\n" + classification_report(y_te, result["y_pred"], target_names=LABEL_NAMES, digits=4))
    return metrics


if __name__ == "__main__":
    run()
