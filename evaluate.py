"""Unified evaluation helpers."""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def evaluate_torch_model(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"].cpu().numpy()
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            logits = model(**inputs)["logits"]
            pred = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(labels)
            y_pred.extend(pred)
    inference_time = time.perf_counter() - start
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "Macro F1-score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
        "Inference Time": inference_time,
        "y_true": np.array(y_true),
        "y_pred": np.array(y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str | Path, label_names=LABEL_NAMES):
    """Vẽ và lưu biểu đồ Confusion Matrix 4x4."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title(f"Confusion Matrix - {title}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    return cm


def get_misclassified_samples(pred_csv_path: str | Path, label_names=LABEL_NAMES) -> pd.DataFrame:
    """Trích xuất các mẫu bị dự đoán sai phục vụ phân tích lỗi (Error Analysis)."""
    df = pd.read_csv(pred_csv_path)
    errors = df[df["true_label"] != df["pred_label"]].copy()
    errors["true_name"] = errors["true_label"].map(lambda i: label_names[int(i)])
    errors["pred_name"] = errors["pred_label"].map(lambda i: label_names[int(i)])
    return errors