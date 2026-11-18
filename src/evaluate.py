"""Unified evaluation helpers."""
from __future__ import annotations
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate_torch_model(model, loader, device):
    model.eval(); y_true=[]; y_pred=[]; start=time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            labels=batch["labels"].cpu().numpy(); inputs={k:v.to(device) for k,v in batch.items() if k != "labels"}
            logits=model(**inputs)["logits"]
            pred=logits.argmax(dim=1).cpu().numpy()
            y_true.extend(labels); y_pred.extend(pred)
    inference_time=time.perf_counter()-start
    return {
        "Accuracy": accuracy_score(y_true,y_pred),
        "Precision": precision_score(y_true,y_pred,average="macro",zero_division=0),
        "Recall": recall_score(y_true,y_pred,average="macro",zero_division=0),
        "Macro F1-score": f1_score(y_true,y_pred,average="macro",zero_division=0),
        "Confusion Matrix": confusion_matrix(y_true,y_pred),
        "Inference Time": inference_time,
        "y_true": np.array(y_true), "y_pred": np.array(y_pred)
    }
