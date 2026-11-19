"""
train.py - Fine-Tuning (Coursework Topic 6, AG News)

Pipeline:
    Text -> Tokenizer -> Pretrained Transformer (TRAINABLE) -> mean pooling
         -> Classification Head (TRAINABLE) -> 4 classes

Same architecture, tokenizer, max_len and data as the Frozen Encoder (src/models.py).
The ONLY difference: the encoder is NOT frozen, so every parameter gets gradients.

Run from the repository root:
    python -m src.train --train_size 5000 --val_size 1000 --test_size 1000   # quick test
    python -m src.train                                                      # full run

A quick test (any *_size given) writes to results/_quick_test/ so it can never
overwrite the official files in results/metrics/ and results/checkpoints/.
"""
from __future__ import annotations

import os

# Must be set before huggingface_hub is imported (Windows without symlink support).
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.utils import logging as hf_logging
from huggingface_hub.utils import logging as hub_logging

hf_logging.set_verbosity_error()   # hides the "LOAD REPORT ... UNEXPECTED" notes
hf_logging.disable_progress_bar()  # hides the "Loading weights" bar (training bars stay)
hub_logging.set_verbosity_error()  # hides the "unauthenticated requests" warning

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # lets `python src/train.py` and notebooks import src.data
    sys.path.insert(0, str(ROOT))

from src.data import DEFAULT_MODEL_NAME, LABEL_NAMES, SEED, clean_text, load_splits  # noqa: E402

MODEL_NAME = DEFAULT_MODEL_NAME  # "distilbert-base-uncased"
NO_DECAY = ("bias", "LayerNorm", "layer_norm")  # parameters that get no weight decay


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Data: tokenize per batch (same as Frozen Encoder), so Inference Time includes
# tokenization and is directly comparable with frozen_encoder_metrics.csv.
# ----------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def make_collate(tokenizer, max_len):
    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(list(texts), padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "labels": torch.tensor(labels, dtype=torch.long)}
    return collate


def make_loader(texts, labels, tokenizer, max_len, batch_size, shuffle, seed):
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(TextDataset(texts, labels), batch_size=batch_size, shuffle=shuffle,
                      collate_fn=make_collate(tokenizer, max_len), generator=gen)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class FineTuningClassifier(nn.Module):
    """DistilBERT encoder + mean pooling + (Dropout, Linear). Everything is trainable."""

    def __init__(self, model_name=MODEL_NAME, num_labels=4, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_labels))

    def pool(self, input_ids, attention_mask):
        """Mean pooling over real tokens only (padding is masked out)."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, input_ids, attention_mask):
        return self.head(self.pool(input_ids, attention_mask))


def build_model(model_name=MODEL_NAME, num_labels=len(LABEL_NAMES)):
    return FineTuningClassifier(model_name, num_labels=num_labels)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return {"total_params": total, "trainable_params": trainable, "model_size_mb": size_bytes / 1024**2}


# ----------------------------------------------------------------------------
# Training / evaluation
# ----------------------------------------------------------------------------
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def predict_logits(model, loader, device, desc="predict"):
    model.eval()
    out = []
    for batch in tqdm(loader, desc=desc, leave=False):
        out.append(model(batch["input_ids"].to(device), batch["attention_mask"].to(device)).cpu())
    return torch.cat(out)


def train_model(model, train_loader, val_loader, y_val, epochs, lr, weight_decay,
                warmup_ratio, max_grad_norm, device):
    """Full fine-tuning. Keeps the weights of the epoch with the best validation accuracy."""
    groups = [
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and not any(k in n for k in NO_DECAY)],
         "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters()
                    if p.requires_grad and any(k in n for k in NO_DECAY)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=lr)
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(warmup_ratio * total_steps), total_steps)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, best_epoch, best_state, history = -1.0, 0, None, []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        running, seen = 0.0, 0
        for batch in tqdm(train_loader, desc=f"epoch {ep}/{epochs}", leave=False):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(ids, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            running += loss.item() * len(labels)
            seen += len(labels)

        val_pred = predict_logits(model, val_loader, device, "validate").argmax(1)
        val_acc = (val_pred == y_val).float().mean().item()
        _sync(device)
        history.append({"epoch": ep, "train_loss": running / seen, "val_accuracy": val_acc,
                        "epoch_time_s": time.perf_counter() - t0})
        print(f"epoch {ep}: train_loss={running / seen:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return history, best_acc, best_epoch


def compute_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision_macro": p,
            "recall_macro": r, "f1_macro": f1}


# ----------------------------------------------------------------------------
# Checkpoint / inference
# ----------------------------------------------------------------------------
def save_checkpoint(model, path, model_name, max_len):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "model_name": model_name, "max_len": max_len,
                "labels": LABEL_NAMES, "pooling": "mean"}, path)
    return path.stat().st_size / 1024**2


def load_model(ckpt_path, device=None):
    """Rebuild the fine-tuned model. Returns (model, tokenizer, max_len)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = FineTuningClassifier(ckpt["model_name"], num_labels=len(ckpt["labels"]))
    model.load_state_dict(ckpt["model_state"])
    tokenizer = AutoTokenizer.from_pretrained(ckpt["model_name"])
    return model.to(device).eval(), tokenizer, ckpt["max_len"]


@torch.no_grad()
def predict_texts(model, tokenizer, texts, max_len=64, batch_size=64, device=None):
    """Returns (predicted label names, confidence). Applies the same clean_text as training."""
    device = device or next(model.parameters()).device
    model.eval()
    texts = [clean_text(t) for t in texts]
    names, confs = [], []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
        probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"]), dim=1)
        conf, idx = probs.max(1)
        names += [LABEL_NAMES[j] for j in idx.tolist()]
        confs += conf.tolist()
    return names, confs


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------
def run_experiment(model_name=MODEL_NAME, max_len=64, batch_size=32, eval_batch_size=64,
                   epochs=3, lr=2e-5, weight_decay=0.01, warmup_ratio=0.1, max_grad_norm=1.0,
                   train_size=None, val_size=None, test_size=None, seed=SEED,
                   out_dir=None, device=None):
    set_seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_quick = any(s is not None for s in (train_size, val_size, test_size))
    if out_dir is None:
        out_dir = ROOT / "results" / "_quick_test" if is_quick else ROOT / "results"
    out_dir = Path(out_dir)
    print(f"Device: {device} | backbone: {model_name} | output: {out_dir}")
    if is_quick:
        print("QUICK TEST: subsampled data, results go to results/_quick_test/ (not official).")

    splits = load_splits(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed)
    (tr_txt, tr_y), (va_txt, va_y), (te_txt, te_y) = splits["train"], splits["val"], splits["test"]
    print(f"train/val/test = {len(tr_txt)}/{len(va_txt)}/{len(te_txt)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader = make_loader(tr_txt, tr_y, tokenizer, max_len, batch_size, True, seed)
    val_loader = make_loader(va_txt, va_y, tokenizer, max_len, eval_batch_size, False, seed)
    test_loader = make_loader(te_txt, te_y, tokenizer, max_len, eval_batch_size, False, seed)
    y_val = torch.tensor(va_y, dtype=torch.long)
    y_test = torch.tensor(te_y, dtype=torch.long)

    model = build_model(model_name, num_labels=len(LABEL_NAMES)).to(device)
    params = count_params(model)
    print(f"Params: total={params['total_params']:,} trainable={params['trainable_params']:,} "
          f"size={params['model_size_mb']:.1f} MB")

    # --- Training time = all epochs, including the validation pass after each epoch ---
    _sync(device)
    t0 = time.perf_counter()
    history, best_val, best_epoch = train_model(model, train_loader, val_loader, y_val, epochs, lr,
                                                weight_decay, warmup_ratio, max_grad_norm, device)
    _sync(device)
    train_time = time.perf_counter() - t0

    # --- Inference on test (end-to-end: tokenize + encoder + head, batch 64 like Frozen Encoder) ---
    _sync(device)
    t0 = time.perf_counter()
    logits = predict_logits(model, test_loader, device, "test")
    _sync(device)
    infer_time = time.perf_counter() - t0

    probs = torch.softmax(logits, dim=1)
    conf, y_pred = probs.max(1)
    metrics = compute_metrics(y_test.numpy(), y_pred.numpy())

    ckpt_path = out_dir / "checkpoints" / "fine_tuning" / "fine_tuned_model.pt"
    ckpt_mb = save_checkpoint(model, ckpt_path, model_name, max_len)

    row = {"model": "fine_tuning", "backbone": model_name, **metrics,
           "train_time_s": train_time, "inference_time_s": infer_time,
           "inference_ms_per_sample": 1000 * infer_time / len(te_txt),
           "trainable_params": params["trainable_params"], "total_params": params["total_params"],
           "model_size_mb": params["model_size_mb"], "checkpoint_size_mb": ckpt_mb,
           "best_val_accuracy": best_val, "best_epoch": best_epoch,
           "n_train": len(tr_txt), "n_val": len(va_txt), "n_test": len(te_txt),
           "max_len": max_len, "epochs": epochs, "lr": lr, "batch_size": batch_size,
           "weight_decay": weight_decay, "warmup_ratio": warmup_ratio, "seed": seed,
           "device": str(device)}

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "fine_tuning_metrics.csv"
    pred_path = metrics_dir / "fine_tuning_predictions.csv"
    pd.DataFrame([row]).to_csv(metrics_path, index=False)

    # Per-sample predictions (same columns as frozen_encoder_predictions.csv)
    pd.DataFrame({
        "text": te_txt, "true_label": y_test.tolist(), "pred_label": y_pred.tolist(),
        "true_name": [LABEL_NAMES[i] for i in y_test.tolist()],
        "pred_name": [LABEL_NAMES[i] for i in y_pred.tolist()],
        "confidence": conf.tolist(), "correct": (y_pred == y_test).tolist(),
    }).to_csv(pred_path, index=False)

    print(f"Test: acc={metrics['accuracy']:.4f} macroF1={metrics['f1_macro']:.4f} | "
          f"train {train_time:.1f}s | infer {infer_time:.1f}s")
    return {"row": row, "history": history, "model": model, "tokenizer": tokenizer,
            "y_true": y_test.numpy(), "y_pred": y_pred.numpy(), "texts": te_txt,
            "metrics_path": metrics_path, "predictions_path": pred_path, "checkpoint_path": ckpt_path}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=MODEL_NAME)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--eval_batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.1)
    ap.add_argument("--train_size", type=int, default=None, help="None = full train set")
    ap.add_argument("--val_size", type=int, default=None)
    ap.add_argument("--test_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=SEED)
    run_experiment(**vars(ap.parse_args()))
