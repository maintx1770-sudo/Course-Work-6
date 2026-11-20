"""
models.py - Frozen Encoder (Coursework Topic 6, AG News)

Pipeline:
    Text -> Tokenizer -> Pretrained Transformer (FROZEN) -> mean pooling
         -> Classification Head (TRAINED) -> 4 classes

Vì encoder bị đóng băng nên đầu ra của nó không đổi giữa các epoch.
Ta trích đặc trưng MỘT LẦN rồi train head trên đặc trưng đã cache
-> nhanh hơn nhiều so với chạy lại Transformer mỗi epoch, kết quả giống hệt.

Chạy từ thư mục gốc repo:
    python src/models.py --train_size 5000          # chạy thử nhanh
    python src/models.py                            # chạy full (Thứ 5)
"""
import argparse
import copy
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer
from src.data import load_tensor_splits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LABELS = ["World", "Sports", "Business", "Sci/Tech"]  # 0,1,2,3 theo AG News
MODEL_NAME = "distilbert-base-uncased"


# ----------------------------------------------------------------------------
# Data (tạm thời tự load để chạy độc lập; khi Nguyễn Chi xong src/data.py
# có thể thay hàm này bằng hàm của data.py)
# ----------------------------------------------------------------------------
data = load_tensor_splits(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed):
    """Trả về dict {'train','val','test'} -> (list[str] texts, LongTensor labels)."""
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/ag_news")
    split = ds["train"].train_test_split(test_size=0.1, seed=seed, stratify_by_column="label")
    parts = {"train": split["train"], "val": split["test"], "test": ds["test"]}
    sizes = {"train": train_size, "val": val_size, "test": test_size}

    out = {}
    for name, d in parts.items():
        n = sizes[name]
        if n is not None and n < len(d):
            d = d.shuffle(seed=seed).select(range(n))
        out[name] = (list(d["text"]), torch.tensor(d["label"], dtype=torch.long))
    return out


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class FrozenEncoderClassifier(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_labels=4, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        for p in self.encoder.parameters():  # FREEZE
            p.requires_grad = False
        self.encoder.eval()
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_labels))

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()  # encoder luôn ở eval (tắt dropout)
        return self

    @torch.no_grad()
    def encode(self, input_ids, attention_mask):
        """Mean pooling trên các token thật (bỏ padding)."""
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)

    def forward(self, input_ids, attention_mask):
        return self.head(self.encode(input_ids, attention_mask))


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    size_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return {"total_params": total, "trainable_params": trainable, "model_size_mb": size_bytes / 1024**2}


# ----------------------------------------------------------------------------
# Feature extraction + head training
# ----------------------------------------------------------------------------
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def extract_features(model, tokenizer, texts, max_len, batch_size, device, desc="extract"):
    model.eval()
    feats = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
        feats.append(model.encode(enc["input_ids"], enc["attention_mask"]).cpu())
    return torch.cat(feats)


def train_head(model, X_tr, y_tr, X_va, y_va, epochs, lr, batch_size, device):
    head = model.head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_acc, best_state, history = -1.0, None, []
    for ep in range(1, epochs + 1):
        head.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(head(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(xb)
        head.eval()
        with torch.no_grad():
            val_pred = head(X_va.to(device)).argmax(1).cpu()
        val_acc = (val_pred == y_va).float().mean().item()
        history.append({"epoch": ep, "train_loss": running / len(X_tr), "val_accuracy": val_acc})
        if val_acc > best_acc:
            best_acc, best_state = val_acc, copy.deepcopy(head.state_dict())
    head.load_state_dict(best_state)  # lấy epoch tốt nhất theo validation
    return history, best_acc


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
    torch.save({"head_state": model.head.state_dict(), "model_name": model_name,
                "max_len": max_len, "labels": LABELS, "pooling": "mean"}, path)
    return path.stat().st_size / 1024**2


def load_model(ckpt_path, device=None):
    """Dựng lại model từ checkpoint (encoder tải từ HuggingFace, head từ file)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = FrozenEncoderClassifier(ckpt["model_name"], num_labels=len(ckpt["labels"]))
    model.head.load_state_dict(ckpt["head_state"])
    tokenizer = AutoTokenizer.from_pretrained(ckpt["model_name"])
    return model.to(device).eval(), tokenizer, ckpt["max_len"]


@torch.no_grad()
def predict_texts(model, tokenizer, texts, max_len=64, batch_size=64, device=None):
    """Trả về (nhãn dự đoán dạng tên, xác suất lớp dự đoán) cho list text."""
    device = device or next(model.head.parameters()).device
    model.eval()
    names, confs = [], []
    for i in range(0, len(texts), batch_size):
        enc = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to(device)
        probs = torch.softmax(model(enc["input_ids"], enc["attention_mask"]), dim=1)
        conf, idx = probs.max(1)
        names += [LABELS[j] for j in idx.tolist()]
        confs += conf.tolist()
    return names, confs


# ----------------------------------------------------------------------------
# Experiment
# ----------------------------------------------------------------------------
def run_experiment(model_name=MODEL_NAME, max_len=64, batch_size=64, head_epochs=20, lr=1e-3,
                   head_batch_size=256, train_size=None, val_size=None, test_size=None,
                   seed=42, out_dir=None, device=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir) if out_dir else ROOT / "results"
    print(f"Device: {device} | backbone: {model_name}")

    data = load_ag_news(train_size, val_size, test_size, seed)
    (tr_txt, y_tr), (va_txt, y_va), (te_txt, y_te) = data["train"], data["val"], data["test"]
    print(f"train/val/test = {len(tr_txt)}/{len(va_txt)}/{len(te_txt)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FrozenEncoderClassifier(model_name, num_labels=len(LABELS)).to(device)
    params = count_params(model)
    print(f"Params: total={params['total_params']:,} trainable={params['trainable_params']:,} "
          f"size={params['model_size_mb']:.1f} MB")

    # --- Training = trích đặc trưng (1 lần) + train head ---
    _sync(device); t0 = time.perf_counter()
    X_tr = extract_features(model, tokenizer, tr_txt, max_len, batch_size, device, "train feats")
    X_va = extract_features(model, tokenizer, va_txt, max_len, batch_size, device, "val feats")
    _sync(device); extract_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    history, best_val = train_head(model, X_tr, y_tr, X_va, y_va, head_epochs, lr, head_batch_size, device)
    _sync(device); head_time = time.perf_counter() - t0

    # --- Inference trên test (đo end-to-end: encoder + head) ---
    model.eval()
    _sync(device); t0 = time.perf_counter()
    X_te = extract_features(model, tokenizer, te_txt, max_len, batch_size, device, "test feats")
    with torch.no_grad():
        logits = model.head(X_te.to(device))
    _sync(device); infer_time = time.perf_counter() - t0

    probs = torch.softmax(logits, dim=1).cpu()
    conf, y_pred = probs.max(1)
    metrics = compute_metrics(y_te.numpy(), y_pred.numpy())

    ckpt_mb = save_checkpoint(model, out_dir / "checkpoints" / "frozen_encoder" / "frozen_encoder_head.pt",
                              model_name, max_len)

    row = {"model": "frozen_encoder", "backbone": model_name, **metrics,
           "train_time_s": extract_time + head_time, "feature_extraction_time_s": extract_time,
           "head_train_time_s": head_time, "inference_time_s": infer_time,
           "inference_ms_per_sample": 1000 * infer_time / len(te_txt),
           "trainable_params": params["trainable_params"], "total_params": params["total_params"],
           "model_size_mb": params["model_size_mb"], "checkpoint_size_mb": ckpt_mb,
           "best_val_accuracy": best_val, "n_train": len(tr_txt), "n_val": len(va_txt),
           "n_test": len(te_txt), "max_len": max_len, "head_epochs": head_epochs, "lr": lr, "seed": seed}

    metrics_dir = out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(metrics_dir / "frozen_encoder_metrics.csv", index=False)

    # Dự đoán từng mẫu: Trúc Chi dùng cho confusion matrix + error analysis
    pd.DataFrame({
        "text": te_txt, "true_label": y_te.tolist(), "pred_label": y_pred.tolist(),
        "true_name": [LABELS[i] for i in y_te.tolist()],
        "pred_name": [LABELS[i] for i in y_pred.tolist()],
        "confidence": conf.tolist(), "correct": (y_pred == y_te).tolist(),
    }).to_csv(metrics_dir / "frozen_encoder_predictions.csv", index=False)

    print(f"Test: acc={metrics['accuracy']:.4f} macroF1={metrics['f1_macro']:.4f} | "
          f"train {row['train_time_s']:.1f}s | infer {infer_time:.1f}s")
    return {"row": row, "history": history, "model": model, "tokenizer": tokenizer,
            "y_true": y_te.numpy(), "y_pred": y_pred.numpy(), "texts": te_txt}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=MODEL_NAME)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--head_epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--train_size", type=int, default=None, help="None = dùng toàn bộ train")
    ap.add_argument("--val_size", type=int, default=None)
    ap.add_argument("--test_size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    run_experiment(**vars(a))
