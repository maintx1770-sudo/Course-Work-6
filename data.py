"""
Chức năng chính:
    1. Load tập dữ liệu AG News (4 lớp: World, Sports, Business, Sci/Tech).
    2. Làm sạch văn bản (HTML entities, khoảng trắng thừa) mà vẫn giữ nguyên ngữ cảnh.
    3. Chia tập Train (108,000) / Validation (12,000) / Test (7,600) cố định với seed=42.
    4. Lưu dữ liệu đã xử lý vào `data/processed/` để toàn nhóm dùng chung.
    5. Cung cấp các hàm load dữ liệu cho Baseline (TF-IDF), Frozen Encoder và Fine-Tuning.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"

LABEL_NAMES: List[str] = ["World", "Sports", "Business", "Sci/Tech"]
LABEL2ID: Dict[str, int] = {name: idx for idx, name in enumerate(LABEL_NAMES)}
ID2LABEL: Dict[int, str] = {idx: name for idx, name in enumerate(LABEL_NAMES)}

DEFAULT_MODEL_NAME = "distilbert-base-uncased"
SEED = 42


# --------------------------------------------------------------------------- #
# 1. Text Preprocessing
# --------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    """Làm sạch văn bản tin tức AG News.

    - Giải mã HTML entities (vd: '&amp;' -> '&', '&#39;' -> "'", '#39;' -> "'").
    - Loại bỏ dấu backslash thoát ký tự thừa (vd: '\\$' -> '$').
    - Chuẩn hóa nhiều khoảng trắng/xuống dòng thành 1 dấu cách.
    - Giữ nguyên chữ hoa/thường (tokenizer 'uncased' tự hạ chữ thường khi tokenize).
    """
    if not isinstance(text, str):
        text = str(text)
    # Một số mẫu AG News bị thiếu dấu '&' trước '#39;' hoặc '#34;'
    text = re.sub(r"(?<!&)#(39|34|151|150);", r"&#\1;", text)
    text = html.unescape(text)
    text = text.replace("\\", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --------------------------------------------------------------------------- #
# 2. Dataset Preparation & Caching (data/processed/)
# --------------------------------------------------------------------------- #
def prepare_and_save_splits(
    val_size: float = 0.1,
    seed: int = SEED,
    force_reload: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Tải AG News, tiền xử lý, chia train/val/test và lưu vào data/processed/."""
    train_csv = PROCESSED_DIR / "train.csv"
    val_csv = PROCESSED_DIR / "val.csv"
    test_csv = PROCESSED_DIR / "test.csv"

    if not force_reload and train_csv.exists() and val_csv.exists() and test_csv.exists():
        return {
            "train": pd.read_csv(train_csv),
            "val": pd.read_csv(val_csv),
            "test": pd.read_csv(test_csv),
        }

    from datasets import load_dataset

    print("Đang tải bộ dữ liệu AG News từ Hugging Face...")
    ds = load_dataset("fancyzhx/ag_news")

    # Kiểm tra label mapping của dataset gốc
    if "label" in ds["train"].features and hasattr(ds["train"].features["label"], "names"):
        hf_names = ds["train"].features["label"].names
        assert hf_names == LABEL_NAMES, f"Label mapping không khớp: {hf_names} != {LABEL_NAMES}"

    raw_train_texts = [clean_text(t) for t in ds["train"]["text"]]
    raw_train_labels = list(ds["train"]["label"])

    test_texts = [clean_text(t) for t in ds["test"]["text"]]
    test_labels = list(ds["test"]["label"])

    x_tr, x_val, y_tr, y_val = train_test_split(
        raw_train_texts,
        raw_train_labels,
        test_size=val_size,
        stratify=raw_train_labels,
        random_state=seed,
    )

    splits = {
        "train": pd.DataFrame(
            {"text": x_tr, "label": y_tr, "label_name": [ID2LABEL[y] for y in y_tr]}
        ),
        "val": pd.DataFrame(
            {"text": x_val, "label": y_val, "label_name": [ID2LABEL[y] for y in y_val]}
        ),
        "test": pd.DataFrame(
            {"text": test_texts, "label": test_labels, "label_name": [ID2LABEL[y] for y in test_labels]}
        ),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, df in splits.items():
        df.to_csv(PROCESSED_DIR / f"{split_name}.csv", index=False, encoding="utf-8")

    print(
        f"Đã lưu dữ liệu đã xử lý vào {PROCESSED_DIR}: "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )
    return splits


def _subsample_df(df: pd.DataFrame, size: Optional[int], seed: int) -> pd.DataFrame:
    if size is not None and size < len(df):
        return df.sample(n=size, random_state=seed).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# 3. Loaders cho Baseline & Transformer
# --------------------------------------------------------------------------- #
def load_splits(
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    seed: int = SEED,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """Trả về {'train': (texts, labels), 'val': ..., 'test': ...} dạng list Python.
    Tương thích trực tiếp với src/baseline.py.
    """
    dfs = prepare_and_save_splits(seed=seed)
    sizes = {"train": train_size, "val": val_size, "test": test_size}
    out: Dict[str, Tuple[List[str], List[int]]] = {}
    for name, df in dfs.items():
        sub = _subsample_df(df, sizes[name], seed)
        out[name] = (sub["text"].astype(str).tolist(), sub["label"].astype(int).tolist())
    return out


def load_tensor_splits(
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    seed: int = SEED,
) -> Dict[str, Tuple[List[str], torch.Tensor]]:
    """Trả về {'train': (texts, LongTensor labels), 'val': ..., 'test': ...}.
    Tương thích trực tiếp với src/models.py (Frozen Encoder).
    """
    splits = load_splits(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed)
    return {
        name: (texts, torch.tensor(labels, dtype=torch.long))
        for name, (texts, labels) in splits.items()
    }


# --------------------------------------------------------------------------- #
# 4. PyTorch Dataset & DataLoaders cho Fine-Tuning
# --------------------------------------------------------------------------- #
class AGNewsDataset(Dataset):
    """PyTorch Dataset chứa dữ liệu đã tokenize cho mô hình Transformer."""

    def __init__(
        self,
        texts: List[str],
        labels: Optional[List[int]],
        tokenizer,
        max_len: int = 128,
    ):
        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


def create_dataloaders(
    model_name: str = DEFAULT_MODEL_NAME,
    max_len: int = 128,
    batch_size: int = 32,
    train_size: Optional[int] = None,
    val_size: Optional[int] = None,
    test_size: Optional[int] = None,
    seed: int = SEED,
    num_workers: int = 0,
) -> Tuple[Dict[str, DataLoader], AutoTokenizer]:
    """Tạo DataLoaders {'train', 'val', 'test'} và tokenizer cho Fine-Tuning."""
    splits = load_splits(train_size=train_size, val_size=val_size, test_size=test_size, seed=seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    datasets = {
        name: AGNewsDataset(texts, labels, tokenizer=tokenizer, max_len=max_len)
        for name, (texts, labels) in splits.items()
    }

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }
    return loaders, tokenizer


if __name__ == "__main__":
    splits = prepare_and_save_splits()
    for split_name, df in splits.items():
        counts = df["label_name"].value_counts().reindex(LABEL_NAMES).to_dict()
        print(f"[{split_name.upper():5s}] total={len(df):6d} | phân bố: {counts}")