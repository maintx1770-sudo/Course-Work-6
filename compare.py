"""Comparison utilities for model result CSVs."""
from pathlib import Path
import pandas as pd
from src.evaluate import plot_confusion_matrix

COLS_TO_KEEP = [
    "model",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "train_time_s",
    "inference_time_s",
    "trainable_params",
    "model_size_mb",
]

RENAME_TO_DISPLAY = {
    "model": "Model",
    "accuracy": "Accuracy",
    "precision_macro": "Precision",
    "recall_macro": "Recall",
    "f1_macro": "Macro F1-score",
    "train_time_s": "Training Time (s)",
    "inference_time_s": "Inference Time (s)",
    "trainable_params": "Trainable Parameters",
    "model_size_mb": "Model Size (MB)",
}


def compare(metric_paths, output="results/metrics/evaluation_results.csv"):
    """Chuẩn hóa về 9 cột chung và gộp kết quả các mô hình thành bảng evaluation_results.csv."""
    frames = []
    for p in metric_paths:
        path = Path(p)
        if path.exists():
            df = pd.read_csv(path)
            if set(COLS_TO_KEEP).issubset(df.columns):
                df = df[COLS_TO_KEEP].rename(columns=RENAME_TO_DISPLAY)
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No metric CSVs found.")

    eval_df = pd.concat(frames, ignore_index=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output, index=False)
    return eval_df


def generate_all_confusion_matrices(pred_configs):
    """Sinh hàng loạt ảnh Confusion Matrix từ danh sách (title, pred_csv_path, fig_save_path)."""
    for title, pred_path, fig_path in pred_configs:
        path = Path(pred_path)
        if path.exists():
            df_pred = pd.read_csv(path)
            plot_confusion_matrix(df_pred["true_label"], df_pred["pred_label"], title=title, save_path=fig_path)