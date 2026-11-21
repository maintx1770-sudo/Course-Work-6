"""Comparison utilities for model result CSVs."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

MODEL_LABEL_MAP = {
    "baseline": "Baseline (TF-IDF + LR)",
    "frozen_encoder": "Frozen Encoder (DistilBERT)",
    "fine_tuning": "Fine-Tuning (DistilBERT)",
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
            if "Model" in df.columns:
                df["Model"] = df["Model"].replace(MODEL_LABEL_MAP)
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
            plot_confusion_matrix(
                df_pred["true_label"],
                df_pred["pred_label"],
                title=title,
                save_path=fig_path,
            )


def plot_model_comparison(eval_df, save_path="results/figures/model_comparison.png"):
    """Vẽ biểu đồ 2 panel so sánh chất lượng phân loại và thời gian huấn luyện."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    models = eval_df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.25

    # Panel 1: Accuracy, Precision, Macro F1
    b1 = axes[0].bar(x - width, eval_df["Accuracy"], width, label="Accuracy", color="#4C72B0")
    b2 = axes[0].bar(x, eval_df["Precision"], width, label="Precision", color="#55A868")
    b3 = axes[0].bar(x + width, eval_df["Macro F1-score"], width, label="Macro F1", color="#C44E52")
    axes[0].set_ylim(0.85, 0.98)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=12, ha="right")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Classification Performance on AG News Test Set")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    for bars in (b1, b2, b3):
        for bar in bars:
            h = bar.get_height()
            axes[0].annotate(
                f"{h:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Panel 2: Training Time (s)
    colors = ["#4C72B0", "#55A868", "#C44E52"][: len(models)]
    t_bars = axes[1].bar(x, eval_df["Training Time (s)"], width=0.5, color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=12, ha="right")
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Training Time Comparison (s)")
    axes[1].grid(axis="y", linestyle="--", alpha=0.5)

    for bar in t_bars:
        h = bar.get_height()
        axes[1].annotate(
            f"{h:.1f}s",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    metric_files = [
        "results/metrics/baseline_metrics.csv",
        "results/metrics/frozen_encoder_metrics.csv",
        "results/metrics/fine_tuning_metrics.csv",
    ]
    df_results = compare(metric_files)
    print(df_results.to_string(index=False))

    plot_model_comparison(df_results, save_path="results/figures/model_comparison.png")

    pred_configs = [
        ("Baseline (TF-IDF + LR)", "results/metrics/baseline_predictions.csv", "results/figures/confusion_matrix_baseline.png"),
        ("Frozen Encoder (DistilBERT)", "results/metrics/frozen_encoder_predictions.csv", "results/figures/confusion_matrix_frozen_encoder.png"),
        ("Fine-Tuning (DistilBERT)", "results/metrics/fine_tuning_predictions.csv", "results/figures/confusion_matrix_fine_tuning.png"),
    ]
    generate_all_confusion_matrices(pred_configs)
    print("\nĐã xuất evaluation_results.csv và 4 biểu đồ vào results/figures/!")