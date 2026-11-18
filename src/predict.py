"""Dự đoán chủ đề tin tức AG News từ văn bản mới.

Chạy tại thư mục gốc Course-Work-6:
    python -m src.predict --model baseline --text "The team won the final match."
    python -m src.predict --model frozen_encoder --text "New chip technology was announced."
    python -m src.predict --model fine_tuning --checkpoint results/checkpoints/fine_tuning --text "Stocks rose today."
    python -m src.predict --model baseline          # nhập liên tục từ bàn phím

Lưu ý: cần huấn luyện và lưu checkpoint trước khi dự đoán.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.data import LABEL_NAMES, clean_text


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = {
    "baseline": ROOT / "results/checkpoints/baseline/baseline_pipeline.joblib",
    "frozen_encoder": ROOT / "results/checkpoints/frozen_encoder/frozen_encoder_head.pt",
    "fine_tuning": ROOT / "results/checkpoints/fine_tuning",
}


class NewsPredictor:
    """Tải mô hình một lần và dự đoán nhiều văn bản tin tức.

    model_type: 'baseline', 'frozen_encoder' hoặc 'fine_tuning'.
    Fine-Tuning yêu cầu thư mục model đã lưu bằng save_pretrained().
    """

    def __init__(
        self,
        model_type: str = "baseline",
        checkpoint: str | Path | None = None,
        device: str = "auto",
        max_length: int = 128,
    ) -> None:
        if model_type not in DEFAULT_CHECKPOINTS:
            raise ValueError(
                f"Model không hợp lệ: {model_type}. "
                f"Chọn một trong: {', '.join(DEFAULT_CHECKPOINTS)}"
            )
        if max_length < 2:
            raise ValueError("max_length phải từ 2 trở lên.")

        self.model_type = model_type
        self.checkpoint = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINTS[model_type]
        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Máy không hỗ trợ CUDA. Hãy chạy với --device cpu.")
        self.max_length = max_length
        self.tokenizer: Any = None

        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint: {self.checkpoint}\n"
                "Hãy chạy huấn luyện mô hình tương ứng trước, "
                "hoặc dùng --checkpoint để chỉ rõ đường dẫn."
            )

        if model_type == "baseline":
            import joblib

            # Pipeline đã gồm TfidfVectorizer + LogisticRegression.
            self.model = joblib.load(self.checkpoint)

        elif model_type == "frozen_encoder":
            # Tận dụng đúng hàm load_model của thành viên phụ trách models.py.
            from src.models import load_model

            self.model, self.tokenizer, self.max_length = load_model(
                self.checkpoint, device=self.device
            )
            self.model.eval()

        else:  # fine_tuning
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if not self.checkpoint.is_dir() or not (self.checkpoint / "config.json").exists():
                raise FileNotFoundError(
                    "Fine-Tuning cần thư mục đã lưu bằng save_pretrained() "
                    "(có config.json và trọng số mô hình). "
                    "Hãy dùng --checkpoint trỏ tới đúng thư mục đó."
                )
            self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint)
            self.model = self.model.to(self.device).eval()

    def predict(self, text: str) -> dict[str, Any]:
        """Trả về ID, tên lớp, confidence và xác suất cả bốn lớp."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Vui lòng nhập văn bản tin tức không rỗng.")

        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Văn bản rỗng sau khi tiền xử lý.")

        if self.model_type == "baseline":
            # TF-IDF tự biến đổi văn bản bên trong sklearn Pipeline.
            class_id = int(self.model.predict([cleaned])[0])
            probabilities = self.model.predict_proba([cleaned])[0]
            classes = [int(value) for value in self.model.classes_]
            scores = {class_index: float(prob) for class_index, prob in zip(classes, probabilities)}
        else:
            # Transformer: tokenizer -> input_ids/attention_mask -> model -> softmax.
            inputs = self.tokenizer(
                cleaned,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                if self.model_type == "frozen_encoder":
                    logits = self.model(inputs["input_ids"], inputs["attention_mask"])
                else:
                    logits = self.model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)[0].cpu().tolist()
            scores = {i: float(prob) for i, prob in enumerate(probabilities)}
            class_id = max(scores, key=scores.get)

        if set(scores) != set(range(len(LABEL_NAMES))):
            raise ValueError("Số lớp của checkpoint không khớp với 4 nhãn AG News.")

        return {
            "text": text,
            "cleaned_text": cleaned,
            "model": self.model_type,
            "prediction": class_id,
            "label": LABEL_NAMES[class_id],
            "confidence": scores[class_id],
            "probabilities": {LABEL_NAMES[i]: scores[i] for i in range(len(LABEL_NAMES))},
        }


def predict_text(
    text: str,
    model_type: str = "baseline",
    checkpoint: str | Path | None = None,
    device: str = "auto",
) -> dict[str, Any]:
    """API đơn giản cho notebook hoặc module khác (mỗi lần gọi sẽ load model)."""
    predictor = NewsPredictor(model_type, checkpoint=checkpoint, device=device)
    return predictor.predict(text)


def print_result(result: dict[str, Any], show_probs: bool = False) -> None:
    print("\n--- KẾT QUẢ DỰ ĐOÁN ---")
    print(f"Văn bản     : {result['cleaned_text']}")
    print(f"Mô hình     : {result['model']}")
    print(f"Nhãn (ID)   : {result['prediction']}")
    print(f"Chủ đề      : {result['label']}")
    print(f"Độ tin cậy  : {result['confidence']:.2%}")
    if show_probs:
        print("Xác suất từng lớp:")
        for label, probability in result["probabilities"].items():
            print(f"  {label:<9}: {probability:.2%}")


def main() -> None:
    parser = argparse.ArgumentParser(description="News Topic Prediction - AG News")
    parser.add_argument(
        "--model", choices=list(DEFAULT_CHECKPOINTS), default="baseline",
        help="Chọn mô hình dự đoán (mặc định: baseline)",
    )
    parser.add_argument("--text", type=str, help="Văn bản tin tức muốn phân loại")
    parser.add_argument("--checkpoint", type=str, help="Đường dẫn checkpoint (nếu khác mặc định)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-length", type=int, default=128, help="Số token tối đa")
    parser.add_argument("--show-probs", action="store_true", help="Hiện xác suất cả 4 lớp")
    args = parser.parse_args()

    try:
        predictor = NewsPredictor(
            args.model, checkpoint=args.checkpoint, device=args.device,
            max_length=args.max_length,
        )
        if args.text is not None:
            print_result(predictor.predict(args.text), args.show_probs)
            return

        print("Nhập tin tức tiếng Anh để phân loại. Gõ 'exit' để thoát.")
        while True:
            try:
                text = input("News > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nĐã thoát.")
                break
            if text.lower() in {"exit", "quit"}:
                print("Đã thoát.")
                break
            if not text:
                print("Vui lòng nhập nội dung tin tức.")
                continue
            print_result(predictor.predict(text), args.show_probs)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(status=1, message=f"Lỗi: {exc}\n")


if __name__ == "__main__":
    main()
