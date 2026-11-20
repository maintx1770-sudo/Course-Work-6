# Nhật ký thay đổi

Tất cả các thay đổi đáng chú ý của dự án sẽ được ghi lại trong tệp này.



### Đã thêm
- Khởi tạo cấu trúc dự án ban đầu.
- Bổ sung tài liệu DEFINE và DESIGN.
- Thiết lập dự án phân loại chủ đề tin tức sử dụng bộ dữ liệu AG News.
- Xây dựng kế hoạch phân loại chủ đề tin tức bằng mô hình Transformer.
- Xây dựng kế hoạch so sánh giữa phương pháp đóng băng Encoder và phương pháp Fine-tuning toàn bộ mô hình.

### 2026-11-20
#### Đã thêm
- Baseline TF-IDF + Logistic Regression (`src/baseline.py`, notebook 02): chạy trên dữ liệu đầy đủ, Accuracy 0,9180, Macro F1 0,9178.
- Frozen Encoder DistilBERT (`src/models.py`, notebook 03): Accuracy 0,9034, Macro F1 0,9032.
- Pipeline Fine-Tuning (`src/train.py`, notebook 04); **chưa có lần chạy chính thức**.
- Evaluation và Comparison (`src/evaluate.py`, `src/compare.py`, notebook 05).
- `src/predict.py` cho cả ba mô hình.
- Confusion matrix Baseline/Frozen Encoder, `model_comparison.png`, `evaluation_results.csv` (chưa gồm Fine-Tuning).
- `docs/results/error_analysis.md`, `docs/results/model_comparison.md` (Baseline + Frozen Encoder).
- Bộ test trong `tests/`; bản nháp `report/report.md`.

#### Đã sửa
- `src/predict.py`: Fine-Tuning nay nạp checkpoint `fine_tuned_model.pt` do `train.py` lưu (trước đó đòi thư mục `save_pretrained()` nên không chạy được).

#### Còn lại
- Chạy Fine-Tuning chính thức (`QUICK_TEST = False`), sinh lại `evaluation_results.csv`, confusion matrix và `model_comparison.png`, rồi cập nhật hai file trong `docs/results/` và `report/report.md`.

