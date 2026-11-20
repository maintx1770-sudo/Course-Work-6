# Nhật ký thay đổi

Tất cả các thay đổi đáng chú ý của dự án **Coursework Topic 6 – News Topic Classification Using Transformer-Based Models** được ghi lại trong tệp này.

---

## Thông tin dự án

- **Thời gian thực hiện:** 18/11/2026 – 22/11/2026
- **Dataset:** AG News
- **Các phương pháp:**
  - TF-IDF + Logistic Regression
  - Frozen Encoder
  - Fine-Tuning
- **Mô hình Transformer:** DistilBERT
- **Các tiêu chí đánh giá:**
  - Accuracy
  - Precision
  - Recall
  - Macro F1-score
  - Confusion Matrix
  - Training Time
  - Inference Time
  - Trainable Parameters
  - Model Size

---

# 2026-11-18 — Development

## MAI — Data + Documentation Package

### Đã thực hiện

- Xây dựng Data Pipeline cho AG News.
- Load và kiểm tra cấu trúc dataset.
- Kiểm tra label mapping gồm:
  - World
  - Sports
  - Business
  - Sci/Tech
- Chuẩn bị Train / Validation / Test split.
- Thực hiện preprocessing và chuẩn hóa dữ liệu đầu vào.
- Hoàn thiện Data Exploration.
- Hoàn thiện tài liệu:
  - `src/data.py`
  - `notebooks/01_data_exploration.ipynb`
  - `docs/Define.md`
  - `docs/Design.md`
  - `docs/research/ag_news.md`

---

## HUYNH — Baseline Package

### Đã thực hiện

- Xây dựng Baseline sử dụng:
  - TF-IDF
  - Logistic Regression
- Xây dựng notebook:
  - `notebooks/02_baseline.ipynb`
- Hoàn thiện tài liệu:
  - `docs/research/tfidf_logistic_regression.md`
- Chạy Baseline trên dataset chính thức.
- Lưu metrics và prediction phục vụ Evaluation và Comparison.

### Kết quả

- Accuracy: **91.80%**
- Macro F1-score: **91.78%**

---

## MY — Transformer Model + Frozen Encoder Package

### Đã thực hiện

- Sử dụng pretrained DistilBERT.
- Xây dựng Transformer model và Classification Head.
- Thực hiện tokenization.
- Freeze Transformer parameters.
- Xây dựng và huấn luyện Frozen Encoder.
- Thực hiện Evaluation cho Frozen Encoder.
- Lưu checkpoint và metrics.
- Hoàn thiện:
  - `src/models.py`
  - `notebooks/03_frozen_encoder.ipynb`
  - `docs/research/frozen_encoder.md`

### Kết quả

- Accuracy: **90.34%**
- Macro F1-score: **90.32%**

---

## ĐỨC — Fine-Tuning Package

### Đã thực hiện

- Xây dựng training pipeline cho Fine-Tuning.
- Sử dụng pretrained DistilBERT.
- Xây dựng Classification Head.
- Chuẩn bị pipeline Fine-Tuning toàn bộ Transformer.
- Hoàn thiện:
  - `src/train.py`
  - `notebooks/04_fine_tuning.ipynb`
  - `docs/research/fine_tuning.md`

### Trạng thái

- Pipeline Fine-Tuning đã được xây dựng.
- Fine-Tuning chính thức trên toàn bộ dataset **chưa hoàn thành**.
- Kết quả Quick Test **không được sử dụng** làm kết quả chính thức của mô hình.
- Chưa có Accuracy, Precision, Recall và Macro F1 chính thức cho Fine-Tuning.

---

## NGUYỄN CHI — Evaluation + Comparison Package

### Đã thực hiện

- Xây dựng hệ thống Evaluation chung cho các phương pháp.
- Hoàn thiện:
  - `src/evaluate.py`
  - `src/compare.py`
  - `notebooks/05_comparison.ipynb`
  - `docs/research/evaluation_metrics.md`
- Chuẩn hóa cách tính metrics cho các phương pháp.
- Tạo kết quả Evaluation cho Baseline và Frozen Encoder.
- Tạo confusion matrix cho:
  - Baseline
  - Frozen Encoder
- Chuẩn bị Comparison cho các phương pháp đã hoàn thành.

### Kết quả đã tạo

- `results/metrics/evaluation_results.csv`
- `results/figures/confusion_matrix_baseline.png`
- `results/figures/confusion_matrix_frozen_encoder.png`
- `results/figures/model_comparison.png`

### Trạng thái

- Comparison hiện tại chỉ bao gồm các mô hình đã có kết quả chính thức.
- Fine-Tuning chưa được đưa vào kết quả Comparison chính thức do chưa hoàn thành thực nghiệm.

---

## TRÚC CHI — Prediction + Error Analysis + Integration

### Đã thực hiện

- Xây dựng Prediction Pipeline.
- Hoàn thiện:
  - `src/predict.py`
- Chuẩn bị Prediction Pipeline cho:
  - Baseline
  - Frozen Encoder
  - Fine-Tuning
- Tham gia kiểm tra Integration giữa các thành phần.
- Phối hợp chuẩn bị Error Analysis.
- Hoàn thiện:
  - `docs/results/error_analysis.md`

### Cập nhật kỹ thuật

- `src/predict.py` đã được chuẩn bị để hỗ trợ prediction từ checkpoint Fine-Tuning sau khi mô hình được huấn luyện chính thức.

---

# 2026-11-19 — Experiment & Integration

## Data Pipeline

- Kiểm tra lại Dataset.
- Kiểm tra label mapping.
- Kiểm tra Train / Validation / Test split.
- Kiểm tra preprocessing.
- Đảm bảo format dữ liệu tương thích với:
  - Baseline
  - Frozen Encoder
  - Fine-Tuning

## Baseline

- Chạy Baseline trên dataset chính thức.
- Kiểm tra metrics.
- Kiểm tra prediction.
- Chuẩn bị kết quả cho Evaluation và Comparison.

## Frozen Encoder

- Chạy Frozen Encoder trên dataset chính thức.
- Kiểm tra checkpoint.
- Kiểm tra metrics.
- Chuẩn bị kết quả cho Evaluation và Comparison.

## Fine-Tuning

- Chuẩn bị pipeline chạy Fine-Tuning trên dataset chính thức.
- Kiểm tra khả năng thực thi của training pipeline.
- Fine-Tuning chính thức trên toàn bộ dataset **chưa hoàn thành**.

## Evaluation & Comparison

- Chuẩn hóa kết quả từ các phương pháp đã hoàn thành.
- Tạo Evaluation Results.
- Tạo Confusion Matrix.
- Chuẩn bị Model Comparison.

---

# 2026-11-20 — Validation & Documentation

## Đã hoàn thiện

- Kiểm tra kết quả Baseline.
- Kiểm tra kết quả Frozen Encoder.
- Kiểm tra Prediction Pipeline.
- Kiểm tra Evaluation Pipeline.
- Hoàn thiện Error Analysis cho các mô hình đã có kết quả.
- Hoàn thiện Model Comparison cho Baseline và Frozen Encoder.
- Cập nhật:
  - `docs/results/error_analysis.md`
  - `docs/results/model_comparison.md`
- Chuẩn bị `report/report.md`.

## Kết quả hiện tại

| Model | Accuracy | Macro F1 | Trạng thái |
|---|---:|---:|---|
| TF-IDF + Logistic Regression | **91.80%** | **91.78%** | Hoàn thành |
| Frozen Encoder | **90.34%** | **90.32%** | Hoàn thành |
| Fine-Tuning | Chưa có | Chưa có | Đang chờ chạy chính thức |

---

# Còn lại

## Fine-Tuning

Cần hoàn thành:

- Chạy Fine-Tuning chính thức với `QUICK_TEST = False`.
- Sử dụng toàn bộ dataset.
- Lưu:
  - Fine-Tuning metrics
  - Fine-Tuning predictions
  - Fine-Tuning checkpoint
- Tạo:
  - Fine-Tuning confusion matrix
- Cập nhật:
  - `results/metrics/evaluation_results.csv`
  - `results/figures/model_comparison.png`
  - `docs/results/model_comparison.md`
  - `docs/results/error_analysis.md`
  - `report/report.md`

## Final Integration

Sau khi Fine-Tuning hoàn thành:

1. Kiểm tra toàn bộ Data Pipeline.
2. Kiểm tra Baseline.
3. Kiểm tra Frozen Encoder.
4. Kiểm tra Fine-Tuning.
5. Kiểm tra metrics của cả ba phương pháp.
6. Kiểm tra confusion matrices.
7. Cập nhật Model Comparison.
8. Kiểm tra Error Analysis.
9. Kiểm tra Prediction Pipeline.
10. Kiểm tra README.
11. Kiểm tra Report.
12. Kiểm tra repository trước khi submit.

---

# 2026-11-22 — Final Submission

## Trạng thái dự kiến

Dự kiến hoàn thiện toàn bộ Coursework theo flow:

```text
Data
   ↓
Baseline
   ↓
Frozen Encoder
   ↓
Fine-Tuning
   ↓
Evaluation
   ↓
Comparison
   ↓
Error Analysis
   ↓
Prediction
   ↓
Final Report
   ↓
Final Repository
   ↓
Submit
```

### Các điều kiện cần hoàn thành trước khi Submit

- Fine-Tuning chính thức có kết quả.
- Có metrics cho cả ba phương pháp.
- Có confusion matrix cho các mô hình.
- Có Model Comparison đầy đủ.
- Có Error Analysis tương ứng.
- Prediction Pipeline hoạt động.
- `report/report.md` được cập nhật.
- README được kiểm tra và cập nhật.
- Repository không chứa file tạm hoặc file không cần thiết.
- Kiểm tra lại toàn bộ cấu trúc project trước khi nộp.

### Trạng thái hiện tại

- Baseline: **Hoàn thành**
- Frozen Encoder: **Hoàn thành**
- Fine-Tuning: **Chưa hoàn thành**
- Evaluation: **Hoàn thành cho các mô hình đã có kết quả**
- Comparison: **Đang chờ kết quả Fine-Tuning để hoàn thiện**
- Error Analysis: **Đã thực hiện cho các mô hình đã có kết quả**
- Prediction: **Đã chuẩn bị**
- Final Report: **Đang hoàn thiện**
- Final Integration: **Chưa hoàn thành**
- Final Submission: **Chưa thực hiện**
