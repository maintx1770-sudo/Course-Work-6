# PHÂN CÔNG COURSEWORK TOPIC 6

## News Topic Classification Using Transformer-Based Models

**Thời gian:** 18/11/2026 – 22/11/2026

---

# 1. NỘI DUNG COURSEWORK TOPIC 6

Coursework xây dựng hệ thống **phân loại chủ đề tin tức** trên tập dữ liệu **AG News**, gồm 4 lớp:

- World
- Sports
- Business
- Sci/Tech

Ba phương pháp được triển khai và so sánh:

1. **Baseline:** TF-IDF + Logistic Regression
2. **Frozen Encoder:** Freeze Transformer Encoder, chỉ train Classification Head
3. **Fine-Tuning:** Train toàn bộ Transformer + Classification Head

### Evaluation

Các mô hình được đánh giá theo:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix
- Training Time
- Inference Time
- Trainable Parameters
- Model Size

### Comparison

So sánh:

```text
TF-IDF + Logistic Regression
          ↓
Frozen Encoder
          ↓
Fine-Tuning
```

### Error Analysis

Phân tích các mẫu dự đoán sai để xác định:

- Các lớp thường bị nhầm lẫn
- Nguyên nhân dự đoán sai
- Đặc điểm của các mẫu khó phân loại
- Hạn chế của từng phương pháp

---

# 2. PHÂN CÔNG THEO THÀNH VIÊN

## 2.1 MAI — DATA + DOCUMENTATION PACKAGE

### Thứ 4 – 18/11

```text
src/
└── data.py

notebooks/
└── 01_data_exploration.ipynb

docs/
├── Define.md
├── Design.md
└── research/
    └── ag_news.md
```

### Công việc

- Load AG News
- Kiểm tra cấu trúc dataset
- Kiểm tra label mapping
- Train / Validation / Test split
- Preprocessing
- Data exploration
- Chuẩn hóa dữ liệu đầu vào
- Xác định format dữ liệu dùng chung cho Baseline và Transformer
- Hoàn thiện `Define.md`
- Hoàn thiện `Design.md`
- Viết tài liệu mô tả dataset

### Thứ 5 – 19/11

Kiểm tra Data Pipeline sau khi Baseline và Transformer bắt đầu sử dụng.

Đảm bảo dữ liệu đầu vào tương thích với:

```text
Baseline
Frozen Encoder
Fine-Tuning
```

### Thứ 6 – 20/11

Kiểm tra lại:

- Dataset
- Label mapping
- Data split
- Preprocessing
- Data format

Nếu có lỗi do integration thì phối hợp với các thành viên liên quan để sửa.

---

# 2.2 HUYNH — BASELINE PACKAGE

### Thứ 4 – 18/11

```text
src/
└── baseline.py

notebooks/
└── 02_baseline.ipynb

docs/
└── research/
    └── tfidf_logistic_regression.md
```

### Công việc

- Xây dựng TF-IDF
- Xây dựng Logistic Regression
- Train Baseline
- Evaluation Baseline
- Lưu kết quả metrics
- Lưu predictions nếu cần
- Chuẩn bị kết quả để Evaluation và Comparison sử dụng

### Metrics cần ghi nhận

```text
Accuracy
Precision
Recall
Macro F1-score
Training Time
Inference Time
Trainable Parameters
Model Size
```

### Thứ 5 – 19/11

Chạy Baseline trên dataset chính thức.

Lưu kết quả:

```text
results/
└── metrics/
    └── baseline_metrics.csv
```

Nếu cần checkpoint/model artifact:

```text
results/
└── checkpoints/
    └── baseline/
```

### Thứ 6 – 20/11

- Kiểm tra lại kết quả Baseline
- Xác nhận metrics hợp lệ
- Sửa lỗi nếu có
- Bàn giao kết quả cho Evaluation + Comparison

### Thứ 7 – 21/11

Phối hợp Integration để đảm bảo Baseline có thể chạy lại từ repository.

---

# 2.3 MY — TRANSFORMER MODEL + FROZEN ENCODER PACKAGE

### Thứ 4 – 18/11

```text
src/
├── models.py
└── train.py

notebooks/
└── 03_frozen_encoder.ipynb

docs/
└── research/
    └── frozen_encoder.md
```

### Công việc

- Load pretrained Transformer
- Tokenization
- Xây dựng Transformer model
- Xây dựng Classification Head
- Freeze Transformer parameters
- Train Frozen Encoder
- Evaluation Frozen Encoder
- Lưu checkpoint
- Lưu metrics

### Metrics cần ghi nhận

```text
Accuracy
Precision
Recall
Macro F1-score
Training Time
Inference Time
Trainable Parameters
Model Size
```

### Thứ 5 – 19/11

Chạy Frozen Encoder trên dataset chính thức.

Lưu:

```text
results/
├── metrics/
│   └── frozen_encoder_metrics.csv
│
└── checkpoints/
    └── frozen_encoder/
```

### Thứ 6 – 20/11

- Kiểm tra Frozen Encoder
- Kiểm tra checkpoint
- Kiểm tra metrics
- Sửa lỗi nếu có
- Bàn giao kết quả cho Evaluation + Comparison

### Thứ 7 – 21/11

Phối hợp Integration để đảm bảo Frozen Encoder chạy đúng trên repository cuối.

---

# 2.4 ĐỨC — FINE-TUNING PACKAGE

### Thứ 4 – 18/11

```text
src/
└── train.py

notebooks/
└── 04_fine_tuning.ipynb

docs/
└── research/
    └── fine_tuning.md
```

### Công việc

- Sử dụng pretrained Transformer
- Tokenization
- Classification Head
- Xây dựng training pipeline
- Fine-Tuning toàn bộ Transformer
- Evaluation Fine-Tuning
- Lưu checkpoint
- Lưu metrics

### Metrics cần ghi nhận

```text
Accuracy
Precision
Recall
Macro F1-score
Training Time
Inference Time
Trainable Parameters
Model Size
```

### Thứ 5 – 19/11

Chạy Fine-Tuning trên dataset chính thức.

Lưu:

```text
results/
├── metrics/
│   └── fine_tuning_metrics.csv
│
└── checkpoints/
    └── fine_tuning/
```

### Thứ 6 – 20/11

- Kiểm tra Fine-Tuning
- Kiểm tra checkpoint
- Kiểm tra metrics
- Sửa lỗi nếu có
- Bàn giao kết quả cho Evaluation + Comparison

### Thứ 7 – 21/11

Phối hợp Integration để đảm bảo Fine-Tuning chạy đúng trên repository cuối.

---

# 2.5 NGUYỄN CHI — EVALUATION + COMPARISON PACKAGE

### Thứ 4 – 18/11

```text
src/
├── evaluate.py
└── compare.py

notebooks/
└── 05_comparison.ipynb

docs/
└── research/
    └── evaluation_metrics.md
```

### Công việc

Xây dựng phần đánh giá thống nhất cho cả 3 phương pháp:

```text
Baseline
Frozen Encoder
Fine-Tuning
```

### Evaluation

Tính:

```text
Accuracy
Precision
Recall
Macro F1-score
Confusion Matrix
Training Time
Inference Time
Trainable Parameters
Model Size
```

### Thứ 5 – 19/11

Nhận kết quả từ:

```text
HUYNH
MY
ĐỨC
```

Chuẩn hóa metrics về cùng format.

Lưu:

```text
results/
└── metrics/
    └── evaluation_results.csv
```

Tạo confusion matrix:

```text
results/
└── figures/
    ├── confusion_matrix_baseline.png
    ├── confusion_matrix_frozen_encoder.png
    └── confusion_matrix_fine_tuning.png
```

### Thứ 6 – 20/11

Chuẩn bị Error Analysis:

```text
docs/
└── results/
    └── error_analysis.md
```

Phân tích:

- Các mẫu dự đoán sai
- Các cặp class thường bị nhầm
- Nguyên nhân có thể gây nhầm lẫn
- Những trường hợp khó phân loại

### Thứ 7 – 21/11

Hoàn thiện Comparison:

```text
src/
└── compare.py

notebooks/
└── 05_comparison.ipynb

docs/
└── results/
    └── model_comparison.md

results/
└── figures/
    └── model_comparison.png
```

So sánh:

| Model | Accuracy | Precision | Recall | Macro F1 | Training Time | Inference Time | Trainable Params | Model Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | | | | | | | | |
| Frozen Encoder | | | | | | | | |
| Fine-Tuning | | | | | | | | |

---

# 2.6 TRÚC CHI — PREDICTION + ERROR ANALYSIS + INTEGRATION

### Thứ 4 – 18/11

```text
src/
└── predict.py
```

### Công việc

- Xây dựng chức năng prediction
- Nhận text mới từ người dùng
- Preprocess/tokenize input
- Load model
- Dự đoán class
- Trả về label và prediction

Flow:

```text
Input News
    ↓
Preprocessing
    ↓
Tokenizer
    ↓
Model
    ↓
Prediction
    ↓
World / Sports / Business / Sci-Tech
```

### Thứ 5 – 19/11

Kiểm tra prediction pipeline với:

```text
Baseline
Frozen Encoder
Fine-Tuning
```

Chuẩn bị dữ liệu prediction phục vụ Error Analysis nếu cần.

### Thứ 6 – 20/11

Phối hợp với Nguyễn Chi để hoàn thiện:

```text
docs/
└── results/
    └── error_analysis.md
```

Kiểm tra:

- Mẫu dự đoán sai
- Class bị nhầm
- Prediction output
- Các trường hợp khó phân loại

### Thứ 7 – 21/11

Phụ trách Integration:

- Kiểm tra toàn bộ pipeline
- Kiểm tra `predict.py`
- Kiểm tra các script chính
- Kiểm tra notebooks
- Kiểm tra kết quả
- Phát hiện lỗi giữa các package
- Phối hợp sửa lỗi trước khi submit

---

# 3. MAI — FINAL REPOSITORY + REPORT

Ngoài phần Data + Documentation, Mai phụ trách kiểm tra repository và tài liệu cuối.

### Thứ 6 – 20/11

Kiểm tra:

```text
README.md
requirements.txt
.gitignore
docs/
data/
src/
notebooks/
results/
report/
```

Đảm bảo:

- README khớp với implementation
- Cấu trúc repository khớp với code
- Hướng dẫn chạy project chính xác
- Không có file thừa hoặc file sai vị trí

### Thứ 7 – 21/11

Kiểm tra:

```text
README
Define
Design
Code
Notebook
Results
Report
```

Đảm bảo nội dung thống nhất.

### Chủ nhật – 22/11

Hoàn thiện:

```text
report/
└── report.md
```

Kiểm tra lần cuối trước khi submit.

---


# 3 DEPENDENCY GIỮA CÁC THÀNH VIÊN

```text
                 MAI
          Data + Documentation
                  │
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
     HUYNH                  MY
    Baseline          Frozen Encoder
        │                   │
        │                   ↓
        │                  ĐỨC
        │              Fine-Tuning
        │                   │
        └─────────┬─────────┘
                  ↓
             NGUYỄN CHI
       Evaluation + Comparison
                  │
                  ↓
              TRÚC CHI
       Prediction + Integration
                  │
                  ↓
                MAI
        Final Report + Submit
```

---

# 4. TIMELINE TỔNG

| Ngày | Giai đoạn | Công việc chính |
|---|---|---|
| **Thứ 4 – 18/11** | Development | Hoàn thiện Data, Baseline, Frozen Encoder, Fine-Tuning, Prediction |
| **Thứ 5 – 19/11** | Experiment | Train 3 phương pháp và lưu metrics/checkpoint |
| **Thứ 6 – 20/11** | Validation | Kiểm tra metrics, confusion matrix, prediction, error analysis |
| **Thứ 7 – 21/11** | Integration | Comparison, Integration, kiểm tra toàn bộ repository |
| **Chủ nhật – 22/11** | Final | Report, README, Final Check và Submit |

---

