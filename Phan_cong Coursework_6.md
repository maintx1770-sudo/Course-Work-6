# PHÂN CÔNG COURSEWORK TOPIC 6

## News Topic Classification Using Transformer-Based Models

**Thời gian:** 18/11/2026 – 22/11/2026

---

# 1. Nội dung Coursework Topic 6

- **Baseline:** TF-IDF + Logistic Regression
- **Frozen Encoder:** Freeze Transformer Encoder, chỉ train Classification Head
- **Fine-Tuning:** Train Transformer + Classification Head
- **Dataset:** AG News
  - World
  - Sports
  - Business
  - Sci/Tech
- **Evaluation:**
  - Accuracy
  - Precision
  - Recall
  - Macro F1-score
  - Confusion Matrix
  - Training Time
  - Inference Time
  - Trainable Parameters
  - Model Size
- **Comparison:** Baseline vs Frozen Encoder vs Fine-Tuning
- **Error Analysis:** Phân tích các mẫu dự đoán sai

---

# 2. PHÂN CÔNG THEO THÀNH VIÊN

## MAI — REPOSITORY PACKAGE

### Thứ 4 – 18/11

```text
README.md
requirements.txt
.gitignore

docs/
├── Define.md
└── Design.md
```

**Công việc:**

- Hoàn thiện `README.md`
- Kiểm tra `requirements.txt`
- Kiểm tra `.gitignore`
- Kiểm tra cấu trúc repository
- Kiểm tra `docs/Define.md`
- Kiểm tra `docs/Design.md`
- Kiểm tra các thư mục:
  - `data/`
  - `src/`
  - `notebooks/`
  - `results/`
  - `report/`

### Thứ 5 – 19/11

```text
README.md
requirements.txt
.gitignore
repository structure
```

Kiểm tra repository sau khi các thành viên bắt đầu chạy code.

### 

```text
report/report.md
```

Kiểm tra lần cuối để đảm bảo README khớp với cách chạy thực tế và cách chạy project.

---

# NGUYỄN CHI — DATA PACKAGE

### Thứ 4 – 18/11

```text
src/
└── data.py

notebooks/
└── 01_data_exploration.ipynb

docs/
└── research/
    └── ag_news.md

data/
└── processed/
```

**Công việc:**

- Load AG News
- Kiểm tra label mapping
- Train / Validation / Test split
- Preprocessing
- Data exploration
- Chuẩn bị dữ liệu cho Baseline và Transformer

### Thứ 5 – 19/11

```text
src/
└── predict.py

results/
└── metrics/
    └── test_predictions.csv
```

Kiểm tra Data → Prediction pipeline.

### Thứ 6 – 20/11

Kiểm tra lại Data pipeline và bảo đảm dữ liệu đầu vào tương thích với:

- Baseline
- Frozen Encoder
- Fine-Tuning

---

# HUYNH — BASELINE PACKAGE

### Thứ 4 – 18/11

```text
src/
└── baseline.py

notebooks/
└── 02_baseline.ipynb

docs/
└── research/
    └── tfidf_logistic_regression.md

results/
├── metrics/
│   └── baseline_metrics.csv
│
└── checkpoints/
    └── baseline/
```

**Công việc:**

- TF-IDF
- Logistic Regression
- Train Baseline
- Evaluation Baseline
- Lưu metrics
- Lưu checkpoint nếu cần

### Thứ 5 – 19/11

```text
results/
└── metrics/
    └── baseline_metrics.csv
```

Chạy Baseline và lưu metrics/checkpoint.

### Thứ 6 – 20/11

Xác nhận Baseline là bản cuối và sửa lỗi nếu có.

### Thứ 7 – 21/11

Kiểm tra Baseline trong quá trình Integration.

---

# MY — FROZEN ENCODER PACKAGE

### Thứ 4 – 18/11

```text
src/
└── models.py

notebooks/
└── 03_frozen_encoder.ipynb

docs/
└── research/
    └── frozen_encoder.md

results/
├── metrics/
│   └── frozen_encoder_metrics.csv
│
└── checkpoints/
    └── frozen_encoder/
```

**Công việc:**

- Load Pretrained Transformer
- Tokenization
- Freeze Transformer parameters
- Classification Head
- Train Frozen Encoder
- Evaluation
- Lưu checkpoint
- Lưu metrics

**Ghi nhận:**

- Accuracy
- Precision
- Recall
- Macro F1
- Training Time
- Inference Time
- Trainable Parameters
- Model Size

### Thứ 5 – 19/11

```text
src/
└── models.py

results/
├── metrics/
│   └── frozen_encoder_metrics.csv
│
└── checkpoints/
    └── frozen_encoder/
```

Chạy Frozen Encoder, lưu metrics và checkpoint.

### Thứ 6 – 20/11

Xác nhận Frozen Encoder chạy đúng và sửa lỗi nếu có.

### Thứ 7 – 21/11

Kiểm tra Frozen Encoder trong quá trình Integration.

---

# ĐỨC — FINE-TUNING PACKAGE

### Thứ 4 – 18/11

```text
src/
└── train.py

notebooks/
└── 04_fine_tuning.ipynb

docs/
└── research/
    └── fine_tuning.md

results/
├── metrics/
│   └── fine_tuning_metrics.csv
│
└── checkpoints/
    └── fine_tuning/
```

**Công việc:**

- Load Pretrained Transformer
- Tokenization
- Classification Head
- Training pipeline
- Fine-Tuning toàn bộ Transformer
- Evaluation
- Lưu checkpoint
- Lưu metrics

**Ghi nhận:**

- Accuracy
- Precision
- Recall
- Macro F1
- Training Time
- Inference Time
- Trainable Parameters
- Model Size

### Thứ 5 – 19/11

```text
src/
└── train.py

results/
├── metrics/
│   └── fine_tuning_metrics.csv
│
└── checkpoints/
    └── fine_tuning/
```

Chạy Fine-Tuning, lưu metrics và checkpoint.

### Thứ 6 – 20/11

Xác nhận Fine-Tuning chạy đúng và sửa lỗi nếu có.

### Thứ 7 – 21/11

Kiểm tra Fine-Tuning trong quá trình Integration.

---

# TRÚC CHI — EVALUATION + COMPARISON PACKAGE

### Thứ 4 – 18/11

```text
src/
└── evaluate.py

docs/
└── research/
    └── evaluation_metrics.md

results/
├── metrics/
│   └── evaluation_results.csv
│
└── figures/
    ├── confusion_matrix_baseline.png
    ├── confusion_matrix_frozen_encoder.png
    └── confusion_matrix_fine_tuning.png
```

**Công việc:**

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix
- Training Time
- Inference Time
- Trainable Parameters
- Model Size

### Thứ 6 – 20/11

```text
docs/
└── results/
    └── error_analysis.md
```

Chuẩn bị và phân tích các mẫu dự đoán sai từ kết quả của 3 mô hình.

### Thứ 7 – 21/11

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

**So sánh:**

```text
Baseline
Frozen Encoder
Fine-Tuning
```

**Theo:**

```text
Accuracy
Precision
Recall
Macro F1
Training Time
Inference Time
Trainable Parameters
Model Size
```

---

# 3. TIMELINE TỔNG

| Ngày | Giai đoạn | Mục tiêu |
|---|---|---|
| **Thứ 4 – 18/11** | Development | Hoàn thiện code chính của từng package |
| **Thứ 5 – 19/11** | Experiment | Chạy Baseline, Frozen Encoder, Fine-Tuning và lưu metrics/checkpoint |
| **Thứ 6 – 20/11** | Validation | Kiểm tra kết quả, confusion matrix, error analysis và sửa lỗi |
| **Thứ 7 – 21/11** | Integration | Ghép toàn bộ project, Comparison và hoàn thiện kết quả |
| **Chủ nhật – 22/11** | Final | Hoàn thiện report, README, kiểm tra toàn bộ repository và submit |

## Flow chung

```text
Thứ 4
Code từng package
        ↓
Thứ 5
Train + Experiment
        ↓
Thứ 6
Validate + Fix
        ↓
Thứ 7
Integration
        ↓
Comparison + Error Analysis
        ↓
Chủ nhật
Report + README + Final Check
        ↓
SUBMIT
```
