# PHÂN CÔNG COURSEWORK TOPIC 6

## News Topic Classification Using Transformer-Based Models

**Thời gian:** 


------------------------------------------------------------------------

# 1. Nội dung Coursework Topic 6

-   **Baseline:** TF-IDF + Logistic Regression
-   **Frozen Encoder:** Freeze Transformer Encoder, chỉ train
    Classification Head
-   **Fine-Tuning:** Train Transformer + Classification Head
-   **Dataset:** AG News
    -   World
    -   Sports
    -   Business
    -   Sci/Tech
-   **Evaluation:**
    -   Accuracy
    -   Precision
    -   Recall
    -   Macro F1-score
    -   Confusion Matrix
    -   Training Time
    -   Inference Time
    -   Trainable Parameters
    -   Model Size
-   **Comparison:** Baseline vs Frozen Encoder vs Fine-Tuning
-   **Error Analysis:** Phân tích các mẫu dự đoán sai

------------------------------------------------------------------------

# 2. Phân công theo NGÀY

## MAI --- REPOSITORY PACKAGE

### **Ngày**

``` text
README.md
requirements.txt
.gitignore

docs/
├── Define.md
└── Design.md
```

### Công việc

-   Hoàn thiện `README.md`
-   Kiểm tra `requirements.txt`
-   Kiểm tra `.gitignore`
-   Kiểm tra cấu trúc repository
-   Kiểm tra `docs/Define.md`
-   Kiểm tra `docs/Design.md`
-   Kiểm tra các thư mục:
    -   `data/`
    -   `src/`
    -   `notebooks/`
    -   `results/`
    -   `report/`

### Ngày

``` text
README.md
requirements.txt
.gitignore
repository structure
```

### Ngày
report/
├── report.md

Kiểm tra lần cuối để đảm bảo README khớp với cách chạy thực tế.

------------------------------------------------------------------------

# NGUYỄN CHI --- DATA PACKAGE

### Ngày

``` text
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

### Công việc

-   Load AG News
-   Kiểm tra label mapping
-   Train / Validation / Test split
-   Preprocessing
-   Data exploration
-   Chuẩn bị dữ liệu cho Baseline và Transformer

### Ngày

``` text
src/
└── predict.py

results/
└── metrics/
    └── test_predictions.csv
```

Kiểm tra Data → Prediction pipeline.

------------------------------------------------------------------------

# HUỲNH --- BASELINE PACKAGE

### Ngày

``` text
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

### Công việc

-   TF-IDF
-   Logistic Regression
-   Train Baseline
-   Evaluation Baseline
-   Lưu metrics
-   Lưu checkpoint nếu cần

### Ngày

``` text
results/
└── metrics/
    └── baseline_metrics.csv
```

Xác nhận Baseline là bản cuối và sửa lỗi nếu Integration phát hiện.

------------------------------------------------------------------------

# MY --- FROZEN ENCODER PACKAGE

### Ngày

``` text
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

### Công việc

-   Load Pretrained Transformer
-   Tokenization
-   Freeze Transformer parameters
-   Classification Head
-   Train Frozen Encoder
-   Evaluation
-   Lưu checkpoint
-   Lưu metrics
-   Ghi nhận:
    -   Accuracy
    -   Precision
    -   Recall
    -   Macro F1
    -   Training Time
    -   Inference Time
    -   Trainable Parameters
    -   Model Size

### Ngày

``` text
src/
└── models.py

results/
├── metrics/
│   └── frozen_encoder_metrics.csv
│
└── checkpoints/
    └── frozen_encoder/
```

Xác nhận Frozen Encoder chạy đúng và sửa lỗi nếu Integration phát hiện.

------------------------------------------------------------------------

# ĐỨC --- FINE-TUNING PACKAGE

### Ngày

``` text
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

### Công việc

-   Load Pretrained Transformer
-   Tokenization
-   Classification Head
-   Training pipeline
-   Fine-Tuning toàn bộ Transformer
-   Evaluation
-   Lưu checkpoint
-   Lưu metrics
-   Ghi nhận:
    -   Accuracy
    -   Precision
    -   Recall
    -   Macro F1
    -   Training Time
    -   Inference Time
    -   Trainable Parameters
    -   Model Size

### Ngày

``` text
src/
└── train.py

results/
├── metrics/
│   └── fine_tuning_metrics.csv
│
└── checkpoints/
    └── fine_tuning/
```

Xác nhận Fine-Tuning chạy đúng và sửa lỗi nếu Integration phát hiện.

------------------------------------------------------------------------

# TRÚC CHI --- EVALUATION + COMPARISON PACKAGE

### Ngày

``` text
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

### Công việc

-   Accuracy
-   Precision
-   Recall
-   Macro F1-score
-   Confusion Matrix
-   Training Time
-   Inference Time
-   Trainable Parameters
-   Model Size

### Thứ 4

``` text
docs/
└── results/
    └── error_analysis.md
```

Phân tích các mẫu dự đoán sai.

### Ngày

``` text
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

``` text
Baseline
Frozen Encoder
Fine-Tuning
```

Theo:

``` text
Accuracy
Precision
Recall
Macro F1
Training Time
Inference Time
Trainable Parameters
Model Size
```

