# PHÂN CÔNG COURSEWORK TOPIC 6

## News Topic Classification Using Transformer-Based Models

**Thời gian:** Thứ 2 → Thứ 5\
**Nguyên tắc:** Mỗi thành viên nhận một package hoàn chỉnh gồm code +
notebook + docs + output liên quan, cố gắng hoàn thành trong một buổi
làm việc. Thứ 5 chỉ dùng để tích hợp, chạy lại và nộp.

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

# 2. Phân công theo package

## MAI --- REPOSITORY PACKAGE

### Thứ 2

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

### Thứ 5

``` text
README.md
requirements.txt
.gitignore
repository structure
```

Kiểm tra lần cuối để đảm bảo README khớp với cách chạy thực tế.

------------------------------------------------------------------------

# NGUYỄN CHI --- DATA PACKAGE

### Thứ 2

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

### Thứ 5

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

### Thứ 2

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

### Thứ 5

``` text
results/
└── metrics/
    └── baseline_metrics.csv
```

Xác nhận Baseline là bản cuối và sửa lỗi nếu Integration phát hiện.

------------------------------------------------------------------------

# MY --- FROZEN ENCODER PACKAGE

### Thứ 3

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

### Thứ 5

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

### Thứ 3

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

### Thứ 5

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

### Thứ 3

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

### Thứ 5

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

------------------------------------------------------------------------

# 3. Cấu trúc file tổng

``` text
Course-Work-6/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── Define.md
│   ├── Design.md
│   │
│   ├── research/
│   │   ├── ag_news.md
│   │   ├── tfidf_logistic_regression.md
│   │   ├── frozen_encoder.md
│   │   ├── fine_tuning.md
│   │   └── evaluation_metrics.md
│   │
│   └── results/
│       ├── error_analysis.md
│       └── model_comparison.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data.py
│   ├── baseline.py
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   ├── compare.py
│   └── predict.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_frozen_encoder.ipynb
│   ├── 04_fine_tuning.ipynb
│   └── 05_comparison.ipynb
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── checkpoints/
│
└── report/
    ├── report.pdf
    └── presentation.pdf
```

------------------------------------------------------------------------

# 4. Dependency

``` text
NGUYỄN CHI
Data
│
├──────────────┬──────────────┐
▼              ▼              ▼
HUỲNH          MY             ĐỨC
Baseline       Frozen         Fine-Tuning
               Encoder
│              │              │
└──────────────┼──────────────┘
               ▼
          TRÚC CHI
          Evaluation
               │
               ▼
          Comparison
               │
               ▼
             MAI
       Final Repository Check
               │
               ▼
            CẢ NHÓM
         Full Integration
               │
               ▼
            SUBMIT
```

------------------------------------------------------------------------

# 5. Thứ 5 --- Final Integration

``` text
1. git pull
2. Kiểm tra structure
3. Chạy Data pipeline
4. Chạy Baseline
5. Chạy Frozen Encoder
6. Chạy Fine-Tuning
7. Chạy Evaluation
8. Chạy Prediction
9. Chạy Comparison
10. Kiểm tra metrics
11. Kiểm tra notebooks
12. Kiểm tra README
13. Sửa lỗi
14. git push
15. Kiểm tra GitHub
16. Submit
```

### Final checklist

``` text
[ ] README.md
[ ] requirements.txt
[ ] .gitignore
[ ] Define.md
[ ] Design.md
[ ] data.py
[ ] baseline.py
[ ] models.py
[ ] train.py
[ ] evaluate.py
[ ] compare.py
[ ] predict.py
[ ] 01_data_exploration.ipynb
[ ] 02_baseline.ipynb
[ ] 03_frozen_encoder.ipynb
[ ] 04_fine_tuning.ipynb
[ ] 05_comparison.ipynb
[ ] Baseline metrics
[ ] Frozen Encoder metrics
[ ] Fine-Tuning metrics
[ ] Confusion Matrix
[ ] Error Analysis
[ ] Model Comparison
[ ] Checkpoints
[ ] Full pipeline chạy được
[ ] GitHub cập nhật
[ ] Submit
```
