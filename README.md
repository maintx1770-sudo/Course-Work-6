# Coursework_Topic 6
## News Topic Classification Using Transformer-Based Models

## 1. Giới thiệu

Course Work 6 xây dựng hệ thống **phân loại chủ đề tin tức (News Topic Classification)** trên tập dữ liệu **AG News**, sử dụng các phương pháp Machine Learning và Transformer-based Deep Learning.

Hệ thống phân loại bài báo thành 4 nhóm:

- **World**
- **Sports**
- **Business**
- **Sci/Tech**

Project thực hiện và so sánh ba phương pháp:

1. **TF-IDF + Logistic Regression** – Baseline truyền thống.
2. **Frozen Encoder** – đóng băng Transformer Encoder và chỉ huấn luyện Classification Head.
3. **Fine-Tuning** – cho phép cập nhật cả Transformer Encoder và Classification Head.

Mô hình Transformer được sử dụng là:

```text
distilbert-base-uncased
```

Mục tiêu của project không chỉ là so sánh Accuracy mà còn phân tích:

- Classification Performance
- Training Cost
- Inference Cost
- Trainable Parameters
- Model Size
- Error Behavior

Toàn bộ quá trình được thực hiện theo flow:

```text
Problem Analysis
      ↓
Requirements
      ↓
EDA
      ↓
Decision
      ↓
Model Selection
      ↓
Training
      ↓
Training Monitoring
      ↓
Evaluation
      ↓
Error Analysis
      ↓
Model Comparison
      ↓
Final Technical Decision
```

---

# 2. Vấn đề và câu hỏi nghiên cứu

Các nền tảng tin tức trực tuyến phải xử lý một lượng lớn bài báo mỗi ngày. Việc phân loại thủ công tốn thời gian và nhân lực.

Project xây dựng hệ thống có khả năng tự động dự đoán chủ đề của một bài báo dựa trên nội dung văn bản.

### Câu hỏi nghiên cứu

> Việc sử dụng Transformer theo hai cách tiếp cận Frozen Encoder và Fine-Tuning khác nhau như thế nào về hiệu năng phân loại và chi phí tính toán so với Baseline TF-IDF + Logistic Regression trong bài toán phân loại chủ đề AG News?

---

# 3. Mục tiêu

Project hướng đến các mục tiêu:

- Xây dựng hệ thống tự động phân loại tin tức.
- Áp dụng Transformer cho bài toán Text Classification.
- Xây dựng Baseline bằng TF-IDF + Logistic Regression.
- Xây dựng và thực nghiệm Frozen Encoder.
- Xây dựng và thực nghiệm Fine-Tuning.
- So sánh hiệu quả của ba phương pháp.
- Đánh giá mô hình bằng các chỉ số định lượng.
- Theo dõi quá trình training và validation.
- Phân tích các lỗi dự đoán.
- Phân tích trade-off giữa hiệu năng và chi phí tính toán.
- Đưa ra quyết định lựa chọn mô hình dựa trên bằng chứng thực nghiệm.

---

# 4. Dataset

## 4.1. AG News

Project sử dụng dataset **AG News** thông qua Hugging Face Datasets.

Dataset gồm 4 lớp:

| Label | Category |
|---:|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

Dataset ban đầu gồm:

- Training: 120,000 mẫu
- Test: 7,600 mẫu

Training set được chia thành:

- Training: 108,000 mẫu
- Validation: 12,000 mẫu
- Test: 7,600 mẫu

Test set được giữ riêng và chỉ sử dụng cho đánh giá cuối cùng.

---

## 4.2. Input

Input của hệ thống là văn bản tin tức.

Pipeline hiện tại sử dụng:

```text
text
label
```

Trong đó:

- `text`: nội dung bài báo.
- `label`: nhãn chủ đề.

---

## 4.3. Output

Hệ thống trả về:

- Predicted Category
- Predicted Label
- Probability của các lớp khi model hỗ trợ probability output.

Ví dụ:

```text
Input:
Manchester United won the match 3-1.

Output:
Predicted Category: Sports
Predicted Label: 1
```

---

# 5. Phương pháp

## 5.1. Baseline – TF-IDF + Logistic Regression

Pipeline:

```text
Text
 ↓
TF-IDF
 ↓
Logistic Regression
 ↓
Prediction
```

Baseline được sử dụng làm mô hình tham chiếu để xác định mức hiệu năng có thể đạt được bằng phương pháp Machine Learning truyền thống.

---

## 5.2. Frozen Encoder

Frozen Encoder sử dụng pretrained **DistilBERT**.

Pipeline:

```text
Input Text
    ↓
Tokenizer
    ↓
DistilBERT Encoder
    ↓
Mean Pooling
    ↓
Dropout
    ↓
Linear Classification Head
    ↓
Prediction
```

Trong phương pháp này:

- DistilBERT được sử dụng làm pretrained Encoder.
- Các tham số của Transformer được **freeze**.
- Chỉ Classification Head được huấn luyện.
- Mean Pooling được sử dụng để tổng hợp `last_hidden_state`.
- Classification Head thực hiện phân loại thành 4 lớp.

---

## 5.3. Fine-Tuning

Fine-Tuning sử dụng cùng pretrained model:

```text
distilbert-base-uncased
```

Pipeline:

```text
Input Text
    ↓
Tokenizer
    ↓
DistilBERT Encoder
    ↓
Mean Pooling
    ↓
Dropout
    ↓
Linear Classification Head
    ↓
Prediction
```

Trong phương pháp này:

- Transformer Encoder được phép cập nhật tham số.
- Classification Head được huấn luyện.
- Toàn bộ mô hình được tối ưu trên AG News.

---

# 6. EDA và quyết định kỹ thuật

EDA được thực hiện trước khi huấn luyện để kiểm tra chất lượng dữ liệu và hỗ trợ lựa chọn cấu hình.

Các nội dung kiểm tra:

- Missing values
- Empty text
- Duplicate
- Class distribution
- Text length
- Token length

Kết quả EDA:

| Kiểm tra | Kết quả |
|---|---:|
| Train | 108,000 |
| Validation | 12,000 |
| Test | 7,600 |
| Missing | 0 |
| Empty | 0 |
| Duplicate – Train | 62 |
| Duplicate – Validation | 3 |
| Duplicate – Test | 0 |

## 6.1. Token coverage

Kết quả khảo sát:

| Max Length | Token Coverage |
|---:|---:|
| 64 | 87.41% |
| 128 | 99.42% |
| 256 | 99.97% |

### Decision

Project sử dụng:

```text
max_length = 64
```

Đây là quyết định dựa trên trade-off giữa:

- Chi phí tính toán.
- Mức độ giữ lại thông tin của văn bản.

`64` có chi phí thấp hơn nhưng có mức truncation cao hơn `128` và `256`.

Do đó, `64` được xem là **cấu hình thực nghiệm được lựa chọn**, không phải kết luận rằng đây là giá trị tối ưu tuyệt đối.

Nếu thay đổi `max_length`, các thí nghiệm Transformer cần được chạy lại để đảm bảo tính công bằng.

---

# 7. Training Monitoring

Training được theo dõi trên validation set theo từng epoch.

Các chỉ số quan sát:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Validation Macro F1

Mục tiêu:

1. Phát hiện overfitting.
2. Xác định epoch tốt nhất.
3. Chọn checkpoint.
4. Đánh giá độ ổn định của quá trình training.
5. Hỗ trợ quyết định về training configuration.

Quy trình:

```text
Training
   ↓
Training Metrics
   ↓
Validation Metrics
   ↓
Best Validation Checkpoint
   ↓
Final Test Evaluation
```

**Test set không được sử dụng để lựa chọn checkpoint hoặc hyperparameter.**

Đối với Fine-Tuning, checkpoint tốt nhất được xác định tại **epoch 2** dựa trên validation performance.

---

# 8. Evaluation

## 8.1. Classification Metrics

Các mô hình được đánh giá bằng:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

## 8.2. Computational Metrics

Project ghi nhận:

- Training Time
- Inference Time
- Trainable Parameters
- Model Size

Các chỉ số này được sử dụng để phân tích trade-off giữa performance và computational cost.

---

# 9. Kết quả thực nghiệm

Các kết quả dưới đây là **kết quả thực nghiệm chính thức** trên test set 7,600 mẫu.

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 91.96% | 91.94% |
| Frozen Encoder | 90.29% | 90.28% |
| **Fine-Tuning** | **94.22%** | **94.23%** |

Fine-Tuning đạt:

- Accuracy: **94.22%**
- Macro F1: **94.23%**
- Macro Precision: **94.24%**
- Macro Recall: **94.22%**

Fine-Tuning sử dụng:

| Parameter | Value |
|---|---|
| Model | `distilbert-base-uncased` |
| Learning Rate | `2e-5` |
| Batch Size | `32` |
| Epochs | `3` |
| Max Length | `64` |
| Optimizer | AdamW |
| Device | CUDA |
| Best Epoch | 2 |
| Random Seed | 42 |

Thời gian huấn luyện Fine-Tuning khoảng:

```text
2353 seconds ≈ 39.2 minutes
```

---

# 10. Error Analysis

Error Analysis được thực hiện sau Evaluation.

Quy trình:

```text
Confusion Matrix
      ↓
Class Pair
      ↓
Wrong Samples
      ↓
Confidence
      ↓
Content Analysis
      ↓
Cause Classification
      ↓
Decision
```

Các nguyên nhân được xem xét:

### Model-driven

Nội dung có tín hiệu rõ cho nhãn thật nhưng mô hình chọn nhãn khác.

### Topic overlap

Nội dung có đặc điểm của nhiều chủ đề.

Đặc biệt cần chú ý:

```text
Business ↔ Sci/Tech
```

### Short / Low-context

Văn bản quá ngắn hoặc thiếu thông tin cần thiết.

### Possible Label Ambiguity / Noise

Nhãn gốc có thể không hoàn toàn phản ánh nội dung.

Error Analysis được sử dụng để:

- xác định pattern lỗi;
- giải thích nguyên nhân;
- đánh giá hạn chế của mô hình;
- hỗ trợ lựa chọn mô hình cuối cùng.

Chi tiết được trình bày tại:

```text
docs/results/error_analysis.md
```

---

# 11. Model Comparison

Ba phương pháp được so sánh:

```text
TF-IDF + Logistic Regression
          vs
Frozen Encoder
          vs
Fine-Tuning
```

Việc so sánh dựa trên:

### Classification Performance

- Accuracy
- Precision
- Recall
- Macro F1

### Computational Cost

- Training Time
- Inference Time
- Trainable Parameters
- Model Size

### Error Behavior

- Confusion Matrix
- Class Pair
- Representative Wrong Samples
- Error Pattern

Chi tiết:

```text
docs/results/model_comparison.md
```

---

# 12. Final Technical Decision

Kết quả chính thức cho thấy:

```text
Fine-Tuning     94.22%
Baseline        91.96%
Frozen Encoder  90.29%
```

Fine-Tuning đạt hiệu năng cao nhất.

### Decision

> **Fine-Tuning được lựa chọn là phương pháp có hiệu năng phân loại tốt nhất trong phạm vi thực nghiệm của project.**

Cơ sở quyết định:

1. Accuracy cao nhất: **94.22%**.
2. Macro F1 cao nhất: **94.23%**.
3. Cao hơn Baseline và Frozen Encoder.
4. Việc cập nhật Transformer giúp mô hình thích nghi tốt hơn với nhiệm vụ phân loại AG News.

Tuy nhiên, Fine-Tuning có chi phí tính toán cao hơn Frozen Encoder.

Do đó, quyết định cuối cùng được dựa trên:

```text
Classification Performance
        +
Computational Cost
        +
Error Analysis
        +
Experimental Evidence
```

Nếu ưu tiên hiệu năng phân loại, Fine-Tuning là lựa chọn phù hợp.

Nếu ưu tiên chi phí tính toán thấp hơn, Baseline hoặc Frozen Encoder có thể được cân nhắc tùy yêu cầu triển khai.

---

# 13. Cấu trúc thư mục

```text
Course-Work-6/
│
├── README.md
├── CHANGELOG.md
├── Phan_cong Coursework_6.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── Define.md
│   ├── Design.md
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
├── tests/
│
└── report/
    ├── report.md
    ├── report.pdf
    └── presentation.pdf
```

---

# 14. Mô tả các thư mục

## `docs/`

Chứa tài liệu phân tích, nghiên cứu, thiết kế và kết quả.

- `Define.md`: Xác định vấn đề, yêu cầu, input/output, constraints và tiêu chí thành công.
- `Design.md`: Mô tả kiến trúc, pipeline, modules, training, evaluation và decision process.
- `research/`: Tài liệu nghiên cứu.
- `results/`: Phân tích kết quả.

## `data/`

Chứa dữ liệu:

```text
data/
├── raw/
└── processed/
```

Dataset không bắt buộc phải commit toàn bộ vào repository.

## `src/`

Chứa source code chính:

| File | Vai trò |
|---|---|
| `data.py` | Load và chuẩn bị dữ liệu |
| `baseline.py` | TF-IDF + Logistic Regression |
| `models.py` | Transformer và Classification Head |
| `train.py` | Training pipeline |
| `evaluate.py` | Evaluation |
| `compare.py` | Model comparison |
| `predict.py` | Prediction |

## `notebooks/`

| Notebook | Vai trò |
|---|---|
| `01_data_exploration.ipynb` | EDA |
| `02_baseline.ipynb` | Baseline |
| `03_frozen_encoder.ipynb` | Frozen Encoder |
| `04_fine_tuning.ipynb` | Fine-Tuning |
| `05_comparison.ipynb` | Evaluation + Comparison |

## `results/`

Chứa:

- Metrics
- Predictions
- Confusion Matrix
- Training curves
- Checkpoints

## `tests/`

Chứa các test cho:

- Data loading
- Model loading
- Training
- Evaluation
- Integration

## `report/`

Chứa:

- Report
- PDF report
- Presentation

---

# 15. Cài đặt

## Clone repository

```bash
git clone https://github.com/maintx1770-sudo/Course-Work-6.git
cd Course-Work-6
```

## Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

# 16. Quy trình chạy project

Các notebook được thực hiện theo thứ tự:

### Step 1 – Data Exploration

```text
01_data_exploration.ipynb
```

Thực hiện:

- Load data
- Data quality check
- Class distribution
- Duplicate analysis
- Token length analysis
- EDA-based decisions

### Step 2 – Baseline

```text
02_baseline.ipynb
```

Thực hiện:

- TF-IDF
- Logistic Regression
- Validation
- Test evaluation
- Save metrics

### Step 3 – Frozen Encoder

```text
03_frozen_encoder.ipynb
```

Thực hiện:

- Load DistilBERT
- Freeze Encoder
- Mean Pooling
- Train Classification Head
- Validation monitoring
- Test evaluation

### Step 4 – Fine-Tuning

```text
04_fine_tuning.ipynb
```

Thực hiện:

- Load DistilBERT
- Train Encoder + Classification Head
- Validation monitoring
- Best checkpoint selection
- Test evaluation

### Step 5 – Comparison

```text
05_comparison.ipynb
```

Thực hiện:

- Load official metrics
- Compare three models
- Generate confusion matrices
- Generate comparison figures
- Perform model comparison
- Support final technical decision

---

# 17. Quy trình ra quyết định

Project không chỉ thực hiện:

```text
Run Code → Get Result
```

mà thực hiện:

```text
EDA
 ↓
Observation
 ↓
Analysis
 ↓
Decision
 ↓
Training
 ↓
Validation Monitoring
 ↓
Evaluation
 ↓
Error Analysis
 ↓
Comparison
 ↓
Final Decision
```

### EDA → Decision

Dữ liệu và token coverage được phân tích trước khi lựa chọn preprocessing và `max_length`.

### Training Monitoring → Decision

Validation performance được sử dụng để lựa chọn checkpoint.

### Evaluation → Interpretation

Metrics và Confusion Matrix được phân tích thay vì chỉ ghi nhận giá trị.

### Error Analysis → Decision

Các mẫu sai và class pair được phân tích để xác định pattern lỗi.

### Comparison → Final Decision

Mô hình được lựa chọn dựa trên:

- Performance
- Computational Cost
- Error Behavior

---

# 18. Kết quả đầu ra

Các kết quả chính được lưu tại:

```text
results/
```

Bao gồm:

- Evaluation metrics
- Test predictions
- Confusion Matrix
- Model comparison figures
- Training results
- Model checkpoints

Tài liệu phân tích:

```text
docs/results/error_analysis.md
docs/results/model_comparison.md
```

---

# 19. Tài liệu

### Project Definition

- [Define](docs/Define.md)
- [Design](docs/Design.md)

### Research

- [AG News](docs/research/ag_news.md)
- [TF-IDF + Logistic Regression](docs/research/tfidf_logistic_regression.md)
- [Frozen Encoder](docs/research/frozen_encoder.md)
- [Fine-Tuning](docs/research/fine_tuning.md)
- [Evaluation Metrics](docs/research/evaluation_metrics.md)

### Results

- [Error Analysis](docs/results/error_analysis.md)
- [Model Comparison](docs/results/model_comparison.md)

---

# 20. Thành viên

| Họ và tên | MSSV | Vai trò |
|---|---|---|
| Mai | ... | Dataset + EDA + Documentation |
| Huynh | ... | Baseline |
| My | ... | Transformer Models + Training |
| Đức | ... | Evaluation + Comparison |
| Trúc Chi | ... | Prediction + Error Analysis |
| Nguyễn Chi | ... | Testing + Integration + Finalization |

---

# 21. Kết luận

Project xây dựng hệ thống **News Topic Classification** sử dụng dataset **AG News** và pretrained model **DistilBERT**.

Ba phương pháp được thực nghiệm:

1. **TF-IDF + Logistic Regression**
2. **Frozen Encoder**
3. **Fine-Tuning**

Kết quả chính thức cho thấy **Fine-Tuning đạt hiệu năng tốt nhất** trong phạm vi thực nghiệm hiện tại với:

```text
Accuracy = 94.22%
Macro F1 = 94.23%
```

Tuy nhiên, Fine-Tuning yêu cầu chi phí tính toán cao hơn. Vì vậy, lựa chọn mô hình cuối cùng phải xem xét đồng thời **classification performance, computational cost và error behavior**.

Project được thiết kế theo hướng:

```text
Problem
  ↓
EDA
  ↓
Decision
  ↓
Experiment
  ↓
Monitoring
  ↓
Evaluation
  ↓
Error Analysis
  ↓
Comparison
  ↓
Technical Decision
```

Qua đó, kết quả cuối cùng không chỉ trả lời **“mô hình nào có Accuracy cao nhất?”** mà còn trả lời **“vì sao chọn mô hình đó và bằng chứng thực nghiệm nào hỗ trợ quyết định?”**
