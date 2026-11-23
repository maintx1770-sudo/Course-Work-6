# THIẾT KẾ (DESIGN)

**Đề tài:** Phân loại chủ đề tin tức AG News bằng Transformer: so sánh Frozen Encoder và Fine-Tuning.

---

# 1. Kiến trúc hệ thống

Hệ thống sử dụng tập dữ liệu **AG News** để xây dựng mô hình phân loại tin tức thành 4 nhóm chủ đề:

- World
- Sports
- Business
- Sci/Tech

Dự án triển khai và so sánh ba phương pháp:

1. **TF-IDF + Logistic Regression (Baseline)**
2. **Frozen Transformer Encoder**
3. **Fine-Tuned Transformer**

Luồng xử lý tổng quát:

```text
                    AG News Dataset
                           |
                           v
                    Data Preprocessing
                           |
                           v
                       Tokenization
                           |
              +------------+------------+
              |                         |
              v                         v
      TF-IDF + Logistic          DistilBERT Encoder
        Regression                     |
              |                +--------+--------+
              |                |                 |
              |             Frozen           Fine-Tuned
              |                |                 |
              |                v                 v
              |          Mean Pooling      Mean Pooling
              |                |                 |
              |                v                 v
              |        Classification Head  Classification Head
              |                |                 |
              +----------------+-----------------+
                               |
                               v
                        Prediction 4 Classes
                               |
                               v
                     Evaluation & Error Analysis
                               |
                               v
                       Model Comparison
                               |
                               v
                        Final Decision
```

---

## 1.1. Baseline: TF-IDF + Logistic Regression

Baseline không sử dụng Transformer và được xây dựng làm mô hình tham chiếu.

```text
Văn bản đầu vào
      |
      v
    TF-IDF
      |
      v
Logistic Regression
      |
      v
   Dự đoán
```

Baseline giúp xác định mức hiệu năng có thể đạt được bằng phương pháp Machine Learning truyền thống trước khi sử dụng Transformer.

---

## 1.2. Frozen Encoder

Frozen Encoder sử dụng Transformer đã được pretrained nhưng **đóng băng toàn bộ tham số của Encoder**.

Chỉ Classification Head được cập nhật trong quá trình huấn luyện.

```text
Văn bản đầu vào
      |
      v
  Tokenizer
      |
      v
DistilBERT Encoder
      |
      | Frozen
      v
Mean Pooling
      |
      v
Classification Head
      |
      v
Dự đoán 4 lớp
```

Mục tiêu của phương pháp này là đánh giá khả năng sử dụng các biểu diễn đã được Transformer pretrained học sẵn mà không cần cập nhật toàn bộ mô hình.

---

## 1.3. Fine-Tuning

Fine-Tuning sử dụng cùng backbone `distilbert-base-uncased` với Frozen Encoder nhưng cho phép cập nhật tham số của:

- Transformer Encoder
- Classification Head

```text
Văn bản đầu vào
      |
      v
  Tokenizer
      |
      v
DistilBERT Encoder
      |
      | Trainable
      v
Mean Pooling
      |
      v
Classification Head
      |
      v
Dự đoán 4 lớp
```

Mục tiêu là đánh giá liệu việc điều chỉnh toàn bộ Transformer trên AG News có giúp cải thiện hiệu năng phân loại hay không.

---

## 1.4. Cấu trúc mô hình

### Encoder

```text
distilbert-base-uncased
```

### Tokenizer

Tokenizer tương ứng với:

```text
distilbert-base-uncased
```

### Biểu diễn văn bản

Sử dụng **Mean Pooling** trên `last_hidden_state`.

`attention_mask` được sử dụng để loại bỏ ảnh hưởng của các token padding khi tính trung bình.

### Classification Head

Classification Head gồm:

- Dropout
- Linear layer

Linear layer ánh xạ vector biểu diễn của Encoder thành 4 logits tương ứng với:

```text
World
Sports
Business
Sci/Tech
```

### Loss Function

Sử dụng:

```text
Cross-Entropy Loss
```

---

# 2. Yêu cầu phân tích và ra quyết định

Thiết kế hệ thống không chỉ tập trung vào việc xây dựng và chạy mô hình mà phải thể hiện được mối liên hệ:

```text
Quan sát
   ↓
Phân tích
   ↓
Quyết định
   ↓
Thực nghiệm
   ↓
Đánh giá
```

Các giai đoạn chính:

| Giai đoạn | Bằng chứng cần thu thập | Quyết định cần trả lời |
|---|---|---|
| EDA | Missing, empty, duplicate, class distribution, token length | Dữ liệu có vấn đề gì? Preprocessing như thế nào? Chọn input và `max_len` ra sao? |
| Training Monitoring | Train loss/accuracy, validation loss/accuracy theo epoch | Có overfitting không? Epoch/checkpoint nào phù hợp? |
| Evaluation | Accuracy, Precision, Recall, Macro F1, Confusion Matrix, thời gian, params, size | Mô hình mạnh/yếu ở đâu? Chi phí tính toán như thế nào? |
| Error Analysis | Wrong samples, class pairs, confidence, nội dung mẫu | Lỗi có pattern gì? Do mô hình hay do dữ liệu? |
| Comparison | Performance + computational cost + error behavior | Mô hình nào phù hợp nhất và tại sao? |

Các quyết định kỹ thuật phải dựa trên kết quả thực nghiệm của chính project.

**Kết quả Quick Test chỉ được sử dụng để kiểm tra pipeline, không được sử dụng làm kết quả nghiên cứu chính thức.**

---

# 3. Quy trình ra quyết định kỹ thuật

Toàn bộ hệ thống được thiết kế theo flow:

```text
EDA
  ↓
Observation
  ↓
Decision về dữ liệu / preprocessing / max_len
  ↓
Model Selection
  ↓
Training
  ↓
Validation Monitoring
  ↓
Interpretation
  ↓
Decision về checkpoint / training configuration
  ↓
Test Evaluation
  ↓
Interpretation về performance + computational cost
  ↓
Error Analysis
  ↓
Decision về pattern lỗi / limitation
  ↓
Model Comparison
  ↓
Final Model Decision
```

## 3.1. Nguyên tắc ra quyết định

### EDA → Decision

Mọi lựa chọn về:

- dữ liệu
- preprocessing
- sequence length
- input format

phải có căn cứ từ EDA.

### Training Monitoring → Decision

Checkpoint được lựa chọn dựa trên **validation performance**, không dựa trên test set.

### Evaluation → Interpretation

Không chỉ liệt kê Accuracy, Precision, Recall và F1 mà phải giải thích:

- mô hình mạnh ở lớp nào;
- mô hình yếu ở lớp nào;
- confusion matrix thể hiện pattern gì;
- chi phí tính toán có đáng kể hay không.

### Error Analysis → Decision

Các lỗi được phân tích theo:

- class pair;
- nội dung mẫu;
- confidence;
- topic overlap;
- short/low-context;
- possible label ambiguity/noise.

### Comparison → Final Decision

Mô hình cuối cùng được lựa chọn dựa trên:

```text
Classification Performance
+
Computational Cost
+
Error Behavior
+
Experimental Evidence
```

---

# 4. Các module

## 4.1. Module 1 – Phân loại chủ đề tin tức

Phân loại bài báo thuộc một trong bốn nhóm:

- World
- Sports
- Business
- Sci/Tech

---

## 4.2. Module 2 – Baseline

Sử dụng:

```text
TF-IDF + Logistic Regression
```

Module này cung cấp mốc tham chiếu để đánh giá hiệu quả của các phương pháp Transformer.

---

## 4.3. Module 3 – Transformer Classification

Sử dụng:

```text
distilbert-base-uncased
```

để tạo biểu diễn văn bản có ngữ cảnh.

---

## 4.4. Module 4 – Frozen Encoder

Đóng băng toàn bộ tham số của Transformer Encoder.

Chỉ Classification Head được cập nhật.

---

## 4.5. Module 5 – Fine-Tuning

Cho phép cập nhật:

```text
Transformer Encoder
+
Classification Head
```

trong quá trình huấn luyện.

---

## 4.6. Module 6 – Evaluation

Tính:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

Đồng thời ghi nhận:

- Training Time
- Inference Time
- Trainable Parameters
- Model Size

---

## 4.7. Module 7 – Error Analysis

Phân tích:

- các mẫu bị phân loại sai;
- các class pair thường bị nhầm;
- confidence của các dự đoán;
- nguyên nhân có thể gây lỗi.

---

## 4.8. Module 8 – Model Comparison

So sánh:

1. TF-IDF + Logistic Regression
2. Frozen Encoder
3. Fine-Tuning

Việc so sánh dựa trên:

- Classification Performance
- Computational Cost
- Error Behavior

---

# 5. Cấu trúc thư mục

```text
Course-Work-6/
│
├── README.md
├── .gitignore
├── requirements.txt
├── CHANGELOG.md
│
├── docs/
│   ├── Define.md
│   ├── Design.md
│   └── research/
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
│   ├── checkpoints/
│   └── error_analysis.md
│
└── report/
    └── report.md
```

| Thư mục / File | Vai trò |
|---|---|
| `README.md` | Mô tả project và hướng dẫn cài đặt/chạy |
| `Define.md` | Xác định vấn đề, yêu cầu và tiêu chí thành công |
| `Design.md` | Thiết kế kiến trúc, pipeline và thực nghiệm |
| `requirements.txt` | Danh sách thư viện |
| `data/` | Dữ liệu |
| `src/` | Mã nguồn |
| `notebooks/` | EDA, training, evaluation và comparison |
| `results/metrics/` | Kết quả định lượng |
| `results/figures/` | Biểu đồ và confusion matrix |
| `results/checkpoints/` | Model checkpoints |
| `report/` | Báo cáo |

---

# 6. Dữ liệu và chia tập

## 6.1. Nguồn dữ liệu

Dự án sử dụng **AG News** thông qua Hugging Face Datasets:

```text
fancyzhx/ag_news
```

Dataset gồm:

- Training: 120,000 mẫu
- Test: 7,600 mẫu

Training set được chia tiếp thành:

- Training: 108,000 mẫu
- Validation: 12,000 mẫu

Test set được giữ riêng cho đánh giá cuối cùng.

---

## 6.2. Các lớp

| Label | Chủ đề |
|---:|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

---

## 6.3. Trường dữ liệu

Pipeline hiện tại sử dụng:

- `text`
- `label`

`text` chứa nội dung tin tức và `label` chứa nhãn tương ứng.

---

## 6.4. Kiểm tra chất lượng dữ liệu

EDA kiểm tra:

- Missing values
- Empty text
- Duplicate
- Class distribution
- Text length
- Token length

Kết quả EDA hiện tại:

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

Các duplicate được ghi nhận trong EDA và được xem xét trong quá trình chuẩn bị dữ liệu.

---

# 7. Tokenization và lựa chọn độ dài chuỗi

Pipeline Transformer sử dụng tokenizer của:

```text
distilbert-base-uncased
```

Tokenizer chuyển văn bản thành:

```text
input_ids
attention_mask
```

Trong đó `attention_mask` giúp mô hình phân biệt token thật và token padding.

## 7.1. Kết quả khảo sát token coverage

EDA cho thấy:

| Max Length | Token Coverage |
|---:|---:|
| 64 | 87.41% |
| 128 | 99.42% |
| 256 | 99.97% |

## 7.2. Quyết định

Nhóm sử dụng:

```text
max_length = 64
```

trong thực nghiệm chính.

Đây là một **trade-off giữa chi phí tính toán và mức độ giữ lại thông tin của văn bản**:

- `64` giúp giảm chi phí tính toán;
- nhưng một phần văn bản dài có thể bị truncation;
- `128` và `256` giữ được nhiều token hơn nhưng làm tăng chi phí xử lý.

Do đó, `64` được xem là **cấu hình thực nghiệm được lựa chọn**, không phải kết luận rằng đây là giá trị tối ưu tuyệt đối.

Nếu thay đổi `max_length`, các thực nghiệm Transformer cần được chạy lại để đảm bảo so sánh công bằng.

---

# 8. Luồng dữ liệu

## 8.1. Inference

```text
Văn bản tin tức
      |
      v
Preprocessing
      |
      v
Tokenizer
      |
      v
input_ids + attention_mask
      |
      v
DistilBERT Encoder
      |
      v
last_hidden_state
      |
      v
Mean Pooling
      |
      v
Classification Head
      |
      v
4 logits
      |
      v
Softmax
      |
      v
Predicted Class + Probabilities
```

---

## 8.2. Training

```text
Văn bản + Label
      |
      v
Preprocessing
      |
      v
Tokenization
      |
      v
input_ids + attention_mask
      |
      v
DistilBERT Encoder
      |
      v
Mean Pooling
      |
      v
Classification Head
      |
      v
Logits
      |
      v
Cross-Entropy Loss
      |
      v
Backpropagation
      |
      v
Parameter Update
```

### Frozen Encoder

```text
Encoder → Frozen
Classification Head → Trainable
```

### Fine-Tuning

```text
Encoder → Trainable
Classification Head → Trainable
```

---

# 9. Cấu hình thí nghiệm

## 9.1. Baseline

Baseline sử dụng:

```text
TF-IDF + Logistic Regression
```

Hyperparameter được lựa chọn dựa trên validation performance.

Cấu hình tốt nhất hiện tại sử dụng:

```text
C = 2.0
```

---

## 9.2. Frozen Encoder

| Tham số | Giá trị |
|---|---|
| Model | `distilbert-base-uncased` |
| Encoder | Frozen |
| Trainable part | Classification Head |
| Learning Rate | `1e-3` |
| Optimizer | AdamW |
| Feature Extraction Batch Size | 64 |
| Training Batch Size | 256 |
| Epochs | 20 |
| Max Length | 64 |
| Random Seed | 42 |
| Loss | Cross-Entropy |

Best checkpoint được lựa chọn dựa trên validation performance.

Kết quả chính thức đạt:

```text
Accuracy = 90.29%
Macro F1 = 90.28%
```

---

## 9.3. Fine-Tuning

| Tham số | Giá trị |
|---|---|
| Model | `distilbert-base-uncased` |
| Encoder | Trainable |
| Classification Head | Trainable |
| Learning Rate | `2e-5` |
| Optimizer | AdamW |
| Batch Size | 32 |
| Epochs | 3 |
| Max Length | 64 |
| Random Seed | 42 |
| Loss | Cross-Entropy |
| Device | CUDA |
| Best Epoch | 2 |

Checkpoint tốt nhất được xác định dựa trên **validation performance**.

Kết quả chính thức trên test:

```text
Accuracy = 94.22%
Macro F1 = 94.23%
Macro Precision = 94.24%
Macro Recall = 94.22%
```

Thời gian huấn luyện khoảng:

```text
2353 seconds ≈ 39.2 minutes
```

---

## 9.4. Quick Test

Quick Test chỉ được sử dụng để kiểm tra:

- pipeline;
- code;
- model loading;
- training loop;
- evaluation flow.

Quick Test không được sử dụng làm kết quả nghiên cứu chính thức.

Cấu hình Quick Test sử dụng subset nhỏ hơn và một epoch nên kết quả không được dùng để so sánh cuối cùng với các mô hình chính thức.

---

# 10. Training Monitoring

Training được theo dõi trên validation set theo từng epoch.

Các chỉ số cần quan sát:

- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy
- Macro F1 trên validation

Mục tiêu của monitoring:

1. Phát hiện overfitting.
2. Xác định epoch tốt nhất.
3. Chọn checkpoint.
4. Phát hiện training không ổn định.
5. Quyết định có cần thay đổi hyperparameter hay không.

### Quy tắc

```text
Training Set
      ↓
Training Metrics
      ↓
Validation Metrics
      ↓
Best Validation Checkpoint
      ↓
Final Test Evaluation
```

**Test set không được sử dụng để lựa chọn epoch hoặc hyperparameter.**

Đối với Fine-Tuning, validation performance tốt nhất xuất hiện tại **epoch 2**, do đó checkpoint tương ứng được sử dụng cho đánh giá test cuối cùng.

---

# 11. Điều kiện so sánh

Để đảm bảo so sánh công bằng, các mô hình sử dụng:

- cùng dataset;
- cùng label mapping;
- cùng test set;
- cùng nguyên tắc preprocessing;
- cùng cách đánh giá;
- cùng metric chính.

Đối với Transformer:

- cùng backbone `distilbert-base-uncased`;
- cùng tokenizer;
- cùng `max_length=64`.

Các hyperparameter riêng của từng phương pháp được ghi nhận để phản ánh đúng trade-off giữa performance và computational cost.

---

# 12. Giao diện các module

## 12.1. Dataset

```python
load_dataset()
```

Nạp và chuẩn bị dữ liệu AG News.

---

## 12.2. Tokenizer

```python
tokenize(text)
```

Chuyển văn bản thành:

```text
input_ids
attention_mask
```

---

## 12.3. Model

```python
build_model(model_name, freeze_encoder)
```

Xây dựng Transformer classification model.

`freeze_encoder` xác định Encoder được đóng băng hay trainable.

---

## 12.4. Training

```python
train(model, train_data, validation_data)
```

Huấn luyện model và theo dõi validation performance.

---

## 12.5. Evaluation

```python
evaluate(model, test_data)
```

Tính:

- Accuracy
- Precision
- Recall
- Macro F1
- Confusion Matrix

---

## 12.6. Prediction

```python
predict(text)
```

Nhận một văn bản và trả về:

```text
Predicted Label
Probabilities
```

---

## 12.7. Comparison

```python
compare(results_baseline, results_frozen, results_finetuned)
```

Tổng hợp kết quả của ba phương pháp.

---

# 13. Phương pháp đánh giá

## 13.1. Classification Performance

Các mô hình được đánh giá trên cùng test set:

```text
Accuracy
Precision
Recall
Macro F1-score
Confusion Matrix
```

Macro averaging được sử dụng để đảm bảo mỗi lớp có trọng số như nhau trong đánh giá tổng thể.

---

## 13.2. Computational Performance

| Chỉ số | Ý nghĩa |
|---|---|
| Training Time | Thời gian cần để huấn luyện |
| Inference Time | Thời gian xử lý dữ liệu khi dự đoán |
| Trainable Parameters | Số tham số được cập nhật |
| Model Size | Dung lượng model/checkpoint |

---

# 14. Error Analysis

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

### 14.1. Model-driven error

Nội dung có tín hiệu rõ cho nhãn thật nhưng model chọn nhãn khác.

### 14.2. Topic overlap

Một bài báo có đặc điểm của nhiều chủ đề.

Ví dụ:

```text
Business ↔ Sci/Tech
```

### 14.3. Short / Low-context

Văn bản quá ngắn hoặc thiếu ngữ cảnh cần thiết.

### 14.4. Possible label ambiguity / noise

Nhãn gốc có thể không hoàn toàn phản ánh nội dung bài báo.

Error Analysis không chỉ dùng để mô tả lỗi mà còn dùng để đưa ra quyết định về:

- hạn chế của mô hình;
- hạn chế của dữ liệu;
- hướng cải thiện;
- lựa chọn mô hình cuối cùng.

---

# 15. Model Comparison

Ba phương pháp được so sánh:

```text
TF-IDF + Logistic Regression
          vs
Frozen Encoder
          vs
Fine-Tuning
```

Việc so sánh gồm ba khía cạnh:

### 15.1. Performance

- Accuracy
- Precision
- Recall
- Macro F1

### 15.2. Computational Cost

- Training Time
- Inference Time
- Trainable Parameters
- Model Size

### 15.3. Error Behavior

- Confusion Matrix
- Class pair
- Representative wrong samples
- Error pattern

Quyết định cuối cùng không dựa trên Accuracy đơn lẻ.

---

# 16. Kết quả thực nghiệm và quyết định kỹ thuật

Kết quả chính thức hiện tại:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 91.96% | 91.94% |
| Frozen Encoder | 90.29% | 90.28% |
| Fine-Tuning | **94.22%** | **94.23%** |

Kết quả cho thấy:

- Frozen Encoder thấp hơn Baseline.
- Baseline đạt hiệu năng cao hơn Frozen Encoder trong cấu hình thực nghiệm hiện tại.
- Fine-Tuning đạt hiệu năng cao nhất.

### Decision

Trong phạm vi thực nghiệm của project:

> **Fine-Tuning được lựa chọn là phương pháp có hiệu năng phân loại tốt nhất cho AG News.**

Lý do:

1. Accuracy cao nhất: **94.22%**.
2. Macro F1 cao nhất: **94.23%**.
3. Cao hơn Baseline và Frozen Encoder.
4. Việc cập nhật Transformer giúp mô hình thích nghi tốt hơn với nhiệm vụ phân loại chủ đề.

Tuy nhiên, Fine-Tuning có chi phí tính toán cao hơn Frozen Encoder. Vì vậy quyết định này là:

```text
Best Classification Performance
        ↓
Fine-Tuning
        ↓
Higher Computational Cost
        ↓
Acceptable Trade-off
```

Trong trường hợp ưu tiên giảm chi phí huấn luyện hoặc triển khai nhẹ hơn, Baseline hoặc Frozen Encoder có thể được cân nhắc tùy yêu cầu hệ thống.

---

# 17. Kết quả đầu ra

| Đầu ra | Vị trí |
|---|---|
| Metrics | `results/metrics/` |
| Predictions | `results/metrics/` |
| Confusion Matrix | `results/figures/` |
| Training curves | `results/figures/` |
| Checkpoints | `results/checkpoints/` |
| Error Analysis | `docs/results/error_analysis.md` |
| Model Comparison | `docs/results/model_comparison.md` |
| Notebooks | `notebooks/` |
| Report | `report/report.md` |

Các file dữ liệu lớn và checkpoint không cần thiết phải đưa trực tiếp lên GitHub nếu vượt quá giới hạn repository.

---

# 18. Các mốc thực hiện

## Mốc 1 – Data Preparation

- Nạp AG News.
- Kiểm tra dữ liệu.
- Kiểm tra missing/empty/duplicate.
- Kiểm tra class distribution.
- Phân tích text/token length.
- Chuẩn bị train/validation/test.
- Chuẩn bị tokenizer.

## Mốc 2 – EDA Decision

- Phân tích dữ liệu.
- Phân tích token coverage.
- Đánh giá trade-off của `max_length`.
- Quyết định preprocessing và cấu hình đầu vào.

## Mốc 3 – Baseline

- TF-IDF.
- Logistic Regression.
- Validation.
- Test evaluation.
- Lưu metrics.

## Mốc 4 – Frozen Encoder

- Load DistilBERT.
- Freeze Encoder.
- Mean Pooling.
- Train Classification Head.
- Validation monitoring.
- Test evaluation.
- Lưu checkpoint và metrics.

## Mốc 5 – Fine-Tuning

- Load DistilBERT.
- Cho phép Encoder trainable.
- Train Encoder + Classification Head.
- Theo dõi validation.
- Chọn best checkpoint.
- Test evaluation.
- Lưu checkpoint và metrics.

## Mốc 6 – Evaluation

- Accuracy.
- Precision.
- Recall.
- Macro F1.
- Confusion Matrix.
- Training Time.
- Inference Time.
- Trainable Parameters.
- Model Size.

## Mốc 7 – Error Analysis

- Xác định wrong samples.
- Phân tích class pairs.
- Kiểm tra confidence.
- Phân loại nguyên nhân lỗi.
- Đưa ra decision.

## Mốc 8 – Model Comparison

- So sánh Baseline.
- So sánh Frozen Encoder.
- So sánh Fine-Tuning.
- Phân tích performance.
- Phân tích computational cost.
- Phân tích error behavior.
- Đưa ra final model decision.

## Mốc 9 – Documentation

- Cập nhật README.
- Kiểm tra Define.md.
- Kiểm tra Design.md.
- Hoàn thiện Error Analysis.
- Hoàn thiện Model Comparison.
- Hoàn thiện Report.
- Cập nhật CHANGELOG.
- Kiểm tra khả năng tái lập.
- Chuẩn bị demo và presentation.

---

# 19. Rủi ro kỹ thuật

## 19.1. Chi phí tính toán cao

Fine-Tuning Transformer yêu cầu nhiều tài nguyên hơn Frozen Encoder và Baseline.

Giải pháp:

- sử dụng GPU/CUDA khi có;
- kiểm soát batch size;
- giới hạn sequence length;
- theo dõi training time.

---

## 19.2. Overfitting

Fine-Tuning có thể học quá sát training data.

Theo dõi:

```text
Train Performance
vs
Validation Performance
```

Nếu validation performance giảm trong khi training performance tiếp tục tăng, cần xem xét overfitting.

---

## 19.3. Văn bản dài

Văn bản dài hơn `max_length` sẽ bị truncation.

Kết quả EDA cho thấy `max_length=64` có coverage thấp hơn 128 và 256, do đó đây là một trade-off cần được ghi nhận.

---

## 19.4. So sánh không công bằng

Cần đảm bảo:

- cùng dataset;
- cùng test set;
- cùng label mapping;
- cùng nguyên tắc preprocessing;
- cùng evaluation metrics.

---

## 19.5. Nhầm lẫn giữa các lớp

Một số bài báo có thể chứa nội dung giao nhau.

Đặc biệt cần theo dõi:

```text
Business ↔ Sci/Tech
```

Error Analysis được sử dụng để kiểm tra pattern này.

---

## 19.6. Khả năng tái lập

Cần ghi nhận:

- Random seed;
- Model name;
- Hyperparameters;
- Dataset split;
- Max length;
- Device;
- Metrics;
- Checkpoint.

---

## 19.7. Dung lượng file lớn

Dataset lớn và checkpoint không nên được đưa trực tiếp vào GitHub nếu không cần thiết.

`.gitignore` được sử dụng để quản lý các file phù hợp.

---

# 20. Decision Evidence Checklist

Trước khi đưa ra kết luận cuối cùng, nhóm phải kiểm tra đủ bốn chuỗi bằng chứng:

| Chuỗi | Bằng chứng tối thiểu | Kết luận cần ghi |
|---|---|---|
| EDA → Decision | Missing, duplicate, class distribution, token coverage | Vì sao xử lý dữ liệu và chọn cấu hình input |
| Training Monitoring → Decision | Train/validation loss + accuracy theo epoch | Có overfitting không, best epoch nào và vì sao |
| Evaluation → Interpretation | Metrics + confusion matrix + computational cost | Mô hình mạnh/yếu ở đâu và trade-off |
| Error Analysis → Decision | Class pair + wrong samples + nguyên nhân | Pattern lỗi nào quan trọng và cần cải thiện gì |

### Nguyên tắc cuối cùng

> **Nếu một quyết định không có bằng chứng từ kết quả thực nghiệm tương ứng, không trình bày quyết định đó như một kết luận thực nghiệm.**

Toàn bộ thiết kế hướng tới việc chứng minh rằng nhóm không chỉ **chạy code và lấy kết quả**, mà thực hiện chuỗi:

```text
DATA
  ↓
OBSERVATION
  ↓
ANALYSIS
  ↓
DECISION
  ↓
EXPERIMENT
  ↓
EVALUATION
  ↓
ERROR ANALYSIS
  ↓
FINAL TECHNICAL DECISION
```
