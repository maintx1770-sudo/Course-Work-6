# Coursework_Topic 6
## News Topic Classification Using Transformer-Based Models

## 1. Giới thiệu

Course Work 6 xây dựng hệ thống **phân loại chủ đề tin tức (News Topic Classification)** sử dụng các mô hình học máy và học sâu dựa trên **Transformer**.

Hệ thống sử dụng tập dữ liệu **AG News** để phân loại các bài báo thành 4 nhóm chủ đề:

- **World**
- **Sports**
- **Business**
- **Sci/Tech**

Project tập trung so sánh ba phương pháp:

1. **TF-IDF + Logistic Regression** – Baseline truyền thống.
2. **Frozen Encoder** – đóng băng các tham số của Transformer và chỉ huấn luyện Classification Head.
3. **Fine-Tuning** – huấn luyện đồng thời Transformer và Classification Head trên tập dữ liệu AG News.

Mô hình Transformer được sử dụng trong project là **DistilBERT (`distilbert-base-uncased`)**.

Mục tiêu của project là đánh giá sự khác biệt giữa các phương pháp về **hiệu năng phân loại** và **chi phí tính toán**.

---

## 2. Mục tiêu

Project hướng đến các mục tiêu:

- Xây dựng hệ thống tự động phân loại tin tức.
- Áp dụng Transformer cho bài toán Text Classification.
- Xây dựng Baseline bằng TF-IDF + Logistic Regression.
- Xây dựng và thực nghiệm Frozen Encoder.
- Xây dựng và thực nghiệm Fine-Tuning.
- So sánh hiệu quả của ba phương pháp.
- Đánh giá mô hình bằng các chỉ số định lượng.
- Phân tích sự cân bằng giữa hiệu năng phân loại và chi phí tính toán.
- Thực hiện Prediction và Error Analysis.

---

## 3. Dataset

### 3.1 AG News

Project sử dụng dataset **AG News** cho bài toán phân loại chủ đề tin tức.

Dataset gồm 4 lớp:

| Label | Category |
|---:|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

### 3.2 Input

Input của hệ thống là văn bản tin tức.

Dữ liệu được sử dụng trong project gồm trường:

```text
text
label
```

Trong đó:

- `text`: nội dung bài báo.
- `label`: nhãn chủ đề của bài báo.

### 3.3 Output

Hệ thống trả về:

- Chủ đề dự đoán.
- Nhãn dự đoán.
- Xác suất dự đoán của các lớp khi mô hình hỗ trợ probability output.

Ví dụ:

```text
Input:
Manchester United won the match 3-1.

Output:
Predicted Category: Sports
```

---

## 4. Phương pháp

Project sử dụng ba phương pháp để thực nghiệm và so sánh.

### 4.1 Baseline

Baseline sử dụng:

```text
Text
 ↓
TF-IDF
 ↓
Logistic Regression
 ↓
Prediction
```

TF-IDF được sử dụng để biểu diễn văn bản dưới dạng vector đặc trưng.

Logistic Regression được sử dụng làm mô hình phân loại.

Baseline đóng vai trò là mô hình tham chiếu để đánh giá mức cải thiện của các phương pháp Transformer.

---

### 4.2 Frozen Encoder

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

- DistilBERT được sử dụng làm pretrained encoder.
- Các tham số của Transformer được **freeze**.
- Chỉ Classification Head được huấn luyện.
- Encoder tạo representation cho văn bản.
- Mean Pooling được sử dụng để tổng hợp representation từ `last_hidden_state`.
- Classification Head thực hiện phân loại thành 4 lớp.

Mục tiêu của Frozen Encoder là đánh giá khả năng sử dụng representation có sẵn từ pretrained Transformer với chi phí huấn luyện thấp hơn so với Fine-Tuning.

---

### 4.3 Fine-Tuning

Fine-Tuning sử dụng cùng pretrained model **DistilBERT**.

Pipeline:

```text
Input Text
    ↓
Tokenizer
    ↓
DistilBERT
    ↓
Classification Head
    ↓
Prediction
```

Trong phương pháp này:

- Các tham số của DistilBERT được phép cập nhật.
- Classification Head được huấn luyện.
- Toàn bộ mô hình được tối ưu trên dataset AG News.

Fine-Tuning nhằm đánh giá khả năng thích nghi của pretrained Transformer với bài toán phân loại chủ đề tin tức.

---

## 5. Evaluation

Các phương pháp được đánh giá trên cùng quy trình thực nghiệm và tập dữ liệu kiểm tra.

### 5.1 Classification Metrics

Các chỉ số chính:

- **Accuracy**
- **Precision**
- **Recall**
- **Macro F1-score**
- **Confusion Matrix**

### 5.2 Computational Metrics

Ngoài hiệu năng phân loại, project đánh giá:

- **Training Time**
- **Inference Time**
- **Trainable Parameters**
- **Model Size**

Các chỉ số này được sử dụng để phân tích trade-off giữa hiệu năng và chi phí tính toán.

---

## 6. Cấu trúc thư mục

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
│   │   ├── baseline_metrics.csv
│   │   ├── frozen_encoder_metrics.csv
│   │   ├── fine_tuning_metrics.csv
│   │   ├── evaluation_results.csv
│   │   └── test_predictions.csv
│   │
│   ├── figures/
│   │   ├── confusion_matrix_baseline.png
│   │   ├── confusion_matrix_frozen_encoder.png
│   │   ├── confusion_matrix_fine_tuning.png
│   │   └── model_comparison.png
│   │
│   └── checkpoints/
│       ├── frozen_encoder/
│       └── fine_tuning/
│
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── test_pipeline.py
│
└── report/
    ├── report.md
    ├── report.pdf
    └── presentation.pdf
```

### Mô tả các thư mục

#### `docs/`

Chứa tài liệu phân tích, nghiên cứu, thiết kế và tổng hợp kết quả.

```text
docs/
├── Define.md
├── Design.md
├── research/
│   ├── ag_news.md
│   ├── tfidf_logistic_regression.md
│   ├── frozen_encoder.md
│   ├── fine_tuning.md
│   └── evaluation_metrics.md
└── results/
    ├── error_analysis.md
    └── model_comparison.md
```

- `Define.md`: Xác định bài toán, đối tượng sử dụng, input/output, phạm vi, constraints và evaluation criteria.
- `Design.md`: Mô tả kiến trúc hệ thống, modules, data flow, interfaces, milestones và technical risks.
- `research/`: Chứa các tài liệu nghiên cứu liên quan đến AG News, TF-IDF + Logistic Regression, Frozen Encoder, Fine-Tuning và evaluation metrics.
- `results/`: Chứa các tài liệu phân tích kết quả.
  - `error_analysis.md`: Phân tích các mẫu dự đoán sai và các trường hợp dễ nhầm.
  - `model_comparison.md`: Tổng hợp và phân tích kết quả so sánh giữa các phương pháp.

---

#### `data/`

Chứa dữ liệu được sử dụng trong project.

```text
data/
├── raw/
└── processed/
```

- `raw/`: Dữ liệu gốc hoặc dữ liệu được tải về.
- `processed/`: Dữ liệu sau preprocessing nếu project cần lưu dưới dạng file.

> Dataset AG News được sử dụng thông qua pipeline dữ liệu của project. Không bắt buộc phải commit toàn bộ dataset vào repository.

---

#### `src/`

Chứa source code chính của project.

```text
src/
├── data.py
├── baseline.py
├── models.py
├── train.py
├── evaluate.py
├── compare.py
└── predict.py
```

- `data.py`: Load và chuẩn bị dữ liệu AG News.
- `baseline.py`: Xây dựng Baseline sử dụng TF-IDF + Logistic Regression.
- `models.py`: Định nghĩa các thành phần model Transformer và Classification Head.
- `train.py`: Xây dựng training pipeline cho Frozen Encoder và Fine-Tuning.
- `evaluate.py`: Tính toán metrics và đánh giá mô hình.
- `compare.py`: Tổng hợp và so sánh kết quả giữa Baseline, Frozen Encoder và Fine-Tuning.
- `predict.py`: Thực hiện prediction trên dữ liệu mới hoặc dữ liệu kiểm tra.

---

#### `notebooks/`

Chứa các notebook phục vụ quá trình khám phá dữ liệu và thực nghiệm.

```text
notebooks/
├── 01_data_exploration.ipynb
├── 02_baseline.ipynb
├── 03_frozen_encoder.ipynb
├── 04_fine_tuning.ipynb
└── 05_comparison.ipynb
```

- `01_data_exploration.ipynb`: Load, khám phá và kiểm tra dữ liệu AG News.
- `02_baseline.ipynb`: Xây dựng và thực nghiệm TF-IDF + Logistic Regression.
- `03_frozen_encoder.ipynb`: Thực nghiệm Frozen Encoder.
- `04_fine_tuning.ipynb`: Thực nghiệm Fine-Tuning.
- `05_comparison.ipynb`: Tổng hợp và so sánh kết quả của ba phương pháp.

---

#### `results/`

Lưu các kết quả thực nghiệm được tạo trong quá trình chạy project.

```text
results/
├── metrics/
│   ├── baseline_metrics.csv
│   ├── frozen_encoder_metrics.csv
│   ├── fine_tuning_metrics.csv
│   ├── evaluation_results.csv
│   └── test_predictions.csv
│
├── figures/
│   ├── confusion_matrix_baseline.png
│   ├── confusion_matrix_frozen_encoder.png
│   ├── confusion_matrix_fine_tuning.png
│   └── model_comparison.png
│
└── checkpoints/
    ├── frozen_encoder/
    └── fine_tuning/
```

- `metrics/`: Lưu Accuracy, Precision, Recall, Macro F1-score, Training Time, Inference Time, Trainable Parameters và Model Size.
- `figures/`: Lưu confusion matrix, biểu đồ so sánh và các hình ảnh trực quan hóa kết quả.
- `checkpoints/`: Lưu các checkpoint của mô hình Transformer trong quá trình thực nghiệm.

Các file `.csv`, `.png` và checkpoint trong `results/` được tạo trong quá trình chạy thực nghiệm. Khi mới tạo repository, có thể sử dụng `.gitkeep` để giữ cấu trúc thư mục.

---

#### `tests/`

Chứa các test phục vụ kiểm tra từng module và toàn bộ pipeline.

```text
tests/
├── test_data.py
├── test_models.py
├── test_training.py
├── test_evaluation.py
└── test_pipeline.py
```

Các test giúp kiểm tra:

- Data loading.
- Model loading.
- Training pipeline.
- Evaluation.
- Prediction.
- Integration giữa các module.

---

#### `report/`

Chứa các phiên bản cuối của báo cáo và presentation.

```text
report/
├── report.md
├── report.pdf
└── presentation.pdf
```

- `report.md`: Source của báo cáo.
- `report.pdf`: Báo cáo Coursework bản cuối.
- `presentation.pdf`: Slide presentation của project.

Các file báo cáo cuối sẽ được tạo sau khi nội dung và kết quả thực nghiệm hoàn tất.

---

## 7. Cài đặt

### Clone repository

```bash
git clone https://github.com/maintx1770-sudo/Course-Work-6.git
cd Course-Work-6
```

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## 8. Quy trình thực hiện

```text
AG News Dataset
      ↓
Data Exploration
      ↓
Preprocessing
      ↓
      ├───────────────┐
      ↓               ↓
   Baseline      Transformer
      │               │
      │       ┌───────┴───────┐
      │       ↓               ↓
      │ Frozen Encoder    Fine-Tuning
      │       │               │
      └───────┴───────┬───────┘
                      ↓
                  Evaluation
                      ↓
                  Comparison
                      ↓
                Prediction
                      ↓
                Error Analysis
                      ↓
                 Final Report
```

---

## 9. Tài liệu

Chi tiết về phân tích và thiết kế hệ thống:

- [Define](docs/Define.md)
- [Design](docs/Design.md)

Các tài liệu nghiên cứu:

- [AG News](docs/research/ag_news.md)
- [TF-IDF + Logistic Regression](docs/research/tfidf_logistic_regression.md)
- [Frozen Encoder](docs/research/frozen_encoder.md)
- [Fine-Tuning](docs/research/fine_tuning.md)
- [Evaluation Metrics](docs/research/evaluation_metrics.md)

---

## 10. Kết quả

Kết quả thực nghiệm được lưu trong:

```text
results/
```

Bao gồm:

- Evaluation metrics
- Confusion Matrix
- Model comparison
- Test predictions
- Trained checkpoints

Kết quả cuối cùng sẽ được sử dụng để so sánh:

```text
TF-IDF + Logistic Regression
          vs
Frozen Encoder
          vs
Fine-Tuning
```

về:

- Classification performance.
- Training cost.
- Inference cost.
- Number of trainable parameters.
- Model size.

---

## 11. Thành viên

| Họ và tên | MSSV | Vai trò |
|---|---|---|
| Mai | ... | Dataset + EDA + Documentation |
| Huynh | ... | Baseline |
| My | ... | Transformer Models + Training |
| Đức | ... | Evaluation + Comparison |
| Trúc Chi | ... | Prediction + Error Analysis |
| Nguyễn Chi | ... | Testing + Integration + Finalization |

---

## 12. Kết luận

Project xây dựng hệ thống **News Topic Classification** sử dụng dataset **AG News** và mô hình **DistilBERT**.

Ba phương pháp được thực nghiệm và so sánh gồm:

1. **TF-IDF + Logistic Regression**
2. **Frozen Encoder**
3. **Fine-Tuning**

Thông qua thực nghiệm, project đánh giá không chỉ khả năng phân loại mà còn xem xét **chi phí huấn luyện, chi phí suy luận, số lượng tham số có thể huấn luyện và kích thước mô hình**.

Kết quả thực nghiệm, comparison và error analysis được sử dụng để đánh giá ưu điểm, hạn chế và trade-off giữa các phương pháp.
