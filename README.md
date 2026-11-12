# COURSE WORK 6
## News Topic Classification Using Transformer-Based Models

## 1. Giới thiệu

Course Work 6 xây dựng hệ thống **phân loại chủ đề tin tức (News Topic Classification)** sử dụng các mô hình học sâu dựa trên **Transformer**.

Hệ thống sử dụng tập dữ liệu **AG News** để phân loại các bài báo thành 4 nhóm chủ đề:

- **World**
- **Sports**
- **Business**
- **Sci/Tech**

Project tập trung so sánh hai phương pháp sử dụng mô hình Transformer:

1. **Frozen Encoder** – đóng băng các tham số của Transformer và chỉ huấn luyện Classification Head.
2. **Fine-Tuning** – huấn luyện đồng thời Transformer và Classification Head trên tập dữ liệu AG News.

Ngoài ra, project có thể sử dụng **TF-IDF + Logistic Regression** làm baseline để so sánh với các phương pháp Transformer.

---

## 2. Mục tiêu

Project hướng đến các mục tiêu:

- Xây dựng hệ thống tự động phân loại tin tức.
- Áp dụng mô hình Transformer cho bài toán Text Classification.
- So sánh Frozen Encoder và Fine-Tuning.
- Đánh giá hiệu quả của các phương pháp bằng các chỉ số định lượng.
- Phân tích sự cân bằng giữa độ chính xác và chi phí tính toán.

---

## 3. Dataset

### AG News

AG News là tập dữ liệu phân loại tin tức gồm 4 nhóm:

| Label | Category |
|---|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

### Input

Văn bản tin tức, có thể bao gồm:

- Tiêu đề
- Mô tả
- Tiêu đề + mô tả

### Output

Hệ thống trả về:

- Chủ đề dự đoán
- Xác suất của lớp dự đoán

Ví dụ:

```text
Input:
Manchester United won the match 3-1.

Output:
Predicted Category: Sports
```

---

## 4. Phương pháp

### 4.1 Baseline

Sử dụng:

```text
TF-IDF → Logistic Regression
```

Baseline được sử dụng để có một mô hình tham chiếu trước khi áp dụng Transformer.

### 4.2 Frozen Encoder

```text
Input Text
     ↓
Tokenizer
     ↓
Pretrained Transformer
     ↓
Frozen Encoder
     ↓
Classification Head
     ↓
Prediction
```

Trong phương pháp này, các tham số của Transformer được giữ nguyên và chỉ Classification Head được huấn luyện.

### 4.3 Fine-Tuning

```text
Input Text
     ↓
Tokenizer
     ↓
Pretrained Transformer
     ↓
Classification Head
     ↓
Prediction
```

Trong phương pháp này, cả Transformer và Classification Head đều được cập nhật trong quá trình huấn luyện.

### Transformer Model

Project có thể sử dụng:

- BERT
- DistilBERT

---

## 5. Evaluation

Các mô hình được đánh giá bằng:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

Ngoài hiệu năng phân loại, project cũng so sánh:

- Training Time
- Inference Time
- Number of Trainable Parameters
- Model Size

---

## 6. Cấu trúc thư mục

```text
Course-Work-6/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── docs/
│   ├── Define.md
│   └── Design.md
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── data.py
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

### Mô tả các thư mục

#### `docs/`

Chứa tài liệu phân tích và thiết kế project.

- `Define.md`: Xác định bài toán, người dùng, input/output, features, constraints và evaluation criteria.
- `Design.md`: Mô tả kiến trúc hệ thống, modules, data flow, interfaces, milestones và technical risks.

#### `data/`

Chứa dữ liệu của project.

```text
data/
├── raw/
└── processed/
```

- `raw/`: dữ liệu gốc.
- `processed/`: dữ liệu sau khi preprocessing.

#### `src/`

Chứa source code chính của project.

- `data.py`: xử lý và chuẩn bị dữ liệu.
- `models.py`: định nghĩa các mô hình.
- `train.py`: huấn luyện mô hình.
- `evaluate.py`: đánh giá mô hình.
- `compare.py`: so sánh các phương pháp.
- `predict.py`: thực hiện dự đoán trên dữ liệu mới.

#### `notebooks/`

Chứa các notebook phục vụ quá trình thực nghiệm:

- `01_data_exploration.ipynb`: khám phá và kiểm tra dữ liệu.
- `02_baseline.ipynb`: xây dựng baseline.
- `03_frozen_encoder.ipynb`: thực nghiệm Frozen Encoder.
- `04_fine_tuning.ipynb`: thực nghiệm Fine-Tuning.
- `05_comparison.ipynb`: so sánh kết quả.

#### `results/`

Lưu kết quả thực nghiệm.

```text
results/
├── metrics/
├── figures/
└── checkpoints/
```

- `metrics/`: Accuracy, Precision, Recall, F1-score,...
- `figures/`: biểu đồ và confusion matrix.
- `checkpoints/`: model đã huấn luyện.

#### `report/`

Chứa báo cáo và presentation cuối cùng.

---

## 7. Cài đặt

Clone repository:

```bash
git clone https://github.com/maintx1770-sudo/Course-Work-6.git
cd Course-Work-6
```

Cài đặt các thư viện:

```bash
pip install -r requirements.txt
```

---

## 8. Quy trình thực hiện

```text
Dataset
   ↓
Data Exploration
   ↓
Preprocessing
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
Report
```

---

## 9. Tài liệu

Chi tiết về yêu cầu và thiết kế hệ thống:

- [Define](docs/Define.md)
- [Design](docs/Design.md)

---

## 10. Kết quả

Kết quả thực nghiệm sẽ được cập nhật trong thư mục:

```text
results/
```

Bao gồm:

- Evaluation metrics
- Confusion Matrix
- Training curves
- Model comparison
- Trained checkpoints

---

## 11. Thành viên

| Họ và tên | MSSV | Vai trò |
|---|---|---|
| ... | ... | ... |
| ... | ... | ... |
| ... | ... | ... |

---

## 12. Kết luận

Project xây dựng hệ thống phân loại tin tức sử dụng Transformer và tập dữ liệu AG News.

Hai phương pháp chính được nghiên cứu là **Frozen Encoder** và **Fine-Tuning**. Kết quả thực nghiệm được sử dụng để đánh giá sự khác biệt về hiệu năng phân loại và chi phí tính toán giữa hai phương pháp.
