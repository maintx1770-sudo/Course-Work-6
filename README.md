# Coursework_Topic 6
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

## 6. Cấu trúc thư 

```text
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
│       ├── baseline/
│       ├── frozen_encoder/
│       └── fine_tuning/
│
└── report/
    ├── report.md
    ├── report.pdf
    └── presentation.pdf
```
Mô tả các thư mục

docs/

Chứa tài liệu phân tích, nghiên cứu, thiết kế và tổng hợp kết quả của 
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
* Define.md: Xác định bài toán, người dùng, input/output, features, constraints và evaluation criteria.
* Design.md: Mô tả kiến trúc hệ thống, modules, data flow, interfaces, milestones và technical risks.
* research/: Chứa các tài liệu nghiên cứu liên quan đến AG News, TF-IDF + Logistic Regression, Frozen Encoder, Fine-Tuning và các phương pháp đánh giá.
* results/: Chứa phần phân tích và tổng hợp kết quả thực nghiệm.
    * error_analysis.md: Phân tích các mẫu dự đoán sai.
    * model_comparison.md: So sánh Baseline, Frozen Encoder và Fine-Tuning.

data/

Chứa dữ liệu của project.
```text
data/
├── raw/
└── processed/
```
* raw/: Dữ liệu AG News gốc.
* processed/: Dữ liệu sau khi preprocessing, được chuẩn bị để sử dụng cho Baseline và Transformer.

src/

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
* data.py: Load, xử lý và chuẩn bị dữ liệu AG News.
* baseline.py: Xây dựng và huấn luyện Baseline sử dụng TF-IDF + Logistic Regression.
* models.py: Định nghĩa và chuẩn bị mô hình Transformer cho Frozen Encoder.
* train.py: Xây dựng training pipeline và thực hiện Fine-Tuning.
* evaluate.py: Tính toán các metrics và đánh giá mô hình.
* compare.py: So sánh Baseline, Frozen Encoder và Fine-Tuning.
* predict.py: Thực hiện dự đoán trên dữ liệu và tạo kết quả prediction.

notebooks/

Chứa các notebook phục vụ quá trình khám phá dữ liệu và thực nghiệm.
```text
notebooks/
├── 01_data_exploration.ipynb
├── 02_baseline.ipynb
├── 03_frozen_encoder.ipynb
├── 04_fine_tuning.ipynb
└── 05_comparison.ipynb
```
* 01_data_exploration.ipynb: Load, khám phá và kiểm tra dữ liệu AG News.
* 02_baseline.ipynb: Xây dựng và thực nghiệm Baseline TF-IDF + Logistic Regression.
* 03_frozen_encoder.ipynb: Thực nghiệm Frozen Encoder.
* 04_fine_tuning.ipynb: Thực nghiệm Fine-Tuning.
* 05_comparison.ipynb: Tổng hợp và so sánh kết quả của ba phương pháp.

results/

Lưu các kết quả thực nghiệm của project.
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
    ├── baseline/
    ├── frozen_encoder/
    └── fine_tuning/
```
* metrics/: Lưu các kết quả đánh giá như Accuracy, Precision, Recall, Macro F1-score, Training Time, Inference Time, Trainable Parameters và Model Size.
* figures/: Lưu các biểu đồ, confusion matrix và hình ảnh so sánh mô hình.
* checkpoints/: Lưu các model checkpoint trong quá trình thực nghiệm.

Các file .csv, .png và checkpoint trong results/ được tạo trong quá trình chạy thực nghiệm. Khi mới tạo repository, có thể sử dụng .gitkeep để giữ cấu trúc thư mục.

report/

Chứa source và các phiên bản cuối cùng của báo cáo và presentation.
```text
report/
├── report.md
├── report.pdf
└── presentation.pdf
```
* report.md: Source của báo cáo, dùng để chỉnh sửa và cập nhật nội dung.
* report.pdf: Báo cáo Coursework bản cuối.
* presentation.pdf: Presentation dùng để trình bày project.

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
