# THIẾT KẾ (DESIGN)

**Đề tài:** Phân loại chủ đề tin tức AG News bằng Transformer (BERT/DistilBERT): so sánh Frozen Encoder và Fine-Tuning.

## 1. Kiến trúc hệ thống

```text
Bộ dữ liệu AG News
      |
      v
Tiền xử lý dữ liệu
      |
      v
Tokenization
      |
      v
Transformer Encoder
      |
      v
Classification Head
      |
      v
Dự đoán 4 lớp
      |
      v
Đánh giá và so sánh
```

### 1.1 Baseline: TF-IDF + Logistic Regression

Mô hình không dùng Transformer, làm mốc để so sánh với hai cấu hình Transformer.

```text
Văn bản đầu vào
    |
TF-IDF
    |
Logistic Regression
    |
Dự đoán
```

### 1.2 Frozen Encoder (đóng băng bộ mã hóa)

```text
Văn bản đầu vào
    |
Tokenizer
    |
Transformer đã huấn luyện trước
    |  Đóng băng (Frozen)
    v
Classification Head
    |
Dự đoán
```

### 1.3 Fine-Tuning (tinh chỉnh)

```text
Văn bản đầu vào
    |
Tokenizer
    |
Transformer đã huấn luyện trước
    |  Có thể huấn luyện (Trainable)
    v
Classification Head
    |
Dự đoán
```

### 1.4 Cấu trúc mô hình

- **Encoder:** `distilbert-base-uncased` (mặc định) hoặc `bert-base-uncased`.
- **Biểu diễn văn bản:** vector của token đầu tiên (`[CLS]`) ở lớp cuối của encoder.
- **Classification Head:** Dropout, sau đó một lớp Linear từ kích thước ẩn (768) xuống 4 logit.
- **Hàm mất mát:** Cross-Entropy Loss.

## 2. Các module

### Module 1 – Phân loại chủ đề tin tức
Phân loại bài báo vào World, Sports, Business hoặc Sci/Tech.

### Module 2 – Phân loại dựa trên Transformer
Dùng BERT hoặc DistilBERT để tạo biểu diễn văn bản có ngữ cảnh.

### Module 3 – Frozen Encoder
Đóng băng tham số Transformer và chỉ huấn luyện classification head.

### Module 4 – Fine-Tuning
Cập nhật cả tham số Transformer và classification head.

### Module 5 – Đánh giá mô hình
Tính Accuracy, Precision, Recall, Macro F1-score và Confusion Matrix.

### Module 6 – So sánh mô hình
So sánh Frozen Encoder và Fine-Tuning (kèm baseline) theo các chỉ số hiệu năng và chi phí tính toán.

### Module 7 – Phân tích lỗi
Xem xét các mẫu bị phân loại sai, tìm cặp lớp hay nhầm lẫn và nguyên nhân có thể.

## 3. Cấu trúc thư mục

```text
Course-Work-6/
│
├── README.md
├── .gitignore
├── requirements.txt
├── docs/
│   ├── Define.md
│   └── Design.md
├── data/
├── src/
├── notebooks/
├── results/
└── report/
```

| Thư mục / File | Vai trò |
|---|---|
| `README.md` | Mô tả dự án, cách cài đặt và chạy |
| `requirements.txt` | Danh sách thư viện cần cài |
| `docs/` | Tài liệu: `Define.md` (định nghĩa bài toán), `Design.md` (thiết kế) |
| `data/` | Dữ liệu AG News (không đẩy file lớn lên GitHub) |
| `src/` | Mã nguồn: tiền xử lý, mô hình, huấn luyện, đánh giá, so sánh |
| `notebooks/` | Notebook huấn luyện, đánh giá và demo |
| `results/` | Chỉ số, biểu đồ, confusion matrix, kết quả so sánh |
| `report/` | Báo cáo cuối kỳ và slide |

### 3.1 Các file dự kiến trong `src/`

| File | Nội dung |
|---|---|
| `data.py` | Nạp AG News, chia train/validation/test, tiền xử lý, tokenization |
| `models.py` | Xây dựng mô hình, bật/tắt đóng băng encoder |
| `train.py` | Vòng lặp huấn luyện, early stopping, lưu mô hình |
| `evaluate.py` | Tính chỉ số, confusion matrix, đo thời gian suy luận |
| `compare.py` | Tổng hợp kết quả Frozen Encoder và Fine-Tuning, vẽ bảng và biểu đồ |
| `predict.py` | Dự đoán cho một bài báo mới |

## 4. Dữ liệu và chia tập

- **Nguồn:** AG News (Kaggle: `amananandrai/ag-news-classification-dataset`), gồm `train.csv` và `test.csv`.
- **Trường dữ liệu:** `Class Index`, `Title`, `Description`.
- **Quy mô:** khoảng 120.000 mẫu train và 7.600 mẫu test, 4 lớp cân bằng.
- **Chia tập:** tách 10% tập train làm validation (chia theo lớp, `stratify`), giữ nguyên tập test gốc để đánh giá cuối cùng.
- **Đầu vào mô hình:** ghép `Title` và `Description` thành một chuỗi.
- **Nhãn:** chuyển `Class Index` (1 đến 4) về 0 đến 3 (0: World, 1: Sports, 2: Business, 3: Sci/Tech).
- **Tiền xử lý:** loại bỏ khoảng trắng thừa, giữ nguyên chữ hoa chữ thường theo tokenizer (với mô hình `uncased`, tokenizer tự hạ chữ thường).

## 5. Luồng dữ liệu

### 5.1 Suy luận (Inference)

```text
Văn bản tin tức thô
      |
Tiền xử lý
      |
Tokenizer
      |
input_ids + attention_mask
      |
Transformer Encoder
      |
Biểu diễn văn bản
      |
Classification Head
      |
4 giá trị logit
      |
Softmax
      |
Danh mục dự đoán + xác suất
```

### 5.2 Huấn luyện (Training)

```text
Văn bản tin tức + Nhãn
      |
Tiền xử lý
      |
Tokenization
      |
Transformer Encoder
      |
Classification Head
      |
Logit dự đoán
      |
Cross-Entropy Loss
      |
Lan truyền ngược (Backpropagation)
      |
Cập nhật tham số
```

Ở cấu hình Frozen Encoder, các tham số của encoder không nhận gradient, chỉ classification head được cập nhật.

## 6. Cấu hình thí nghiệm

Các giá trị dưới đây là cấu hình đề xuất ban đầu, có thể điều chỉnh theo phần cứng và được ghi lại trong `results/` sau khi chạy.

| Tham số | Frozen Encoder | Fine-Tuning |
|---|---|---|
| Mô hình | DistilBERT (hoặc BERT) | giống bên trái |
| Tham số huấn luyện | chỉ classification head | toàn bộ mô hình |
| Learning rate | 1e-3 | 2e-5 |
| Optimizer | AdamW | AdamW |
| Batch size | 32 | 32 (giảm nếu thiếu bộ nhớ) |
| Số epoch | 3 đến 5 | 3 |
| Độ dài chuỗi tối đa | 128 | 128 |
| Random seed | 42 | 42 |

Điều kiện chung để so sánh công bằng: cùng cách chia dữ liệu, cùng tiền xử lý, cùng tokenizer, cùng độ dài chuỗi tối đa, cùng random seed và cùng tập test.

## 7. Giao diện / API

### Dataset
```python
load_dataset()
```
Nạp dữ liệu AG News và trả về train, validation, test.

### Tokenizer
```python
tokenize(text)
```
Chuyển văn bản thành `input_ids` và `attention_mask`.

### Xây dựng mô hình
```python
build_model(model_name, freeze_encoder)
```
Tạo mô hình với encoder đóng băng (`True`) hoặc có thể huấn luyện (`False`).

### Dự đoán
```python
predict(text)
```
Trả về lớp dự đoán và xác suất.

### Huấn luyện
```python
train(model, train_data, validation_data)
```
Huấn luyện cấu hình mô hình được chọn.

### Đánh giá
```python
evaluate(model, test_data)
```
Trả về Accuracy, Precision, Recall, Macro F1-score và Confusion Matrix.

### So sánh
```python
compare(results_frozen, results_finetuned)
```
So sánh hai hướng tiếp cận.

## 8. Phương pháp đánh giá và đo lường

### 8.1 Hiệu năng phân loại
- Accuracy, Precision, Recall, Macro F1-score trên tập test.
- Confusion Matrix cho từng mô hình.
- Báo cáo kết quả của baseline TF-IDF + Logistic Regression cùng bảng so sánh.

### 8.2 Hiệu năng tính toán

| Chỉ số | Cách đo |
|---|---|
| Thời gian huấn luyện | Tổng thời gian chạy huấn luyện (`time.perf_counter`) |
| Thời gian suy luận | Thời gian trung bình trên mỗi mẫu hoặc mỗi batch ở tập test |
| Số tham số huấn luyện được | Đếm tham số có `requires_grad = True` |
| Kích thước mô hình | Dung lượng file mô hình đã lưu (MB) |

### 8.3 Phân tích lỗi
- Liệt kê các cặp lớp bị nhầm nhiều nhất từ confusion matrix (ví dụ Business và Sci/Tech).
- Xem xét một số mẫu bị phân loại sai và nhận xét nguyên nhân.
- Nêu giới hạn của mô hình (văn bản dài bị cắt, chủ đề nằm giữa hai lớp).

## 9. Kết quả đầu ra

| Đầu ra | Vị trí |
|---|---|
| Chỉ số, bảng so sánh | `results/` |
| Biểu đồ và confusion matrix | `results/` |
| Mô hình đã huấn luyện | `results/` (không đẩy lên GitHub nếu nặng) |
| Notebook huấn luyện, đánh giá, demo | `notebooks/` |
| Báo cáo cuối kỳ, slide | `report/` |
| Hướng dẫn cài đặt và chạy | `README.md` |

## 10. Các mốc thực hiện

### Mốc 1 – Chuẩn bị dữ liệu
- Lấy bộ AG News
- Khảo sát dữ liệu
- Chuẩn bị tập train/validation/test
- Cài đặt tiền xử lý

### Mốc 2 – Baseline
- Cài đặt TF-IDF + Logistic Regression
- Chọn BERT hoặc DistilBERT
- Nạp tokenizer và mô hình
- Cài đặt classification head và kiểm tra forward pass

### Mốc 3 – Frozen Encoder
- Đóng băng Transformer
- Huấn luyện classification head
- Lưu mô hình và chỉ số

### Mốc 4 – Fine-Tuning
- Bật cập nhật cho Transformer
- Huấn luyện toàn bộ mô hình
- Lưu mô hình và chỉ số

### Mốc 5 – Đánh giá
- Tính toàn bộ chỉ số đánh giá
- Tạo confusion matrix
- Đo thời gian huấn luyện và suy luận

### Mốc 6 – So sánh và phân tích lỗi
- So sánh hiệu năng phân loại và chi phí tính toán
- Phân tích các đánh đổi (trade-off)
- Phân tích mẫu bị phân loại sai

### Mốc 7 – Tài liệu và demo
- Cập nhật README
- Ghi lại thiết lập và kết quả
- Chuẩn bị demo, báo cáo cuối và bài thuyết trình

## 11. Rủi ro kỹ thuật

### Rủi ro 1 – Chi phí tính toán cao
Dùng DistilBERT, giảm batch size, dùng GPU (hoặc Google Colab) và giới hạn độ dài chuỗi khi cần.

### Rủi ro 2 – Overfitting
Theo dõi hiệu năng trên tập validation và dùng early stopping khi phù hợp.

### Rủi ro 3 – Văn bản đầu vào dài
Dùng độ dài chuỗi tối đa cố định và cách cắt (truncation) nhất quán.

### Rủi ro 4 – So sánh không công bằng
Dùng cùng cách chia dữ liệu, tiền xử lý, chỉ số và random seed.

### Rủi ro 5 – Nhầm lẫn giữa các lớp
Phân tích confusion matrix và xem xét các mẫu bị phân loại sai.

### Rủi ro 6 – Khả năng tái lập
Ghi lại tên mô hình, random seed, siêu tham số, cách chia dữ liệu và môi trường phần mềm.

### Rủi ro 7 – Lựa chọn mô hình
Chọn BERT hay DistilBERT dựa trên phần cứng, thời gian huấn luyện, kích thước mô hình và hiệu năng.

### Rủi ro 8 – Dung lượng file lớn trên GitHub
Không đẩy dữ liệu và checkpoint nặng lên repo, thêm vào `.gitignore` và ghi cách tải trong `README.md`.
