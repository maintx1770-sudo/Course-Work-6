# THIẾT KẾ (DESIGN)

**Đề tài:** Phân loại chủ đề tin tức AG News bằng Transformer: so sánh Frozen Encoder và Fine-Tuning.

## 1. Kiến trúc hệ thống

Hệ thống sử dụng tập dữ liệu AG News để xây dựng mô hình phân loại tin tức thành 4 nhóm chủ đề: World, Sports, Business và Sci/Tech.

Luồng xử lý tổng quát:

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
Mean Pooling
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

### 1.1. Baseline: TF-IDF + Logistic Regression

Baseline không sử dụng Transformer và được xây dựng làm mô hình tham chiếu để so sánh với hai cấu hình Transformer.

```text
Văn bản đầu vào
    |
TF-IDF
    |
Logistic Regression
    |
Dự đoán
```

### 1.2. Frozen Encoder

Frozen Encoder sử dụng Transformer đã được huấn luyện trước nhưng đóng băng toàn bộ tham số của encoder. Chỉ Classification Head được cập nhật trong quá trình huấn luyện.

```text
Văn bản đầu vào
    |
Tokenizer
    |
DistilBERT Encoder
    |  Đóng băng (Frozen)
    v
Mean Pooling
    |
Classification Head
    |
Dự đoán
```

### 1.3. Fine-Tuning

Fine-Tuning sử dụng cùng backbone Transformer với Frozen Encoder nhưng cho phép cập nhật tham số của cả Transformer Encoder và Classification Head.

```text
Văn bản đầu vào
    |
Tokenizer
    |
DistilBERT Encoder
    |  Có thể huấn luyện (Trainable)
    v
Mean Pooling
    |
Classification Head
    |
Dự đoán
```

### 1.4. Cấu trúc mô hình

- **Encoder:** `distilbert-base-uncased`.
- **Tokenizer:** tokenizer tương ứng với `distilbert-base-uncased`.
- **Biểu diễn văn bản:** sử dụng Mean Pooling trên `last_hidden_state` của Transformer Encoder. `attention_mask` được sử dụng để bỏ qua các token padding khi tính trung bình.
- **Classification Head:** gồm Dropout và một lớp Linear, ánh xạ vector biểu diễn của encoder xuống 4 logit tương ứng với 4 lớp.
- **Hàm mất mát:** Cross-Entropy Loss.
- **Các lớp đầu ra:** World, Sports, Business và Sci/Tech.


## 2.0. Yêu cầu phân tích và ra quyết định

Thiết kế phải thể hiện được mối liên hệ giữa quan sát thực nghiệm và quyết định kỹ thuật:

| Giai đoạn | Bằng chứng cần thu thập | Quyết định cần trả lời |
|---|---|---|
| EDA | Missing/empty, duplicate, class distribution, token length | Preprocessing, split, input và `max_len` |
| Training Monitoring | Train loss/accuracy, validation loss/accuracy theo epoch | Có overfitting không? Chọn checkpoint nào? Có cần đổi cấu hình không? |
| Evaluation | Accuracy, Precision, Recall, Macro F1, confusion matrix, time, params, size | Mô hình mạnh ở đâu? Chi phí nào đáng kể? |
| Error Analysis | Wrong samples, class pairs, confidence, nội dung mẫu | Lỗi có pattern gì? Do mô hình hay do dữ liệu? Cần cải thiện gì? |
| Comparison | Performance + cost + error behavior | Mô hình nào phù hợp nhất và vì sao? |

Các quyết định phải dựa trên kết quả thực nghiệm của chính project; không dùng kết quả Quick Test thay cho kết quả chính thức.

## 2.1. Quy trình ra quyết định kỹ thuật

Thiết kế của hệ thống không dừng ở việc chạy ba mô hình mà sử dụng kết quả của từng giai đoạn để đưa ra quyết định cho giai đoạn tiếp theo:

```text
EDA
  ↓
Observation
  ↓
Decision về dữ liệu / preprocessing / max_len
  ↓
Training + Validation Monitoring
  ↓
Interpretation
  ↓
Decision về checkpoint / cấu hình huấn luyện
  ↓
Test Evaluation
  ↓
Interpretation về performance + cost
  ↓
Error Analysis
  ↓
Decision về hạn chế / nguyên nhân lỗi
  ↓
Model Comparison
  ↓
Final Model Decision
```

### Nguyên tắc

- **EDA → Decision:** mọi lựa chọn về dữ liệu, preprocessing và sequence length phải có căn cứ từ EDA.
- **Training Monitoring → Decision:** checkpoint được chọn dựa trên validation, không dựa trên test.
- **Evaluation → Interpretation:** mỗi metric phải được giải thích về ý nghĩa đối với bài toán, không chỉ liệt kê giá trị.
- **Error Analysis → Decision:** lỗi phải được nhóm theo pattern/class pair và dùng để xác định hạn chế hoặc hướng cải thiện.
- **Comparison → Final Decision:** mô hình cuối được chọn dựa trên performance, computational cost và error behavior.

Quy trình này giúp thể hiện phần phân tích và quyết định kỹ thuật của nhóm thay vì chỉ trình bày kết quả chạy code.

## 2. Các module

### Module 1 – Phân loại chủ đề tin tức

Phân loại bài báo thuộc một trong bốn nhóm chủ đề: World, Sports, Business hoặc Sci/Tech.

### Module 2 – Phân loại dựa trên Transformer

Sử dụng mô hình `distilbert-base-uncased` để tạo biểu diễn văn bản có ngữ cảnh phục vụ nhiệm vụ phân loại chủ đề.

### Module 3 – Frozen Encoder

Đóng băng toàn bộ tham số của Transformer Encoder và chỉ huấn luyện Classification Head.

### Module 4 – Fine-Tuning

Cho phép cập nhật tham số của Transformer Encoder và Classification Head trong quá trình huấn luyện.

### Module 5 – Đánh giá mô hình

Tính các chỉ số Accuracy, Precision, Recall, Macro F1-score và Confusion Matrix. Ngoài ra, hệ thống ghi nhận các chỉ số về chi phí tính toán như Training Time, Inference Time, Trainable Parameters và Model Size khi các module thực nghiệm hoàn thiện.

### Module 6 – So sánh mô hình

So sánh ba cấu hình:

- TF-IDF + Logistic Regression.
- Frozen Encoder.
- Fine-Tuning.

Việc so sánh tập trung vào hiệu năng phân loại và chi phí tính toán.

### Module 7 – Phân tích lỗi

Phân tích các mẫu bị phân loại sai, xác định các cặp lớp thường bị nhầm lẫn và xem xét những nguyên nhân có thể dẫn đến lỗi phân loại.

## 3. Cấu trúc thư mục

```text
Course-Work-6/
│
├── README.md
├── .gitignore
├── requirements.txt
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
│
├── results/
│   ├── metrics/
│   ├── figures/
│   └── checkpoints/
│
└── report/
```

| Thư mục / File | Vai trò |
|---|---|
| `README.md` | Mô tả dự án, hướng dẫn cài đặt và cách chạy |
| `requirements.txt` | Danh sách thư viện cần cài đặt |
| `.gitignore` | Xác định các file và thư mục không đưa lên Git |
| `docs/` | Tài liệu phân tích, thiết kế và nghiên cứu |
| `data/` | Dữ liệu sử dụng trong project |
| `data/raw/` | Dữ liệu đầu vào chưa qua xử lý nếu nhóm sử dụng file raw |
| `data/processed/` | Dữ liệu sau xử lý nếu cần lưu |
| `src/` | Mã nguồn của hệ thống |
| `notebooks/` | Notebook EDA, huấn luyện, đánh giá và so sánh |
| `results/metrics/` | Các file kết quả định lượng |
| `results/figures/` | Biểu đồ và confusion matrix |
| `results/checkpoints/` | Checkpoint của các mô hình |
| `report/` | Báo cáo và bài thuyết trình |

### 3.1. Các file trong `src/`

| File | Nội dung |
|---|---|
| `data.py` | Nạp dữ liệu AG News, tiền xử lý và chuẩn bị dữ liệu cho các mô hình |
| `baseline.py` | Cài đặt và huấn luyện mô hình TF-IDF + Logistic Regression |
| `models.py` | Xây dựng mô hình Transformer, Mean Pooling và Classification Head; hỗ trợ cấu hình Frozen Encoder |
| `train.py` | Module dành cho quy trình huấn luyện mô hình Transformer |
| `evaluate.py` | Module dành cho đánh giá mô hình và tính các chỉ số |
| `compare.py` | Module dành cho tổng hợp và so sánh kết quả các mô hình |
| `predict.py` | Module dành cho dự đoán văn bản tin tức mới |

Các module `train.py`, `evaluate.py`, `compare.py` và `predict.py` được tổ chức để phục vụ các bước huấn luyện, đánh giá, so sánh và dự đoán của hệ thống. Chi tiết implementation được cập nhật theo quá trình phát triển project.

## 4. Dữ liệu và chia tập

### 4.1. Nguồn dữ liệu

Dự án sử dụng tập dữ liệu **AG News**, được tải thông qua thư viện Hugging Face Datasets với dataset `fancyzhx/ag_news`.

Dataset gồm hai tập chính:

- Training set: 120.000 mẫu.
- Test set: 7.600 mẫu.

Dữ liệu thuộc bốn nhóm chủ đề:

- World.
- Sports.
- Business.
- Sci/Tech.

### 4.2. Trường dữ liệu

Dữ liệu được sử dụng trong pipeline hiện tại gồm:

- `text`: nội dung văn bản tin tức.
- `label`: nhãn tương ứng với chủ đề của tin tức.

Mapping nhãn:

| Label | Chủ đề |
|---:|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

### 4.3. Chia tập dữ liệu

Tập training được chia thành training set và validation set để phục vụ quá trình huấn luyện và theo dõi hiệu năng.

Tập test được giữ riêng và chỉ sử dụng cho đánh giá cuối cùng.

Việc chia dữ liệu được thực hiện theo hướng stratified split nhằm duy trì phân bố nhãn giữa các tập.

Random seed được sử dụng để hỗ trợ khả năng tái lập kết quả.

### 4.4. Tiền xử lý

Dữ liệu văn bản được chuẩn hóa trước khi đưa vào tokenizer.

Đối với Transformer `distilbert-base-uncased`, tokenizer tương ứng xử lý văn bản theo cơ chế của mô hình uncased.

Trong quá trình tokenization, văn bản được chuyển thành `input_ids` và `attention_mask`.

Đối với Transformer, các token padding được nhận diện thông qua `attention_mask` và không được tính vào Mean Pooling.

## 5. Luồng dữ liệu

### 5.1. Suy luận (Inference)

```text
Văn bản tin tức
      |
Tiền xử lý
      |
Tokenizer
      |
input_ids + attention_mask
      |
DistilBERT Encoder
      |
last_hidden_state
      |
Mean Pooling
      |
Classification Head
      |
4 giá trị logit
      |
Softmax
      |
Danh mục dự đoán + xác suất
```

### 5.2. Huấn luyện (Training)

```text
Văn bản tin tức + Nhãn
      |
Tiền xử lý
      |
Tokenization
      |
input_ids + attention_mask
      |
DistilBERT Encoder
      |
Mean Pooling
      |
Classification Head
      |
Logit dự đoán
      |
Cross-Entropy Loss
      |
Lan truyền ngược
      |
Cập nhật tham số
```

Đối với Frozen Encoder, các tham số của Transformer Encoder được đóng băng và không nhận gradient. Chỉ các tham số của Classification Head được cập nhật.

Đối với Fine-Tuning, các tham số của Transformer Encoder và Classification Head đều được phép cập nhật trong quá trình huấn luyện.

## 6. Cấu hình thí nghiệm

### 6.1. Frozen Encoder

Cấu hình hiện tại của Frozen Encoder được triển khai trong project như sau:

| Tham số | Giá trị |
|---|---|
| Mô hình | `distilbert-base-uncased` |
| Phần được huấn luyện | Classification Head |
| Encoder | Frozen |
| Learning rate | `1e-3` |
| Optimizer | AdamW |
| Batch size trích xuất đặc trưng | `64` |
| Batch size huấn luyện Classification Head | `256` |
| Số epoch | `20` |
| Độ dài chuỗi tối đa | `64` |
| Random seed | `42` |
| Loss function | Cross-Entropy Loss |

Cấu hình trên được sử dụng cho implementation Frozen Encoder hiện tại.

### 6.2. Fine-Tuning

Fine-Tuning sử dụng cùng backbone `distilbert-base-uncased` và cùng pipeline dữ liệu với Frozen Encoder để đảm bảo khả năng so sánh.

Các hyperparameter của Fine-Tuning sẽ được xác định và ghi nhận khi module Fine-Tuning được triển khai và chạy thực nghiệm.

Các nguyên tắc chung:

- Không đóng băng Transformer Encoder.
- Transformer Encoder và Classification Head đều trainable.
- Sử dụng Cross-Entropy Loss.
- Sử dụng AdamW.
- Sử dụng cùng tập test với Frozen Encoder.
- Ghi nhận các chỉ số Accuracy, Precision, Recall, Macro F1-score, Training Time, Inference Time, Trainable Parameters và Model Size.

### 6.3. Điều kiện so sánh

Để đảm bảo so sánh công bằng giữa các mô hình, các cấu hình thực nghiệm cần thống nhất:

- Dataset.
- Cách chia train/validation/test.
- Label mapping.
- Tiền xử lý.
- Tokenizer.
- Backbone Transformer.
- Random seed khi phù hợp.
- Tập test đánh giá cuối cùng.
- Các chỉ số đánh giá.

Các hyperparameter cụ thể của từng mô hình được ghi lại cùng với kết quả thực nghiệm.

## 7. Giao diện các module

Các giao diện dưới đây mô tả chức năng dự kiến của từng module. Tên hàm và tham số có thể được điều chỉnh trong quá trình triển khai nhưng phải đảm bảo thống nhất giữa các module.

### 7.1. Dataset

```python
load_dataset()
```

Nạp dữ liệu AG News và chuẩn bị các tập dữ liệu phục vụ huấn luyện, validation và test.

### 7.2. Tokenizer

```python
tokenize(text)
```

Chuyển văn bản thành `input_ids` và `attention_mask`.

### 7.3. Xây dựng mô hình

```python
build_model(model_name, freeze_encoder)
```

Tạo mô hình Transformer với Classification Head. Tham số `freeze_encoder` xác định Transformer Encoder được đóng băng hay được phép huấn luyện.

### 7.4. Dự đoán

```python
predict(text)
```

Nhận một văn bản tin tức và trả về lớp dự đoán cùng xác suất tương ứng.

### 7.5. Huấn luyện

```python
train(model, train_data, validation_data)
```

Thực hiện quá trình huấn luyện mô hình trên training set và theo dõi hiệu năng trên validation set.

### 7.6. Đánh giá

```python
evaluate(model, test_data)
```

Tính các chỉ số đánh giá trên test set, bao gồm Accuracy, Precision, Recall, Macro F1-score và Confusion Matrix.

### 7.7. So sánh

```python
compare(results_baseline, results_frozen, results_finetuned)
```

Tổng hợp kết quả của Baseline, Frozen Encoder và Fine-Tuning để tạo bảng và biểu đồ so sánh.

## 8. Phương pháp đánh giá và đo lường

### 8.1. Hiệu năng phân loại

Các mô hình được đánh giá trên tập test bằng:

- Accuracy.
- Precision.
- Recall.
- Macro F1-score.
- Confusion Matrix.

Kết quả của TF-IDF + Logistic Regression được sử dụng làm baseline để so sánh với Frozen Encoder và Fine-Tuning.

### 8.2. Hiệu năng tính toán

| Chỉ số | Cách đo |
|---|---|
| Training Time | Tổng thời gian thực hiện quá trình huấn luyện |
| Inference Time | Tổng thời gian xử lý tập test; có thể quy đổi sang thời gian trung bình trên mỗi mẫu |
| Trainable Parameters | Số tham số có `requires_grad = True` |
| Model Size | Dung lượng checkpoint/model được lưu |

### 8.3. Phân tích lỗi

Phân tích lỗi bao gồm:

- Xác định các cặp lớp có mức độ nhầm lẫn cao từ Confusion Matrix.
- Kiểm tra một số mẫu bị phân loại sai.
- Xem xét các nguyên nhân có thể dẫn đến lỗi.
- Phân tích trường hợp nội dung có đặc điểm của nhiều chủ đề.
- Xem xét ảnh hưởng của việc giới hạn độ dài chuỗi đối với các văn bản dài.

## 9. Kết quả đầu ra

| Đầu ra | Vị trí |
|---|---|
| Metrics | `results/metrics/` |
| Biểu đồ và Confusion Matrix | `results/figures/` |
| Checkpoint mô hình | `results/checkpoints/` |
| Notebook EDA, huấn luyện và so sánh | `notebooks/` |
| Báo cáo và slide | `report/` |
| Hướng dẫn cài đặt và chạy | `README.md` |

Các file dữ liệu lớn và checkpoint có dung lượng lớn không nên được đưa trực tiếp lên GitHub. Các file này được quản lý thông qua `.gitignore` theo quy định của project.

## 10. Các mốc thực hiện

### Mốc 1 – Chuẩn bị dữ liệu

- Nạp bộ dữ liệu AG News.
- Khảo sát dữ liệu.
- Kiểm tra phân bố nhãn.
- Chuẩn bị training, validation và test set.
- Thực hiện tiền xử lý.
- Chuẩn hóa label mapping.
- Chuẩn bị tokenizer cho Transformer.

### Mốc 2 – Baseline

- Chuẩn bị dữ liệu văn bản cho baseline.
- Chuyển văn bản thành đặc trưng TF-IDF.
- Huấn luyện Logistic Regression.
- Đánh giá baseline.
- Lưu metrics và các kết quả cần thiết.

### Mốc 3 – Frozen Encoder

- Nạp `distilbert-base-uncased`.
- Nạp tokenizer tương ứng.
- Đóng băng Transformer Encoder.
- Thực hiện Mean Pooling.
- Xây dựng Classification Head.
- Huấn luyện Classification Head.
- Lưu checkpoint và metrics.

### Mốc 4 – Fine-Tuning

- Sử dụng cùng backbone và pipeline dữ liệu với Frozen Encoder.
- Không đóng băng Transformer Encoder.
- Huấn luyện Transformer Encoder cùng Classification Head.
- Theo dõi kết quả trên validation set.
- Lưu checkpoint và metrics.

### Mốc 5 – Đánh giá

- Tính Accuracy.
- Tính Precision.
- Tính Recall.
- Tính Macro F1-score.
- Tạo Confusion Matrix.
- Đo Training Time.
- Đo Inference Time.
- Đếm Trainable Parameters.
- Xác định Model Size.

### Mốc 6 – So sánh và phân tích lỗi

- So sánh TF-IDF + Logistic Regression, Frozen Encoder và Fine-Tuning.
- So sánh hiệu năng phân loại.
- So sánh chi phí tính toán.
- Phân tích trade-off giữa hiệu năng và chi phí.
- Phân tích các mẫu bị phân loại sai.

### Mốc 7 – Tài liệu và demo

- Cập nhật README.
- Kiểm tra Define.md và Design.md.
- Ghi lại môi trường và cấu hình thực nghiệm.
- Tổng hợp kết quả.
- Chuẩn bị demo.
- Hoàn thiện báo cáo.
- Hoàn thiện bài thuyết trình.

## 11. Rủi ro kỹ thuật

### Rủi ro 1 – Chi phí tính toán cao

Transformer có thể yêu cầu nhiều tài nguyên tính toán. Có thể giảm batch size, giới hạn độ dài chuỗi và sử dụng GPU hoặc thiết bị tăng tốc phù hợp khi cần.

### Rủi ro 2 – Overfitting

Theo dõi hiệu năng trên validation set và sử dụng các biện pháp phù hợp như early stopping hoặc điều chỉnh số epoch khi cần.

### Rủi ro 3 – Văn bản đầu vào dài

Văn bản có độ dài vượt quá giới hạn của mô hình có thể bị cắt bằng truncation. Cần sử dụng cùng quy tắc xử lý giữa các mô hình để đảm bảo tính nhất quán.

### Rủi ro 4 – So sánh không công bằng

Các mô hình cần sử dụng cùng dataset, cách chia dữ liệu, preprocessing, label mapping và test set.

### Rủi ro 5 – Nhầm lẫn giữa các lớp

Một số tin tức có thể chứa nội dung liên quan đến nhiều chủ đề. Confusion Matrix và error analysis được sử dụng để xác định các cặp lớp thường bị nhầm lẫn.

### Rủi ro 6 – Khả năng tái lập

Cần ghi lại random seed, tên mô hình, hyperparameter, cách chia dữ liệu và môi trường phần mềm để hỗ trợ tái lập thực nghiệm.

### Rủi ro 7 – Chi phí và hiệu năng của Transformer

`distilbert-base-uncased` được sử dụng làm backbone nhằm cân bằng giữa khả năng biểu diễn ngữ cảnh và chi phí tính toán. Hiệu năng, thời gian huấn luyện và kích thước mô hình cần được ghi nhận để đánh giá mức độ phù hợp.

### Rủi ro 8 – Dung lượng file lớn trên GitHub

Dữ liệu lớn và checkpoint có dung lượng lớn không nên đưa trực tiếp lên repository. Các file này cần được đưa vào `.gitignore` và README cần mô tả cách chuẩn bị hoặc tải dữ liệu khi cần.


## 4. Decision Evidence Checklist

Trước khi kết luận, nhóm phải kiểm tra đủ bốn chuỗi bằng chứng sau:

| Chuỗi | Bằng chứng tối thiểu | Kết luận cần ghi |
|---|---|---|
| EDA → Decision | Missing/duplicate, class distribution, token coverage | Vì sao giữ/xử lý dữ liệu và chọn `max_len` |
| Training Monitoring → Decision | Train/validation loss + accuracy theo epoch | Có overfitting không, best epoch nào, vì sao chọn checkpoint |
| Evaluation → Interpretation | Metrics + confusion matrix + cost | Mô hình mạnh/yếu ở đâu và trade-off là gì |
| Error Analysis → Decision | Class pair + wrong samples + nguyên nhân | Pattern lỗi nào quan trọng và cần cải thiện gì |

**Nguyên tắc:** nếu một quyết định không có bằng chứng từ kết quả chạy tương ứng, không trình bày quyết định đó như một kết luận thực nghiệm.
