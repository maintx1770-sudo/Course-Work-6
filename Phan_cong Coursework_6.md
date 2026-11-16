# PHÂN CÔNG COURSEWORK TOPIC 6 

## News Topic Classification Using Transformer-Based Models



## 1. Nội dung Coursework Topic 6

- Xây dựng hệ thống **News Topic Classification** sử dụng dataset **AG News**.
- Phân loại tin tức thành 4 nhóm:
  - World
  - Sports
  - Business
  - Sci/Tech
- So sánh 3 hướng:
  - **TF-IDF + Logistic Regression**
  - **Frozen Encoder**
  - **Fine-Tuning**
- Input mô hình Transformer: **Title + Description**.
- Transformer mặc định: `distilbert-base-uncased`.
- Đánh giá:
  - Accuracy
  - Precision
  - Recall
  - Macro F1-score
  - Confusion Matrix
  - Training Time
  - Inference Time
  - Trainable Parameters
  - Model Size


---

# 2. Phân công 6 thành viên

## Thứ 

| Thành viên | Hoàn thành |
|---|---|
| **TV1 Mai** | `src/data.py`, `notebooks/01_data_exploration.ipynb`, kiểm tra `data/raw/train.csv`, `data/raw/test.csv`, chuẩn hóa label mapping, train/validation/test split, preprocessing, tokenizer. **Phần đã làm:** `docs/Define.md`, `docs/Design.md`. Kiểm tra lại Define/Design có khớp với code thực tế. |
| **TV2 Huynh** | `notebooks/02_baseline.ipynb`, xây dựng **TF-IDF + Logistic Regression**, lưu kết quả vào `results/metrics/`, `results/figures/`, tính Accuracy, Precision, Recall, Macro F1-score, Confusion Matrix, Training Time, Inference Time. |
| **TV6 Nguyễn Chi** | `README.md`, kiểm tra cấu trúc repository, `requirements.txt`, `.gitignore` nếu cần, `docs/` và `results/` đã có đúng thư mục chưa. **Phần còn lại nếu rảnh:** tạo checklist tích hợp và danh sách các file cần kiểm tra trước khi nộp. |

### Output Thứ 

- TV1 bàn giao **data pipeline + preprocessing + tokenizer configuration**.
- TV2 bàn giao **baseline metrics**.
- TV6 bàn giao **repository checklist**.
- Các thông số dùng chung phải được chốt:
  - Label mapping
  - Train/validation/test split
  - Random seed
  - `max_length`
  - Tokenizer
  - Cấu trúc thư mục

---

## Thứ 

| Thành viên | Hoàn thành |
|---|---|
| **TV3 My** | `src/models.py`, `src/train.py`, `notebooks/03_frozen_encoder.ipynb`, triển khai **Frozen Encoder**, Classification Head, freeze Transformer, training, validation, checkpoint, metrics trong `results/metrics/`, checkpoint trong `results/checkpoints/`. |
| **TV4 Đức** | `src/evaluate.py`, `src/compare.py`, `notebooks/05_comparison.ipynb` phần khung evaluation; `results/tables/`, `results/figures/`; chuẩn bị code tính Accuracy, Precision, Recall, Macro F1-score, Confusion Matrix, Training Time, Inference Time, Trainable Parameters, Model Size. |
| **TV6 Nguyễn Chi** | Kiểm tra integration giữa output của TV1, TV2 và TV3; chuẩn bị test/checklist cho data pipeline, model loading và kết quả training. |

### TV3 – Frozen Encoder cần hoàn thành

- Pretrained Transformer: `distilbert-base-uncased`.
- Classification Head: Dropout + Linear → 4 classes.
- Freeze toàn bộ Transformer Encoder.
- Chỉ Classification Head được train.
- Cross-Entropy Loss.
- AdamW.
- Cấu hình đề xuất:
  - Learning rate: `1e-3`
  - Batch size: `32`
  - Epoch: `3–5`
  - `max_length`: `128`
  - Seed: `42`
- Lưu checkpoint.
- Ghi:
  - Accuracy
  - Precision
  - Recall
  - Macro F1-score
  - Training Time
  - Inference Time
  - Trainable Parameters
  - Model Size

### TV4 – Evaluation chuẩn bị

- Viết hàm evaluation dùng chung.
- Chuẩn hóa format kết quả.
- Chuẩn bị bảng so sánh 3 mô hình:
  - TF-IDF + Logistic Regression
  - Frozen Encoder
  - Fine-Tuning
- Chưa chốt số liệu cuối nếu Fine-Tuning chưa hoàn thành.

---

# Thứ 

| Thành viên | Hoàn thành |
|---|---|
| **TV4 Đức** | Hoàn thiện `src/evaluate.py`, `src/compare.py`, `notebooks/05_comparison.ipynb`; tạo bảng metrics và confusion matrix; chuẩn bị `results/tables/comparison_results.csv`, `results/figures/`. |
| **TV5 Trúc Chi** | `src/predict.py`, notebook/demo prediction, **Error Analysis**, phân tích confusion matrix và các mẫu dự đoán sai; chuẩn bị nội dung nhận xét kết quả. |
| **TV6 Nguyễn Chi** | Kiểm thử project: kiểm tra data loading, preprocessing, model loading, training và evaluation; tạo `tests/` nếu nhóm sử dụng test tự động; ghi lỗi và báo đúng TV phụ trách. |
| **TV3 My** | Hoàn thiện **Fine-Tuning** sau Frozen Encoder nếu chưa xong: `src/models.py`, `src/train.py`, `notebooks/04_fine_tuning.ipynb`, checkpoint và metrics Fine-Tuning. |

### TV3 – Fine-Tuning cần hoàn thành

- Dùng cùng backbone và preprocessing với Frozen Encoder.
- Không freeze Transformer.
- Transformer + Classification Head đều trainable.
- Cross-Entropy Loss.
- AdamW.
- Cấu hình đề xuất:
  - Learning rate: `2e-5`
  - Batch size: `32`
  - Epoch: `3`
  - `max_length`: `128`
  - Seed: `42`
- Đánh giá trên **cùng test set** với Frozen Encoder.
- Lưu đủ metrics:
  - Accuracy
  - Precision
  - Recall
  - Macro F1-score
  - Training Time
  - Inference Time
  - Trainable Parameters
  - Model Size

### TV5 – Prediction + Error Analysis

- Viết hàm:

```text
predict(text)
        ↓
Tokenizer
        ↓
Fine-tuned Model
        ↓
Predicted Class + Probability
```

- Test với một số câu/tin tức mới.
- Dùng confusion matrix để tìm các cặp lớp dễ nhầm.
- Phân tích một số mẫu dự đoán sai.
- Nhận xét nguyên nhân có thể:
  - Chủ đề giao nhau.
  - Từ khóa xuất hiện ở nhiều lớp.
  - Text dài bị cắt bởi `max_length`.
  - Nội dung bài báo có nhiều chủ đề.

### TV6 – Testing

Kiểm tra:

- Dataset loading.
- Label mapping.
- Preprocessing.
- Tokenizer.
- Model loading.
- Frozen Encoder.
- Fine-Tuning.
- Evaluation.
- Prediction.
- File output.
- Đường dẫn trong README.

---

# Thứ 

| Thành viên | Hoàn thành |
|---|---|
| **TV4 Đức** | Chốt `results/tables/comparison_results.csv`, confusion matrix, biểu đồ so sánh; hoàn thiện phần **Evaluation + Comparison + Conclusion**. |
| **TV1 Mai** | Kiểm tra lại `docs/Define.md`, `docs/Design.md`, `README.md`, `requirements.txt`; đảm bảo tài liệu khớp với implementation thực tế. |
| **TV5 Trúc Chi** | Chạy lại `notebooks/05_comparison.ipynb` và prediction/error analysis với số liệu thật; cập nhật hình và nhận xét nếu kết quả thay đổi. |
| **TV6 Nguyễn Chi** | Chạy kiểm tra cuối toàn bộ project; kiểm tra các file, đường dẫn, output và lỗi integration; cập nhật checklist trước khi nộp. |
| **TV2 Huynh, TV3 My** | Sửa lỗi trong file của mình theo báo cáo kiểm tra; xác nhận baseline, Frozen Encoder và Fine-Tuning đều có kết quả cuối. |
| **Cả nhóm** | Pull code, chạy full pipeline, sửa lỗi liên kết, kiểm tra README, kiểm tra GitHub, commit/push bản cuối và chuẩn bị nộp. |

---

# 3. Thứ tự phụ thuộc công việc

```text
TV1 – Dataset / Preprocessing
          ↓
     ┌────┴────┐
     ↓         ↓
TV2 Baseline   TV3 Frozen Encoder
                   ↓
              TV3 Fine-Tuning
                   ↓
            ┌──────┴──────┐
            ↓             ↓
       TV4 Evaluation   TV5 Prediction
            │             │
            └──────┬──────┘
                   ↓
             TV6 Testing
                   ↓
          Thứ 5 – Integration
                   ↓
              Final Project
```

## Dependency chính

1. **TV1 → TV2:** TV2 cần text + label + split thống nhất.
2. **TV1 → TV3:** TV3 cần preprocessing/tokenizer thống nhất.
3. **TV3 → TV4:** TV4 cần output/checkpoint và format metrics.
4. **TV3 → TV5:** TV5 cần model tốt nhất để prediction và error analysis.
5. **TV2 + TV3 + TV5 → TV4:** TV4 cần toàn bộ kết quả để comparison.
6. **Tất cả TV → TV6:** TV6 kiểm tra integration cuối.
7. **Tất cả → Thứ 5:** merge, full pipeline, sửa lỗi và nộp.

---

# 4. Bảng file cuối cùng theo thành viên

| TV | Người phụ trách | File chính |
|---|---|---|
| **TV1** | Mai | `src/data.py`<br>`notebooks/01_data_exploration.ipynb`<br>`docs/Define.md`<br>`docs/Design.md`<br>`README.md` |
| **TV2** | Huynh | `notebooks/02_baseline.ipynb`<br>`results/metrics/`<br>`results/figures/` |
| **TV3** | My | `src/models.py`<br>`src/train.py`<br>`notebooks/03_frozen_encoder.ipynb`<br>`notebooks/04_fine_tuning.ipynb`<br>`results/checkpoints/` |
| **TV4** | Đức | `src/evaluate.py`<br>`src/compare.py`<br>`notebooks/05_comparison.ipynb`<br>`results/tables/`<br>`results/figures/` |
| **TV5** | Trúc Chi | `src/predict.py`<br>`notebooks/` prediction/error analysis<br>`results/figures/` |
| **TV6** | Nguyễn Chi | `tests/`<br>`README/checklist`<br>integration testing<br>final verification |

---

# 5. Kết quả bắt buộc trước khi nộp

## Dataset

- [ ] `train.csv`
- [ ] `test.csv`
- [ ] Label mapping đúng 4 lớp.
- [ ] Train/Validation/Test split thống nhất.
- [ ] Preprocessing thống nhất.
- [ ] Tokenizer thống nhất.

## Baseline

- [ ] TF-IDF + Logistic Regression.
- [ ] Accuracy.
- [ ] Precision.
- [ ] Recall.
- [ ] Macro F1-score.
- [ ] Confusion Matrix.
- [ ] Training Time.
- [ ] Inference Time.

## Frozen Encoder

- [ ] Transformer được freeze.
- [ ] Classification Head được train.
- [ ] Checkpoint.
- [ ] Metrics đầy đủ.
- [ ] Trainable Parameters.

## Fine-Tuning

- [ ] Transformer được trainable.
- [ ] Classification Head được trainable.
- [ ] Checkpoint.
- [ ] Metrics đầy đủ.
- [ ] Trainable Parameters.

## Comparison

- [ ] TF-IDF + Logistic Regression.
- [ ] Frozen Encoder.
- [ ] Fine-Tuning.
- [ ] Accuracy.
- [ ] Precision.
- [ ] Recall.
- [ ] Macro F1-score.
- [ ] Training Time.
- [ ] Inference Time.
- [ ] Trainable Parameters.
- [ ] Model Size.
- [ ] Confusion Matrix.
- [ ] Biểu đồ so sánh.

## Final

- [ ] Prediction demo.
- [ ] Error Analysis.
- [ ] README đúng với code.
- [ ] Define.md đúng.
- [ ] Design.md đúng.
- [ ] Không còn lỗi integration.
- [ ] Full pipeline chạy được.
- [ ] GitHub đầy đủ file cần thiết.
- [ ] Pull → test → commit → push bản cuối.
