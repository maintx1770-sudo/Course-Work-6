# So sánh mô hình (Model Comparison)

## 1. Mục tiêu

Phần này so sánh ba phương pháp trên cùng bài toán phân loại AG News:

1. TF-IDF + Logistic Regression (Baseline)
2. Frozen Encoder với DistilBERT
3. Fine-Tuning với DistilBERT

Mục tiêu không chỉ là xác định mô hình có Accuracy cao nhất mà còn phân tích trade-off giữa:

- Hiệu năng phân loại
- Khả năng tổng quát hóa
- Chi phí huấn luyện
- Số lượng tham số được cập nhật
- Khả năng triển khai

Toàn bộ mô hình được đánh giá trên cùng tập test AG News gồm **7.600 mẫu**.

---

## 2. Kết quả tổng hợp

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | **91.96%** | - | - | **91.94%** |
| Frozen Encoder | **90.29%** | - | - | **90.28%** |
| Fine-Tuning | **94.22%** | **94.24%** | **94.22%** | **94.23%** |

> Các giá trị `-` được giữ nguyên khi kết quả tổng hợp hiện tại không lưu riêng giá trị Macro Precision/Recall cho hai mô hình tương ứng. Không tự suy diễn từ Accuracy.

### Xếp hạng theo Accuracy

| Hạng | Model | Accuracy |
|---|---|---:|
| 1 | Fine-Tuning | **94.22%** |
| 2 | TF-IDF + Logistic Regression | **91.96%** |
| 3 | Frozen Encoder | **90.29%** |

Fine-Tuning cao hơn:

- Baseline: **+2.26 điểm phần trăm**
- Frozen Encoder: **+3.93 điểm phần trăm**

---

## 3. Phân tích kết quả

### 3.1. Baseline – TF-IDF + Logistic Regression

Baseline đạt:

- Accuracy: **91.96%**
- Macro F1: **91.94%**

Đây là kết quả đáng chú ý vì một mô hình truyền thống có hiệu năng cao hơn Frozen Encoder.

Điều này cho thấy bài toán AG News có nhiều tín hiệu phân loại nằm trong từ khóa và cụm từ đặc trưng của từng chủ đề. TF-IDF có thể khai thác tốt các tín hiệu này mà không cần cập nhật một Transformer.

### Interpretation

Baseline có hiệu năng tốt trong khi chi phí mô hình thấp hơn Transformer. Vì vậy, việc sử dụng Transformer không mặc nhiên đảm bảo kết quả tốt hơn.

### Decision

Baseline được giữ lại làm:

- Mốc tham chiếu cho toàn bộ thí nghiệm.
- Phương án triển khai nhẹ.
- Phương án fallback khi tài nguyên tính toán hạn chế.

---

## 4. Frozen Encoder

Frozen Encoder đạt:

- Accuracy: **90.29%**
- Macro F1: **90.28%**

Kết quả thấp hơn Baseline khoảng:

**91.96% − 90.29% = 1.67 điểm phần trăm**

Điều này cho thấy việc sử dụng pretrained Transformer nhưng đóng băng toàn bộ encoder chưa đủ để thích nghi tốt với đặc trưng của AG News.

Trong cấu hình này, Transformer chủ yếu cung cấp representation có sẵn, còn classification head phải học cách phân biệt bốn chủ đề dựa trên representation đó.

### Interpretation

Kết quả cho thấy representation pretrained không phải lúc nào cũng tối ưu cho một task cụ thể nếu encoder không được điều chỉnh.

Đặc biệt, Error Analysis cho thấy Frozen Encoder vẫn gặp nhiều lỗi ở các lớp có nội dung giao nhau, đặc biệt là:

`Business ↔ Sci/Tech`

### Decision

Frozen Encoder không được chọn làm mô hình cuối vì:

- Accuracy thấp hơn Baseline.
- Macro F1 thấp hơn Baseline.
- Không mang lại lợi ích hiệu năng tương xứng với việc sử dụng Transformer.

Tuy nhiên, cấu hình này vẫn có giá trị thực nghiệm vì nó cho phép nhóm kiểm tra tác động của việc **freeze encoder** trước khi chuyển sang Fine-Tuning.

---

## 5. Fine-Tuning

Fine-Tuning đạt:

- Accuracy: **94.22%**
- Macro Precision: **94.24%**
- Macro Recall: **94.22%**
- Macro F1: **94.23%**

Đây là kết quả cao nhất trong ba phương pháp.

Mô hình sử dụng:

- DistilBERT
- Learning rate: `2e-5`
- Batch size: `32`
- Epochs: `3`
- Max length: `64`
- Best validation epoch: **2**
- Trainable Transformer + Classification Head
- Training trên GPU

Thời gian huấn luyện chính thức được ghi nhận khoảng **2.353 giây (~39,2 phút)**.

### Training Monitoring → Decision

Trong quá trình Fine-Tuning, nhóm theo dõi validation performance thay vì chỉ chờ đến epoch cuối.

Kết quả cho thấy checkpoint tốt nhất nằm ở **epoch 2**, trong khi quá trình huấn luyện được cấu hình 3 epochs.

Điều này cho thấy việc theo dõi validation là cần thiết để lựa chọn checkpoint tốt nhất thay vì mặc định sử dụng model sau epoch cuối.

### Decision

Nhóm chọn checkpoint tốt nhất dựa trên validation performance để đánh giá trên test set.

---

## 6. So sánh Error Analysis

Kết quả Error Analysis cho thấy các mô hình thường gặp khó khăn ở những mẫu có chủ đề giao nhau.

Đặc biệt:

`Business ↔ Sci/Tech`

Đây là vấn đề quan trọng vì các bài báo về công ty công nghệ có thể đồng thời chứa:

- thông tin doanh nghiệp,
- sản phẩm,
- thị trường,
- tài chính,
- công nghệ.

Do đó, chỉ dựa vào Accuracy tổng chưa đủ để hiểu sự khác biệt giữa các mô hình.

### Interpretation

Baseline có khả năng khai thác tốt các từ khóa đặc trưng.

Frozen Encoder có representation pretrained nhưng encoder không được điều chỉnh theo task.

Fine-Tuning cho phép toàn bộ Transformer thích nghi với đặc điểm của AG News, từ đó cải thiện khả năng phân biệt các lớp có ranh giới ngữ nghĩa gần nhau.

### Decision

Khi đánh giá mô hình cuối, nhóm sử dụng đồng thời:

```text
Overall Metrics
      ↓
Confusion Matrix
      ↓
Error Analysis
      ↓
Model Comparison
      ↓
Final Decision
```

Không lựa chọn mô hình chỉ dựa trên một chỉ số Accuracy.

---

## 7. Trade-off giữa hiệu năng và chi phí

| Tiêu chí | Baseline | Frozen Encoder | Fine-Tuning |
|---|---|---|---|
| Hiệu năng | Cao | Thấp nhất | **Cao nhất** |
| Kiến trúc | Đơn giản | Transformer | Transformer |
| Transformer trainable | Không | Không | **Có** |
| Chi phí huấn luyện | Thấp | Cao hơn Baseline | **Cao nhất** |
| Khả năng thích nghi task | Thấp hơn | Hạn chế | **Cao nhất** |
| Phù hợp làm mô hình cuối | Có thể | Không | **Có** |

> Thời gian thực thi phụ thuộc phần cứng và môi trường chạy, vì vậy không dùng thời gian tuyệt đối giữa các môi trường khác nhau để kết luận về tốc độ nếu điều kiện chạy không giống nhau.

---

## 8. Quyết định cuối cùng

Dựa trên:

- Accuracy
- Macro F1
- Validation performance
- Confusion Matrix
- Error Analysis
- Trade-off giữa hiệu năng và chi phí

nhóm đưa ra quyết định:

### Final Model: Fine-Tuning

Fine-Tuning được chọn làm mô hình cuối vì đạt:

**94.22% Accuracy và 94.23% Macro F1**

cao nhất trong ba phương pháp.

### Baseline: Reference/Fallback Model

TF-IDF + Logistic Regression được giữ lại vì đạt **91.96% Accuracy**, cho thấy một phương pháp truyền thống vẫn có hiệu quả cao và có ưu điểm về tính đơn giản, chi phí tính toán và khả năng triển khai.

### Frozen Encoder: Experimental Model

Frozen Encoder đạt **90.29% Accuracy** và không vượt qua Baseline.

Do đó, mô hình này không được chọn làm mô hình cuối nhưng vẫn cần giữ trong nghiên cứu vì nó giúp chứng minh sự khác biệt giữa:

```text
Pretrained Representation
        vs.
Task-specific Fine-Tuning
```

---

## 9. Final Technical Decision

Kết quả cuối cùng của nhóm:

```text
                 AG News
                    |
          ┌─────────┼─────────┐
          ↓         ↓         ↓
       Baseline   Frozen   Fine-Tuning
       91.96%     90.29%     94.22%
          |         |          |
          └─────────┴──────────┘
                    ↓
             Error Analysis
                    ↓
             Model Comparison
                    ↓
          ┌───────────────────┐
          │   Fine-Tuning     │
          │   Final Model     │
          └───────────────────┘
```

Quyết định này không chỉ dựa trên việc Fine-Tuning có Accuracy cao nhất mà dựa trên toàn bộ bằng chứng thực nghiệm:

**Training Monitoring → Evaluation → Error Analysis → Comparison → Decision**

---

## 10. Kết luận

Thực nghiệm cho thấy:

1. Baseline đạt hiệu năng cao và là một mốc tham chiếu mạnh.
2. Frozen Encoder không vượt qua Baseline khi encoder không được điều chỉnh theo task.
3. Fine-Tuning giúp DistilBERT thích nghi với bài toán AG News và đạt kết quả tốt nhất.
4. Error Analysis cho thấy các lớp `Business` và `Sci/Tech` vẫn có sự giao thoa về nội dung.
5. Việc lựa chọn mô hình cần xét cả hiệu năng và chi phí tính toán thay vì chỉ nhìn Accuracy.

**Mô hình được nhóm lựa chọn cho bài toán AG News là Fine-Tuning DistilBERT.**
