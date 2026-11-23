# So sánh mô hình (Model Comparison)

## 1. Mục tiêu

Phần này so sánh ba phương pháp trên cùng bài toán phân loại AG News:

1. TF-IDF + Logistic Regression (Baseline)
2. Frozen Encoder với DistilBERT
3. Fine-Tuning với DistilBERT

Mục tiêu không chỉ là xác định mô hình có Accuracy cao nhất mà còn phân tích trade-off giữa:

- Hiệu năng phân loại trên tập test
- Khả năng thích nghi với bài toán
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

> Các giá trị `-` được giữ nguyên vì kết quả tổng hợp hiện tại không lưu riêng Macro Precision và Macro Recall cho Baseline và Frozen Encoder. Nhóm không suy diễn các giá trị này từ Accuracy.

### Xếp hạng theo Accuracy

| Hạng | Model | Accuracy |
|---|---|---:|
| 1 | **Fine-Tuning** | **94.22%** |
| 2 | TF-IDF + Logistic Regression | **91.96%** |
| 3 | Frozen Encoder | **90.29%** |

Fine-Tuning cải thiện Accuracy:

- So với Baseline: **+2.26 điểm phần trăm**
- So với Frozen Encoder: **+3.93 điểm phần trăm**

Kết quả này cho thấy Fine-Tuning giúp DistilBERT thích nghi tốt hơn với đặc trưng của bài toán AG News.

---

## 3. Baseline – TF-IDF + Logistic Regression

Baseline đạt:

- Accuracy: **91.96%**
- Macro F1: **91.94%**

Đây là kết quả đáng chú ý vì mô hình truyền thống đạt hiệu năng cao hơn Frozen Encoder.

Điều này cho thấy AG News có nhiều tín hiệu phân loại nằm trong các từ khóa và cụm từ đặc trưng của từng chủ đề. TF-IDF có thể khai thác tốt các tín hiệu này mà không cần cập nhật một Transformer.

### Interpretation

Baseline cho thấy một phương pháp truyền thống vẫn có thể đạt hiệu năng cao trên bài toán phân loại văn bản nhiều lớp.

Đồng thời, kết quả này cung cấp một mốc tham chiếu quan trọng để đánh giá liệu việc sử dụng Transformer có thực sự mang lại lợi ích hay không.

### Decision

Nhóm giữ Baseline làm:

- Mốc tham chiếu cho toàn bộ thực nghiệm.
- Phương án có chi phí mô hình thấp.
- Phương án fallback khi tài nguyên tính toán hạn chế.

---

## 4. Frozen Encoder

Frozen Encoder đạt:

- Accuracy: **90.29%**
- Macro F1: **90.28%**

Kết quả thấp hơn Baseline:

**91.96% − 90.29% = 1.67 điểm phần trăm**

Trong cấu hình này, encoder DistilBERT được giữ nguyên và chỉ classification head được huấn luyện.

Do đó, representation pretrained được sử dụng gần như nguyên trạng trong khi classification head phải học cách phân biệt bốn chủ đề của AG News.

### Interpretation

Kết quả cho thấy pretrained representation không phải lúc nào cũng phù hợp tối ưu với một task cụ thể nếu encoder không được điều chỉnh theo dữ liệu của task đó.

Error Analysis cũng cho thấy Frozen Encoder gặp nhiều khó khăn hơn ở các lớp có nội dung giao nhau, đặc biệt là:

```text
Business ↔ Sci/Tech
```

### Decision

Frozen Encoder không được chọn làm mô hình cuối vì:

- Accuracy thấp hơn Baseline.
- Macro F1 thấp hơn Baseline.
- Không mang lại lợi ích hiệu năng tương xứng với việc sử dụng DistilBERT.

Tuy nhiên, Frozen Encoder vẫn được giữ trong thực nghiệm vì nó giúp nhóm kiểm tra một câu hỏi quan trọng:

```text
Pretrained Representation
        vs.
Task-specific Fine-Tuning
```

Qua đó có thể đánh giá trực tiếp tác động của việc cập nhật encoder đối với kết quả phân loại.

---

## 5. Fine-Tuning

Fine-Tuning đạt:

- Accuracy: **94.22%**
- Macro Precision: **94.24%**
- Macro Recall: **94.22%**
- Macro F1: **94.23%**

Đây là kết quả cao nhất trong ba phương pháp.

### Cấu hình thực nghiệm

| Parameter | Configuration |
|---|---|
| Model | DistilBERT |
| Learning Rate | `2e-5` |
| Batch Size | `32` |
| Epochs | `3` |
| Max Length | `64` |
| Best Validation Epoch | **2** |
| Trainable Components | Transformer + Classification Head |
| Device | GPU |

Thời gian huấn luyện chính thức được ghi nhận khoảng **2.353 giây (~39,2 phút)**.

### Training Monitoring → Decision

Trong quá trình Fine-Tuning, nhóm theo dõi validation performance thay vì chỉ sử dụng model ở epoch cuối.

Kết quả cho thấy **checkpoint tốt nhất nằm ở epoch 2**, trong khi quá trình huấn luyện được cấu hình tối đa 3 epochs.

Điều này cho thấy validation performance có vai trò trong việc lựa chọn checkpoint, thay vì mặc định sử dụng checkpoint cuối cùng.

### Decision

Nhóm sử dụng checkpoint có validation performance tốt nhất để thực hiện đánh giá trên test set.

Kết quả chính thức của checkpoint được chọn:

- Accuracy: **94.22%**
- Macro F1: **94.23%**

Fine-Tuning vì vậy trở thành ứng viên mạnh nhất cho mô hình cuối.

---

## 6. Error Analysis và sự khác biệt giữa các mô hình

Model Comparison không chỉ dựa trên Accuracy và Macro F1 mà được kết hợp với Error Analysis.

Kết quả Error Analysis cho thấy các mô hình thường gặp khó khăn ở những mẫu có chủ đề giao nhau. Pattern nổi bật là:

```text
Business ↔ Sci/Tech
```

Các bài viết về công ty công nghệ có thể đồng thời chứa:

- thông tin doanh nghiệp;
- sản phẩm;
- thị trường;
- tài chính;
- công nghệ.

Do đó, một prediction sai ở các trường hợp này không phải lúc nào cũng chứng minh mô hình không hiểu nội dung.

### Interpretation

Ba phương pháp có cách khai thác thông tin khác nhau:

**Baseline**

TF-IDF khai thác hiệu quả các từ khóa và cụm từ đặc trưng.

**Frozen Encoder**

DistilBERT cung cấp representation pretrained nhưng encoder không được điều chỉnh theo AG News.

**Fine-Tuning**

Toàn bộ Transformer được cập nhật theo task, giúp representation thích nghi tốt hơn với đặc điểm của dữ liệu AG News.

Error Analysis cho thấy Fine-Tuning giảm tổng số lỗi và cải thiện nhiều pattern lỗi, mặc dù `Business ↔ Sci/Tech` vẫn là một trong những nguồn lỗi chính.

### Decision

Nhóm không lựa chọn mô hình chỉ dựa trên Accuracy.

Quy trình đánh giá được thực hiện theo:

```text
Overall Metrics
      ↓
Classification Report
      ↓
Confusion Matrix
      ↓
Error Analysis
      ↓
Model Comparison
      ↓
Final Decision
```

---

## 7. Trade-off giữa hiệu năng và chi phí

| Tiêu chí | Baseline | Frozen Encoder | Fine-Tuning |
|---|---|---|---|
| Accuracy | 91.96% | 90.29% | **94.22%** |
| Macro F1 | 91.94% | 90.28% | **94.23%** |
| Kiến trúc | TF-IDF + LR | DistilBERT | DistilBERT |
| Transformer trainable | Không | Không | **Có** |
| Trainable parameters | 200,004 | 3,076 | **66,365,956** |
| Chi phí huấn luyện | Thấp | Cao hơn Baseline | **Cao nhất** |
| Khả năng thích nghi task | Hạn chế | Hạn chế | **Cao nhất** |
| Phù hợp làm mô hình cuối | Có thể | Không | **Có** |

### Interpretation

Có sự đánh đổi rõ ràng giữa hiệu năng và chi phí:

- **Baseline** có hiệu năng cao với kiến trúc đơn giản và chi phí tính toán thấp.
- **Frozen Encoder** sử dụng Transformer nhưng không đạt hiệu năng bằng Baseline.
- **Fine-Tuning** đạt hiệu năng cao nhất nhưng cần cập nhật toàn bộ Transformer nên chi phí huấn luyện lớn hơn.

Thời gian thực thi phụ thuộc phần cứng và môi trường chạy. Vì vậy, nhóm không sử dụng thời gian training tuyệt đối để kết luận mô hình nào nhanh hơn nếu điều kiện thực nghiệm không giống nhau.

### Decision

Nếu ưu tiên **hiệu năng phân loại**, Fine-Tuning là lựa chọn phù hợp.

Nếu ưu tiên **đơn giản và tiết kiệm tài nguyên**, Baseline vẫn là phương án có giá trị.

Frozen Encoder không cho thấy lợi thế đủ lớn để được lựa chọn làm mô hình cuối.

---

## 8. Quyết định cuối cùng

Dựa trên:

- Accuracy;
- Macro F1;
- Validation performance;
- Confusion Matrix;
- Error Analysis;
- Trade-off giữa hiệu năng và chi phí;

nhóm đưa ra quyết định:

### Final Model: Fine-Tuning

Fine-Tuning được chọn làm mô hình cuối vì đạt:

**94.22% Accuracy và 94.23% Macro F1**

cao nhất trong ba phương pháp.

Ngoài ra, kết quả Error Analysis cho thấy Fine-Tuning giảm tổng số lỗi và cải thiện nhiều nhóm lỗi so với các phương pháp còn lại.

### Baseline: Reference / Fallback Model

TF-IDF + Logistic Regression được giữ lại vì đạt **91.96% Accuracy**.

Mặc dù thấp hơn Fine-Tuning, Baseline có ưu điểm về:

- kiến trúc đơn giản;
- chi phí tính toán thấp;
- khả năng triển khai dễ hơn;
- hiệu năng vẫn ở mức cao.

### Frozen Encoder: Experimental Model

Frozen Encoder đạt **90.29% Accuracy** và không vượt qua Baseline.

Do đó, mô hình này không được chọn làm mô hình cuối nhưng vẫn có giá trị trong nghiên cứu vì giúp chứng minh sự khác biệt giữa:

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
             ┌────────────┼────────────┐
             ↓            ↓            ↓
         Baseline      Frozen      Fine-Tuning
         91.96%        90.29%        94.22%
             |            |            |
             └────────────┴────────────┘
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

```text
EDA
 ↓
Model Selection
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

Trong đó:

- **EDA** giúp hiểu đặc điểm dữ liệu và đưa ra các quyết định tiền xử lý/cấu hình.
- **Training Monitoring** giúp lựa chọn checkpoint dựa trên validation performance.
- **Evaluation** xác định hiệu năng trên cùng tập test.
- **Error Analysis** xác định các pattern lỗi và nguyên nhân khả dĩ.
- **Model Comparison** tổng hợp hiệu năng và trade-off.
- **Final Decision** lựa chọn mô hình phù hợp với mục tiêu bài toán.

---

## 10. Kết luận

Thực nghiệm cho thấy:

1. **Baseline** đạt hiệu năng cao và là một mốc tham chiếu mạnh với Accuracy **91.96%**.
2. **Frozen Encoder** đạt **90.29%**, thấp hơn Baseline, cho thấy pretrained representation chưa được điều chỉnh theo task chưa đủ để vượt qua phương pháp truyền thống.
3. **Fine-Tuning** đạt kết quả tốt nhất với **94.22% Accuracy** và **94.23% Macro F1**.
4. Error Analysis cho thấy `Business` và `Sci/Tech` vẫn có sự giao thoa về nội dung, đặc biệt ở các bài viết liên quan đến doanh nghiệp công nghệ.
5. Việc lựa chọn mô hình cần xét đồng thời **hiệu năng, pattern lỗi và chi phí**, thay vì chỉ nhìn vào Accuracy.
6. Dựa trên toàn bộ evidence thực nghiệm, **Fine-Tuning DistilBERT được nhóm lựa chọn làm mô hình cuối cùng cho bài toán AG News**.
