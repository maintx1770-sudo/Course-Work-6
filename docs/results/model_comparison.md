# So sánh mô hình

> **Trạng thái:** đã có Baseline và Frozen Encoder (chạy trên dữ liệu đầy đủ).
> **Fine-Tuning chưa có kết quả chính thức** – dòng tương ứng để trống, cần điền sau khi chạy `04_fine_tuning.ipynb` với `QUICK_TEST = False` rồi chạy lại `05_comparison.ipynb`.

Tập đánh giá: AG News test, 7.600 mẫu (mỗi lớp 1.900). Train/Val = 108.000 / 12.000, seed = 42.
Nguồn số liệu: `results/metrics/evaluation_results.csv`. Hình: `results/figures/model_comparison.png`.

| Model | Accuracy | Precision | Recall | Macro F1 | Training Time | Inference Time | Trainable Params | Model Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.9180 | 0.9179 | 0.9180 | 0.9178 | 14,8 s | 0,34 s | 200.004 | 3,41 MB |
| Frozen Encoder (DistilBERT) | 0.9034 | 0.9036 | 0.9034 | 0.9032 | 3.289,5 s | 182,5 s | 3.076 | 253,17 MB |
| Fine-Tuning (DistilBERT) | _chưa chạy_ | | | | | | 66.365.956 | 253,17 MB |

(Với Fine-Tuning, số tham số và kích thước mô hình lấy từ cấu hình DistilBERT đã xác nhận trong lần chạy thử; các chỉ số còn lại chờ lần chạy chính thức.)

## Nhận xét (Baseline vs Frozen Encoder)

- **Baseline thắng Frozen Encoder về chất lượng:** Macro F1 0,9178 so với 0,9032 (cao hơn khoảng 1,5 điểm phần trăm). Đây là kết quả khá bất ngờ nhưng hợp lý: AG News là bài toán chủ đề, từ khóa (unigram + bigram TF-IDF) đã mang rất nhiều tín hiệu; còn Frozen Encoder chỉ học một head tuyến tính (3.076 tham số) trên đặc trưng DistilBERT không được điều chỉnh cho miền tin tức.
- **Chi phí:** Baseline huấn luyện chỉ ~15 giây trên CPU, suy luận 0,34 giây cho toàn bộ tập test; Frozen Encoder tốn ~55 phút (hầu hết là trích xuất đặc trưng, ~3.257 s, còn train head chỉ ~33 s) và ~183 giây suy luận do phải chạy DistilBERT.
- **Số tham số huấn luyện:** Frozen Encoder rất ít (3.076) nhưng mô hình vẫn phải lưu/tải toàn bộ encoder (~253 MB) khi suy luận. Baseline có nhiều tham số học hơn (200.004) nhưng nhỏ gọn (3,4 MB).
- **Kỳ vọng với Fine-Tuning:** cần kiểm chứng bằng lần chạy chính thức. Không nên kết luận trước khi có số liệu.

> Lưu ý khi so sánh thời gian: các lần chạy dùng thiết bị khác nhau/thiết lập khác nhau (Baseline `cpu`), nên thời gian chỉ mang tính tham khảo tương đối.


## 3. Evaluation → Interpretation → Decision

### 3.1. Interpretation của các kết quả đã hoàn thành

**Baseline – TF-IDF + Logistic Regression** đạt Accuracy 0,9180 và Macro F1 0,9178. Hai chỉ số gần nhau, phù hợp với việc các lớp trong tập test được cân bằng. Kết quả cho thấy một baseline dựa trên từ khóa/ngữ cảnh cục bộ đã rất mạnh đối với AG News.

**Frozen Encoder – DistilBERT** đạt Accuracy 0,9034 và Macro F1 0,9032, thấp hơn Baseline khoảng 1,45 điểm phần trăm về Accuracy và 1,46 điểm phần trăm về Macro F1. Điều này cho thấy việc dùng representation pretrained nhưng chỉ huấn luyện classification head chưa đủ để thích nghi hoàn toàn với ranh giới chủ đề của AG News.

Về chi phí, Baseline có training time khoảng 14,8 giây và inference time khoảng 0,34 giây, trong khi Frozen Encoder cần khoảng 3.289,5 giây training và 182,5 giây inference. Frozen Encoder có rất ít trainable parameters (3.076) nhưng vẫn phải sử dụng toàn bộ DistilBERT khi suy luận, nên số tham số trainable thấp không đồng nghĩa với chi phí inference thấp.

### 3.2. Decision hiện tại

Với hai mô hình đã có kết quả chính thức, **Baseline là mốc tốt hơn về cả classification performance và chi phí tính toán**. Tuy nhiên, đây chưa phải kết luận cuối cùng của câu hỏi nghiên cứu vì Fine-Tuning chưa có kết quả chính thức.

Do đó nhóm **không được kết luận Fine-Tuning tốt hơn hoặc kém hơn** dựa trên Quick Test. Quyết định cuối cùng chỉ được đưa ra sau khi Fine-Tuning chạy toàn bộ AG News với cùng test set và cùng quy trình đánh giá.

### 3.3. Decision criteria cho mô hình cuối

Sau khi có Fine-Tuning chính thức, nhóm sẽ ra quyết định theo ba nhóm tiêu chí:

1. **Performance:** Accuracy, Precision, Recall, Macro F1.
2. **Computational cost:** Training Time, Inference Time, Trainable Parameters, Model Size.
3. **Error behavior:** Confusion Matrix, các class pair thường nhầm và representative wrong predictions.

Mô hình cuối không được chọn chỉ vì có Accuracy cao nhất; nhóm cần giải thích trade-off giữa chất lượng phân loại, chi phí tính toán và loại lỗi mà mô hình tạo ra.

## 4. Decision Matrix

| Tiêu chí | Baseline | Frozen Encoder | Fine-Tuning | Cách quyết định |
|---|---|---|---|---|
| Classification performance | **Đã có** | **Đã có** | Chưa có | So sánh Accuracy + Macro F1 và xem thêm Precision/Recall |
| Training cost | **Thấp** | **Cao** | Chưa có | Ưu tiên mô hình đạt hiệu năng phù hợp với chi phí chấp nhận được |
| Inference cost | **Thấp** | **Cao** | Chưa có | Xem xét khi triển khai |
| Trainable parameters | 200.004 | 3.076 | 66.365.956 | Không dùng riêng tiêu chí này để kết luận chi phí |
| Error behavior | Có | Có | Chưa có | Kiểm tra Business ↔ Sci/Tech và các mẫu sai đại diện |

## 5. Decision sau khi có Fine-Tuning

Nhóm sẽ không chọn mô hình chỉ vì Accuracy cao nhất. Quyết định cuối cùng phải trả lời ba câu hỏi:

1. **Mô hình nào có performance tốt nhất?**
2. **Performance tăng thêm có đáng với training/inference cost không?**
3. **Fine-Tuning có thực sự giảm các pattern lỗi đã phát hiện, đặc biệt Business ↔ Sci/Tech, hay không?**

Nếu Fine-Tuning cải thiện rõ rệt Macro F1 và giảm các lỗi quan trọng với chi phí chấp nhận được, nhóm có cơ sở chọn Fine-Tuning. Nếu mức cải thiện nhỏ trong khi chi phí tăng mạnh, Baseline có thể là lựa chọn thực dụng hơn. Đây là **tiêu chí ra quyết định**, chưa phải kết luận cuối khi Fine-Tuning chưa chạy chính thức.
