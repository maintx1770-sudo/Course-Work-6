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
