# Các chỉ số đánh giá (Evaluation Metrics)

Cả ba phương pháp đều được đánh giá trên cùng một tập chỉ số tiêu chuẩn trên tập dữ liệu kiểm tra AG News (7.600 mẫu)[cite: 20, 21]:

## 1. Nhóm chỉ số hiệu năng phân loại (Classification Performance Metrics)
- **Accuracy (Độ chính xác tổng thể):** Tỷ lệ dự đoán đúng trên tổng số mẫu kiểm tra: $\text{Accuracy} = \frac{\text{Số mẫu dự đoán đúng}}{\text{Tổng số mẫu}}$[cite: 20].
- **Precision (Macro):** Trung bình cộng độ chính xác (Precision) của cả 4 lớp (`World`, `Sports`, `Business`, `Sci/Tech`), đánh giá mức độ tin cậy khi mô hình gán nhãn cho một chủ đề[cite: 20, 21].
- **Recall (Macro):** Trung bình cộng độ phủ (Recall) của cả 4 lớp, đánh giá khả năng nhận diện đầy đủ và không bỏ sót mẫu thuộc từng chủ đề[cite: 20, 21].
- **Macro F1-score:** Trung bình điều hòa giữa Macro Precision và Macro Recall, đảm bảo mọi lớp chủ đề đều có trọng số ngang nhau[cite: 20].
- **Confusion Matrix (Ma trận nhầm lẫn):** Ma trận kích thước $4 \times 4$ thể hiện chi tiết số lượng dự đoán đúng và các cặp chủ đề thường bị nhầm lẫn[cite: 20, 21].

## 2. Nhóm chỉ số chi phí tính toán (Computational Cost Metrics)
- **Training Time (s):** Tổng thời gian huấn luyện mô hình trên tập train (tính bằng giây)[cite: 20, 21].
- **Inference Time (s):** Tổng thời gian suy luận trên toàn bộ 7.600 mẫu của tập test (tính bằng giây)[cite: 20, 21].
- **Trainable Parameters:** Số lượng tham số được cập nhật trọng số trong quá trình huấn luyện[cite: 20, 21].
- **Model Size (MB):** Dung lượng lưu trữ của trọng số mô hình sau khi huấn luyện (tính bằng MB)[cite: 20, 21].

Việc sử dụng chung một quy trình và bộ chỉ số đánh giá giúp đảm bảo tính công bằng khi so sánh giữa ba phương pháp: **TF-IDF + Logistic Regression**, **Frozen Encoder** và **Fine-Tuning**[cite: 20].