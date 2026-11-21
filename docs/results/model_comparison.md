# So sánh Tổng hợp 3 Mô hình Phân loại Văn bản (AG News)

## 1. Bảng So sánh Hiệu năng và Chi phí Tính toán

| Model | Accuracy | Precision | Recall | Macro F1 | Training Time | Inference Time | Trainable Params | Model Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.9204 | 0.9203 | 0.9204 | 0.9203 | 27.16 s | 0.57 s | 200,004 | 3.41 MB |
| Frozen Encoder (DistilBERT) | 0.9034 | 0.9036 | 0.9034 | 0.9032 | 3,289.52 s | 182.53 s | 3,076 | 253.17 MB |
| Fine-Tuning (DistilBERT) | 0.9422 | 0.9424 | 0.9422 | 0.9423 | 2,353.08 s | 16.67 s | 66,365,956 | 253.17 MB |

*(Ghi chú: Đánh giá trên tập kiểm thử AG News gồm 7,600 mẫu chia đều 4 lớp `World`, `Sports`, `Business`, `Sci/Tech`; `Baseline` và `Frozen Encoder` chạy trên CPU, `Fine-Tuning` chạy 3 epochs trên GPU CUDA.)*

---

## 2. Phân tích & Nhận xét So sánh

1. **Chất lượng phân loại (Accuracy & Macro F1):**
   - **Fine-Tuning (DistilBERT)** đạt hiệu năng cao nhất với **Accuracy = 94.22%** và **Macro F1 = 0.9423**, vượt **+2.18%** so với `TF-IDF + Logistic Regression` (92.04%) và **+3.88%** so với `Frozen Encoder` (90.34%)[cite: 6, 7]. Việc cập nhật toàn bộ 66.37M tham số của encoder giúp mô hình điều chỉnh biểu diễn ngữ cảnh phù hợp với cấu trúc tin tức AG News[cite: 6, 7].
   - **TF-IDF + Logistic Regression (Baseline)** đạt **92.04%**, cao hơn **Frozen Encoder** (+1.70%) dù chỉ dùng đặc trưng từ vựng n-gram (1–2)[cite: 6, 7]. Nguyên nhân là tin tức AG News chứa nhiều từ khóa đặc trưng theo từng chủ đề, trong khi `Frozen Encoder` giữ nguyên trọng số pre-trained (chỉ huấn luyện 3,076 tham số ở lớp tuyến tính cuối) nên chưa thích nghi sâu với phân phối nhãn của bài toán[cite: 6, 7].

2. **Đánh đổi giữa Hiệu năng và Tài nguyên tính toán:**
   - **Tốc độ & Kích thước:** `Baseline` vượt trội về hiệu quả triển khai với kích thước chỉ **3.41 MB**, huấn luyện trong **27.16 giây** và suy luận toàn bộ 7,600 mẫu test chỉ mất **0.57 giây** (~0.075 ms/mẫu)[cite: 6, 7].
   - **Chi phí mô hình Transformer:** Cả hai biến thể DistilBERT đều có kích thước mô hình **253.17 MB** (gấp ~74 lần Baseline)[cite: 6, 7]. `Fine-Tuning` cần GPU để huấn luyện trong **2,353.08 giây** (~39.2 phút) nhưng đạt suy luận nhanh (**16.67 giây**, ~2.19 ms/mẫu)[cite: 6, 7], phù hợp cho hệ thống ưu tiên độ chính xác tối đa.

3. **Phân tích lỗi điển hình (Error Analysis):**
   - Cả 3 mô hình đều phân loại rất tốt lớp **Sports** (F1 trên 0.97; riêng Fine-Tuning đạt F1 = 0.9869 với 1,878/1,900 mẫu đúng)[cite: 9, 12].
   - Nhầm lẫn tập trung chủ yếu ở cặp **Business ↔ Sci/Tech** (ở Fine-Tuning chiếm 243/439 tổng số lỗi: 136 mẫu `Business → Sci/Tech` và 107 mẫu `Sci/Tech → Business`)[cite: 12, 13] do các bản tin về hoạt động kinh doanh, nhân sự, cổ phiếu của các tập đoàn công nghệ (IBM, Intel, Google, Yahoo!) mang nội dung giao thoa giữa hai lĩnh vực[cite: 14].

---

## 3. Biểu đồ Trực quan hóa Kết quả
- Biểu đồ so sánh tổng hợp hiệu năng và chi phí: `results/figures/model_comparison.png`
- Ma trận nhầm lẫn từng mô hình:
  - `results/figures/confusion_matrix_baseline.png`[cite: 6, 15]
  - `results/figures/confusion_matrix_frozen_encoder.png`[cite: 6, 8]
  - `results/figures/confusion_matrix_fine_tuning.png`[cite: 6, 9]