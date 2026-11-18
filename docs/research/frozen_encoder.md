# Frozen Encoder

## 1. Ý tưởng

Frozen Encoder là cách dùng Transformer đã pretrain như một **bộ trích đặc trưng cố định**.
Toàn bộ tham số của Transformer bị đóng băng (`requires_grad = False`), chỉ có **Classification Head** nhỏ được huấn luyện trên AG News.

```text
Input Text
    ↓
Tokenizer
    ↓
Pretrained Transformer (FROZEN)
    ↓
Mean Pooling
    ↓
Classification Head (TRAINED)
    ↓
Prediction: World / Sports / Business / Sci-Tech
```

## 2. Cấu hình sử dụng

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| Backbone | `distilbert-base-uncased` | Nhỏ, nhanh (~66M tham số), phù hợp máy sinh viên; project cho phép BERT hoặc DistilBERT |
| Max length | 64 token | Tin tức AG News ngắn (tiêu đề + mô tả), 64 là đủ và giúp chạy nhanh |
| Pooling | Mean pooling (bỏ padding) | Với encoder bị freeze, mean pooling thường cho đặc trưng câu tốt hơn lấy riêng token `[CLS]` |
| Head | `Dropout(0.1)` → `Linear(768, 4)` | Head đơn giản, ít tham số, đúng tinh thần "chỉ train head" |
| Optimizer | AdamW, lr = 1e-3, weight decay = 0.01 | Head nhỏ, train từ đầu nên dùng lr lớn hơn fine-tuning |
| Loss | CrossEntropy | Phân loại 4 lớp |
| Chọn mô hình | Epoch có validation accuracy cao nhất | Tránh overfit head |

## 3. Cách train hiệu quả

Vì encoder bị freeze, đầu ra của nó **không thay đổi giữa các epoch**. Code vì vậy:

1. Chạy encoder **một lần** trên train/val/test để lấy vector 768 chiều cho mỗi bài báo.
2. Train head trên các vector đã lưu trong RAM (rất nhanh, nhiều epoch vẫn chỉ vài giây).

Kết quả giống hệt việc chạy lại encoder mỗi epoch (do encoder ở chế độ `eval`, không có dropout), nhưng tiết kiệm rất nhiều thời gian.

## 4. Các chỉ số được ghi nhận

| Chỉ số | Cách đo |
|---|---|
| Accuracy, Precision, Recall, Macro F1 | Trên tập test AG News (7.600 mẫu); Precision/Recall/F1 lấy trung bình macro |
| Training Time | Thời gian trích đặc trưng train + val **cộng** thời gian train head (có ghi riêng từng phần) |
| Inference Time | Tổng thời gian dự đoán toàn bộ test (encoder + head), kèm ms/mẫu |
| Trainable Parameters | Tham số có `requires_grad = True` (chỉ head: 768×4 + 4 = 3.076) |
| Model Size | Tổng kích thước tham số + buffer của cả encoder và head (MB); ghi thêm kích thước checkpoint head |

Kết quả lưu ở `results/metrics/frozen_encoder_metrics.csv`; dự đoán từng mẫu lưu ở `results/metrics/frozen_encoder_predictions.csv` (cho confusion matrix và error analysis).

## 5. Kỳ vọng và đánh đổi (giả thuyết, cần kiểm chứng bằng thực nghiệm)

- **Ưu điểm:** rất ít tham số cần train, train nhanh, ít tốn bộ nhớ, khó overfit.
- **Nhược điểm:** đặc trưng không được điều chỉnh cho bài toán tin tức nên thường kém Fine-Tuning về độ chính xác.
- Về cách so sánh: Baseline (TF-IDF + LR) có thể cạnh tranh trên AG News vì dữ liệu có nhiều từ khóa đặc trưng theo chủ đề; con số cụ thể sẽ do thực nghiệm quyết định.
- Khi so sánh Training Time cần nói rõ Frozen Encoder có chi phí trích đặc trưng một lần, còn Fine-Tuning phải chạy encoder ở mọi epoch.

## 6. Cách chạy

```bash
# chạy thử nhanh
python src/models.py --train_size 5000 --val_size 1000 --test_size 1000

# chạy đầy đủ (Thứ 5)
python src/models.py
```

Hoặc mở `notebooks/03_frozen_encoder.ipynb`.

## 7. Tham khảo

- Sanh et al., 2019 — *DistilBERT, a distilled version of BERT*.
- Devlin et al., 2019 — *BERT: Pre-training of Deep Bidirectional Transformers*.
- Hugging Face `transformers` và `datasets` (dataset `ag_news`).
