# Fine-Tuning DistilBERT cho phân loại tin tức

Dùng cho Coursework Topic 6 (AG News, 4 lớp). Tài liệu này giải thích Fine-Tuning, cách nó khác Frozen Encoder, và các lựa chọn trong `src/train.py`.

## 1. Fine-Tuning là gì

Bộ mã hóa (encoder) DistilBERT đã được huấn luyện trước (pretrained) trên lượng văn bản rất lớn, nên đã "biết" ngôn ngữ. Có hai cách dùng nó cho bài phân loại

| | Frozen Encoder | Fine-Tuning |
|---|---|---|
| Encoder | Đóng băng, không đổi | Được cập nhật |
| Phần được train | Chỉ Classification Head (3,076 tham số) | Encoder + Head (66,365,956 tham số) |
| Ý nghĩa | Dùng DistilBERT như máy trích đặc trưng cố định | Điều chỉnh DistilBERT cho đúng bài toán tin tức |
| Chi phí | Thấp: đặc trưng trích 1 lần | Cao: mỗi bước phải lan truyền ngược qua cả encoder |

Cả hai dùng cùng kiến trúc: `Tokenizer → DistilBERT → Mean Pooling → Dropout → Linear(768, 4)`. Khác biệt duy nhất là có cho encoder nhận gradient hay không. Nhờ vậy so sánh giữa chúng là so sánh công bằng.

## 2. Các thành phần

### Mean Pooling (tính tay)

Encoder trả ra một vector cho mỗi token. Cần gộp thành một vector cho cả câu, bằng cách lấy trung bình các token thật và **bỏ qua token padding** nhờ `attention_mask`.

Ví dụ: câu có 3 token thật và 1 token padding, vector 2 chiều.

| Token | Vector | mask |
|---|---|---|
| t1 | [1, 2] | 1 |
| t2 | [3, 0] | 1 |
| t3 | [2, 4] | 1 |
| pad | [9, 9] | 0 |

- Tổng có mask: [1+3+2, 2+0+4] = [6, 6]. Chia cho số token thật (3): **[2, 2]**.
- Nếu quên mask và chia cho 4: [(1+3+2+9)/4, (2+0+4+9)/4] = [3.75, 3.75], bị lệch vì token padding.

### Loss và gradient (tính tay)

Head cho 4 logit theo thứ tự World, Sports, Business, Sci/Tech. Giả sử `z = [2.0, 1.0, 0.1, 0.1]`.

- `exp(z) = [7.389, 2.718, 1.105, 1.105]`, tổng = 12.317.
- Softmax: `p = [0.600, 0.221, 0.090, 0.090]`.

Cross-Entropy Loss = `-ln(p của lớp đúng)`:

- Nếu nhãn đúng là World: loss = -ln(0.600) = **0.511**.
- Nếu nhãn đúng là Business: loss = -ln(0.090) = **2.41**. Mô hình tin sai nên loss lớn hơn nhiều.

Gradient của loss theo logit là `p - y` (với `y` là one-hot):

- Nhãn World: `[0.600-1, 0.221, 0.090, 0.090]` = `[-0.400, 0.221, 0.090, 0.090]`. Logit của lớp đúng được đẩy lên, các lớp còn lại bị đẩy xuống.

Trong Frozen Encoder, gradient này dừng ở Head. Trong Fine-Tuning, nó chạy ngược qua mean pooling và toàn bộ các tầng của DistilBERT, nên toàn bộ trọng số đều thay đổi.

## 3. Cấu hình trong `src/train.py`

Các giá trị này là **lựa chọn của nhóm, chưa được tune**. `Design.md` mục 6.2 chưa quy định hyperparameter Fine-Tuning.

| Tham số | Giá trị | Lý do |
|---|---|---|
| Backbone | `distilbert-base-uncased` | Giống Frozen Encoder (yêu cầu so sánh công bằng) |
| `max_len` | 64 | Giống Frozen Encoder. Chú ý `data.py` mặc định là 128, nên phải truyền 64 |
| Optimizer | AdamW | Theo `Design.md` |
| Learning rate | 2e-5 | Nhỏ hơn nhiều so với Head của Frozen (1e-3), vì encoder đã pretrained, lr lớn dễ phá trọng số |
| Batch size | 32 | Nằm trong khoảng bài BERT khuyến nghị (16 hoặc 32) |
| Epoch | 3 | Nằm trong khoảng bài BERT khuyến nghị (2 đến 4) |
| Weight decay | 0.01 (không áp cho bias và LayerNorm) | Quy ước phổ biến khi dùng AdamW |
| Warmup | 10% số bước đầu, sau đó giảm tuyến tính về 0 | Tránh cập nhật quá mạnh khi mới bắt đầu |
| Gradient clipping | 1.0 | Giữ gradient ổn định |
| Chọn mô hình | Epoch có validation accuracy cao nhất | Giống cách Frozen Encoder chọn |
| Seed | 42 | Giống các mô hình khác |

Ví dụ số bước: 108,000 mẫu / batch 32 = 3,375 bước mỗi epoch. 3 epoch = 10,125 bước. Warmup 10% = 1,012 bước. Ở bước 506 (nửa warmup), lr = 1e-5, tức nửa giá trị đỉnh 2e-5.

## 4. Cách đo các chỉ số chi phí

Các định nghĩa dưới đây khớp với `models.py` của Frozen Encoder để `compare.py` so sánh cùng một thước đo.

- **Training Time**: tổng thời gian cả vòng huấn luyện, gồm cả lượt validation sau mỗi epoch.
- **Inference Time**: thời gian chạy toàn bộ test set (7,600 mẫu), tính cả tokenization, batch 64. Ghi thêm ms/mẫu.
- **Trainable Parameters**: số tham số có `requires_grad = True`. Fine-Tuning: 66,365,956. Con số này lấy từ kết quả `total_params` của Frozen Encoder (cùng kiến trúc). Vì không có tham số nào bị đóng băng, trainable bằng total.
- **Model Size**: `model_size_mb`, tính từ bộ nhớ của tham số và buffer (fp32), khoảng 253 MB. File checkpoint (`checkpoint_size_mb`) có cùng cỡ vì lưu toàn bộ mô hình. Với Frozen Encoder, file checkpoint chỉ 13 KB vì chỉ lưu Head, nên khi so sánh phải dùng `model_size_mb`, không dùng `checkpoint_size_mb`.

## 5. Chi phí tính toán (ước lượng)

- **Sự thật (từ notebook 03 của My):** Frozen Encoder chạy trên CPU, mất 3,257 s chỉ để trích đặc trưng 120,000 mẫu (chỉ lan truyền xuôi).
- **Suy luận:** Fine-Tuning phải lan truyền xuôi và ngược mỗi bước, thường tốn khoảng gấp 3 lần một lượt xuôi (quy tắc kinh nghiệm, chưa đo). Nếu đúng, mỗi epoch trên CPU mất vài giờ, và 3 epoch có thể hơn 7 giờ. Cần chạy trên GPU (Colab, Kaggle).
- **Hệ quả cho báo cáo:** nếu Frozen chạy CPU còn Fine-Tuning chạy GPU thì Training Time và Inference Time không so sánh trực tiếp được. Trong `fine_tuning_metrics.csv` có cột `device` để ghi lại, và báo cáo phải nêu rõ phần cứng của từng mô hình.

## 6. Giới hạn và rủi ro

- **Overfitting:** train nhiều epoch có thể làm validation giảm, nên lưu epoch tốt nhất theo validation, không lấy epoch cuối.
- **Cắt văn bản:** `max_len = 64` cắt các bài dài, thông tin ở cuối bài bị mất (dùng chung cho cả hai mô hình Transformer, nhưng baseline TF-IDF thì không bị cắt).
- **Chưa tune:** kết quả phụ thuộc cấu hình ở mục 3. Báo cáo không nên khẳng định "Fine-Tuning tốt nhất có thể", chỉ nêu "với cấu hình này".
- **Kết quả:** chỉ lấy số trong `results/metrics/fine_tuning_metrics.csv` chạy thật, không điền số từ tài liệu này.

## 7. Tài liệu tham khảo

- Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019. arXiv:1810.04805. (Khoảng khuyến nghị batch size, learning rate, số epoch khi fine-tune nằm ở phụ lục.)
- Sanh, V., Debut, L., Chaumond, J., Wolf, T. (2019). *DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*. arXiv:1910.01108.
- Loshchilov, I., Hutter, F. (2019). *Decoupled Weight Decay Regularization* (AdamW). ICLR 2019. arXiv:1711.05101.
- Hugging Face Transformers, `AutoModel` và `get_linear_schedule_with_warmup`: https://huggingface.co/docs/transformers
