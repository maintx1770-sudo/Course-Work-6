# TF-IDF + Logistic Regression (Baseline)

Người phụ trách: Huynh. Dùng cho Coursework Topic 6, phân loại tin tức AG News.

## 1. Vì sao cần baseline

Baseline là mô hình đơn giản, rẻ, dùng làm mốc so sánh. Nếu Transformer chỉ hơn baseline vài phần trăm mà tốn gấp trăm lần thời gian và dung lượng thì phải cân nhắc có đáng không. Vì vậy báo cáo so sánh cả thời gian và kích thước mô hình, không chỉ accuracy.

## 2. TF-IDF

Biến mỗi văn bản thành một vector số, mỗi chiều ứng với một từ (hoặc cụm từ) trong từ vựng.

- **TF** (Term Frequency, tần suất từ): từ xuất hiện càng nhiều trong văn bản thì điểm càng cao. Cấu hình dùng `sublinear_tf=True`, tức là thay `tf` bằng `1 + ln(tf)` để một từ lặp 20 lần không áp đảo.
- **IDF** (Inverse Document Frequency, nghịch đảo tần suất văn bản): từ xuất hiện trong ít văn bản thì mang nhiều thông tin hơn. Công thức của scikit-learn (mặc định `smooth_idf=True`):

  `idf(t) = ln((1 + n) / (1 + df(t))) + 1`

  với `n` là số văn bản, `df(t)` là số văn bản chứa từ `t`.
- Sau đó vector được chuẩn hóa L2 (chia cho độ dài vector) để văn bản dài không có điểm lớn hơn chỉ vì dài.

### Ví dụ tính tay

Ba văn bản: d1 = "oil price rise", d2 = "team win", d3 = "oil team win". `n = 3`. (Dùng tf thường, không sublinear, cho đơn giản.)

| Từ | df | idf |
|---|---|---|
| oil | 2 | ln(4/3) + 1 = 1.288 |
| price | 1 | ln(4/2) + 1 = 1.693 |
| rise | 1 | 1.693 |
| team | 2 | 1.288 |
| win | 2 | 1.288 |

Vector d1 (oil, price, rise đều có tf = 1): `[1.288, 1.693, 1.693]`.
Độ dài = √(1.288² + 1.693² + 1.693²) = √7.391 = 2.719.
Sau chuẩn hóa L2: oil = 0.474, price = 0.623, rise = 0.623.

Nhận xét: "price" và "rise" chỉ có ở d1 nên nặng hơn "oil" (có ở 2 văn bản).

## 3. Logistic Regression nhiều lớp (multinomial)

Với 4 lớp, mô hình có một vector trọng số `w_k` và một bias `b_k` cho mỗi lớp.

1. Điểm của lớp k: `z_k = w_k · x + b_k`
2. Softmax đổi điểm thành xác suất: `p_k = exp(z_k) / Σ exp(z_j)`
3. Loss (cross-entropy) của một mẫu: `-ln(p_đúng)`

### Ví dụ tính tay (rút gọn còn 2 lớp)

Giả sử `z = [1.0, 2.0]`.
`exp(1) = 2.718`, `exp(2) = 7.389`, tổng = 10.107.
`p = [0.269, 0.731]`.
Nếu lớp đúng là lớp 2: loss = -ln(0.731) = 0.313. Nếu lớp đúng là lớp 1: loss = -ln(0.269) = 1.313, lớn hơn nhiều vì mô hình tin sai.

Huấn luyện là tìm `w, b` để tổng loss nhỏ nhất. Có thêm phạt L2 (regularization, điều chuẩn) để trọng số không quá lớn. Tham số `C` của scikit-learn là **nghịch đảo** độ mạnh của phạt: C nhỏ = phạt mạnh, C lớn = phạt nhẹ. Code chọn C trong {1, 5, 10} theo Macro F1 trên tập validation.

## 4. Cấu hình đã chọn

| Thành phần | Giá trị | Lý do |
|---|---|---|
| ngram_range | (1, 2) | bigram bắt được cụm như "stock market", "white house" |
| max_features | 50,000 | giữ mô hình nhỏ, các từ hiếm ít giá trị |
| min_df | 2 | bỏ từ chỉ xuất hiện 1 lần (thường là nhiễu, lỗi gõ) |
| sublinear_tf | True | giảm ảnh hưởng của từ lặp nhiều |
| Solver | lbfgs (mặc định) | hỗ trợ multinomial, chạy ổn với dữ liệu cỡ này |
| C | chọn trên validation | tránh chọn theo test |

## 5. Cách tính các chỉ số đặc thù của baseline

- **Trainable Parameters**: `coef_.size + intercept_.size` = `4 × số_đặc_trưng + 4`. TF-IDF không có tham số học bằng gradient nên không tính. Với 50,000 đặc trưng: 4 × 50,000 + 4 = 200,004.
- **Model Size**: dung lượng file `.joblib` của cả pipeline (gồm từ vựng TF-IDF và trọng số).
- **Training Time**: thời gian `fit` của pipeline (TF-IDF + Logistic Regression) với giá trị C được chọn, chạy trên CPU.
- **Inference Time**: thời gian `predict` trên toàn bộ test set, đã gồm bước biến văn bản thành vector. Ghi thêm ms trên mỗi mẫu để dễ so với Transformer.

## 6. Dữ liệu AG News (sự thật)

- 4 lớp: World, Sports, Business, Sci/Tech (nhãn 0..3).
- Bộ gốc: 120,000 mẫu train (30,000 mỗi lớp), 7,600 mẫu test (1,900 mỗi lớp), nên test set cân bằng.
- Nhóm chia chung bằng `src/data.py`: train 108,000 / val 12,000 / test 7,600, stratified, seed = 42. Baseline dùng đúng `load_splits()` của file này, văn bản đã qua `clean_text` (giải mã HTML entity, bỏ backslash, gộp khoảng trắng).
- Nguồn: Zhang, Zhao, LeCun (2015), *Character-level Convolutional Networks for Text Classification*, NeurIPS 2015.

## 7. Giả định và giới hạn

- **Suy luận**: TF-IDF coi văn bản là "túi từ" (bag of words), bỏ qua thứ tự từ và ngữ cảnh, nên khó phân biệt các bài mang nghĩa mơ hồ (ví dụ tin công ty công nghệ có thể là Business hoặc Sci/Tech).
- **Khác biệt với Transformer**: baseline dùng toàn bộ văn bản, không cắt độ dài. Transformer bị cắt ở `max_len` (64 hoặc 128 token). Phải ghi rõ điểm này trong báo cáo vì `Design.md` (rủi ro 3) yêu cầu nhất quán cách xử lý văn bản dài.
- **Thời gian chạy** phụ thuộc máy. Baseline chạy trên CPU, Transformer thường chạy trên GPU, nên khi so Training Time và Inference Time phải ghi rõ phần cứng từng mô hình.
- **Kết quả**: chỉ lấy số trong `results/metrics/baseline_metrics.csv` chạy trên máy nhóm, không lấy số từ tài liệu này.

## 8. Tài liệu tham khảo

- scikit-learn, `TfidfVectorizer`: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
- scikit-learn, `LogisticRegression`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- Zhang, X., Zhao, J., LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. NeurIPS 2015.
