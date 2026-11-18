# Nghiên cứu Bộ dữ liệu AG News (AG's News Topic Classification Dataset)

## 1. Giới thiệu & Nguồn gốc dữ liệu
* **Tên bộ dữ liệu:** AG News (`fancyzhx/ag_news` trên Hugging Face Hub).
* **Nguồn gốc:** Được tổng hợp từ kho tin tức *AG's corpus of news articles* (hơn 1 triệu bài báo từ hơn 2.000 nguồn tin tức trực tuyến do Antonio Gulli thu thập) và được chuẩn hóa làm bộ benchmark phân loại văn bản kinh điển bởi Xiang Zhang, Junbo Zhao và Yann LeCun (*Character-level Convolutional Networks for Text Classification*, NeurIPS 2015).
* **Cấu trúc bản ghi:** Mỗi mẫu dữ liệu bao gồm phần tiêu đề (`title`) và đoạn tóm tắt mở đầu (`description`) của bài báo được gộp thành một chuỗi văn bản đầu vào (`text`), đi kèm một nhãn chủ đề nguyên (`label` từ `0` đến `3`).
* **Đặc điểm bài toán:** Phân loại văn bản đa lớp (Multi-class Text Classification) gồm **4 chủ đề cân bằng tuyệt đối**.

| Label ID | Tên chủ đề (`label_name`) | Nội dung đặc trưng |
| :---: | :--- | :--- |
| `0` | **World** | Tin tức quốc tế, chính trị, ngoại giao, xung đột và các sự kiện toàn cầu. |
| `1` | **Sports** | Tin tức thể thao, giải đấu, vận động viên và kết quả thi đấu. |
| `2` | **Business** | Tin tức kinh tế, tài chính, thị trường chứng khoán, doanh nghiệp và thương mại. |
| `3` | **Sci/Tech** | Tin tức khoa học, công nghệ thông tin, phần mềm, không gian và phát minh kỹ thuật. |

---

## 2. Chiến lược chia tập (Data Splitting) & Phân bố nhãn
Bộ dữ liệu gốc gồm **120.000** mẫu huấn luyện và **7.600** mẫu kiểm thử. Để đánh giá khách quan trong quá trình huấn luyện và tinh chỉnh siêu tham số (hyperparameter tuning) mà không làm rò rỉ dữ liệu kiểm thử (data leakage), nhóm thực hiện tách **10%** tập huấn luyện gốc làm tập Validation bằng phương pháp lấy mẫu phân tầng (`stratified split`, `random_state=42`):

* **Tập Train (`data/processed/train.csv`):** 108.000 mẫu (chiếm 84.64% tổng dữ liệu).
* **Tập Validation (`data/processed/val.csv`):** 12.000 mẫu (chiếm 9.40% tổng dữ liệu).
* **Tập Test (`data/processed/test.csv`):** 7.600 mẫu (chiếm 5.96% tổng dữ liệu).

| Chủ đề | Train (n) | Validation (n) | Test (n) | Tỷ lệ mỗi lớp |
| :--- | :---: | :---: | :---: | :---: |
| **World (0)** | 27,000 | 3,000 | 1,900 | 25.00% |
| **Sports (1)** | 27,000 | 3,000 | 1,900 | 25.00% |
| **Business (2)** | 27,000 | 3,000 | 1,900 | 25.00% |
| **Sci/Tech (3)** | 27,000 | 3,000 | 1,900 | 25.00% |
| **Tổng cộng** | **108,000** | **12,000** | **7,600** | **100.00%** |

**Nhận xét:** Phân bố 4 lớp cân bằng tuyệt đối (25.00% mỗi lớp trên cả 3 tập). Do đó, chỉ số **Accuracy** và **Macro F1-score** đều phản ánh trung thực hiệu năng phân loại của các mô hình mà không bị thiên lệch lớp (class imbalance).

---

## 3. Phân tích chất lượng dữ liệu & Độ dài văn bản (EDA)

### 3.1. Tính toàn vẹn dữ liệu
Kết quả kiểm tra trên toàn bộ các tập dữ liệu sau xử lý (`notebooks/01_data_exploration.ipynb`):
* **Giá trị thiếu (`Null`) & Chuỗi rỗng (`Empty`):** `0` mẫu trên cả 3 tập Train, Validation và Test.
* **Bản ghi trùng lặp văn bản (`Exact duplicates`):**
  * Tập Train: `62` mẫu trùng lặp (~0.057%).
  * Tập Validation: `3` mẫu trùng lặp (~0.025%).
  * Tập Test: `0` mẫu trùng lặp.
  * Tỷ lệ trùng lặp cực kỳ nhỏ (< 0.06%), không gây ảnh hưởng đến quá trình học và đánh giá mô hình.

### 3.2. Phân tích độ dài văn bản & Cơ sở chọn `max_len`
Thống kê độ dài trên **108.000** mẫu của tập Train theo **số lượng từ** (tách theo khoảng trắng) và **số lượng subword token** của `distilbert-base-uncased` (đã bao gồm 2 token đặc biệt `[CLS]` và `[SEP]`):

| Chỉ số thống kê | Số từ (Word count) | Số token DistilBERT (`input_ids`) |
| :--- | :---: | :---: |
| **Trung bình (Mean)** | 37.98 | 51.77 |
| **Độ lệch chuẩn (Std)** | 10.20 | 17.62 |
| **Nhỏ nhất (Min)** | 8.00 | 14.00 |
| **Trung vị (50% / Median)** | 37.00 | 49.00 |
| **Phân vị 90% (90th)** | 48.00 | 67.00 |
| **Phân vị 95% (95th)** | 53.00 | 78.00 |
| **Phân vị 99% (99th)** | 70.00 | 116.00 |
| **Lớn nhất (Max)** | 171.00 | 327.00 |

**Tỷ lệ bao phủ dữ liệu (không bị cắt cụt – truncation) theo ngưỡng `max_len`:**
* `max_len = 64`: Bao phủ **87.41%** số mẫu (12.59% bài báo dài hơn sẽ bị cắt bớt phần đuôi).
* `max_len = 128`: Bao phủ **99.42%** số mẫu (hầu như toàn bộ thông tin tiêu đề và tóm tắt được giữ trọn vẹn).
* `max_len = 256`: Bao phủ **99.97%** số mẫu, nhưng chi phí tính toán Self-Attention tăng gấp 4 lần so với `128`.

**Quyết định thiết kế:** Nhóm lựa chọn **`max_len = 128`** cho các mô hình Transformer (`distilbert-base-uncased`), đạt sự cân bằng tối ưu giữa việc bảo toàn **99.42%** nội dung văn bản và tốc độ huấn luyện/suy luận.

---

## 4. Đặc điểm nhiễu văn bản & Quy trình tiền xử lý (`src/data.py`)
Qua khảo sát thực tế các mẫu trong AG News, dữ liệu gốc chứa một số dạng nhiễu đặc thù do quá trình thu thập (web scraping) từ các nguồn tin RSS/HTML:
1. **Ký tự HTML Entities bị lỗi hoặc chưa giải mã:** Xuất hiện các chuỗi như `&amp;`, `&quot;`, `&#39;`, hoặc các chuỗi bị khuyết dấu `&` ở đầu như `#39;` (ví dụ: `It #39;s`) và `quot;` (ví dụ: `quot;Reuters`).
2. **Ký tự thoát (Escape characters):** Xuất hiện dấu gạch chéo ngược thừa như `\$` thay cho `$` hoặc `\\`.
3. **Khoảng trắng không đồng nhất:** Nhiều dấu cách liên tiếp hoặc ký tự xuống dòng xen kẽ giữa phần tiêu đề và phần mô tả.
4. **Từ khóa tên hãng thông tấn lặp lại:** Các từ như `Reuters`, `AP`, `AFP`, `ESPN.com` xuất hiện thường xuyên ở đầu đoạn mô tả.

**Chiến lược tiền xử lý của nhóm (`clean_text` trong `src/data.py`):**
* Chuẩn hóa các HTML entity bị khuyết ký tự `&` (`#39;` $\rightarrow$ `&#39;`) và giải mã về ký tự gốc bằng `html.unescape()`.
* Loại bỏ dấu gạch chéo ngược thừa (`\`) và chuẩn hóa mọi chuỗi khoảng trắng liên tiếp thành một dấu cách duy nhất.
* **Không** xóa stop-words, **không** lemmatize và giữ nguyên định dạng chữ hoa/thường để bảo toàn cấu trúc ngữ pháp tự nhiên cho cơ chế Self-Attention của Transformer (bộ tokenizer `distilbert-base-uncased` sẽ tự động chuyển về chữ thường trong bước tokenize).
* Toàn bộ dữ liệu sau tiền xử lý được lưu cố định tại `data/processed/{train,val,test}.csv` để đảm bảo cả 3 mô hình (**TF-IDF + Logistic Regression**, **Frozen DistilBERT**, **Fine-tuned DistilBERT**) được huấn luyện và đánh giá trên cùng một phân chia dữ liệu.