# ERROR ANALYSIS — NEWS TOPIC CLASSIFICATION (AG NEWS)

## 1. Tổng quan tỷ lệ lỗi & Các cặp lớp thường bị nhầm lẫn

### 1.1. Thống kê lỗi tổng thể trên tập Test (7,600 mẫu)

Dựa trên kết quả đánh giá chuẩn hóa từ `results/metrics/evaluation_results.csv` và các biểu đồ ma trận nhầm lẫn (`results/figures/confusion_matrix_*.png`), tỷ lệ dự đoán sai của 3 phương pháp trên tập Test (7,600 mẫu; mỗi lớp 1,900 mẫu) được tổng hợp như sau:

| Phương pháp | Accuracy (%) | Macro F1-score (%) | Tổng số mẫu đúng | Tổng số mẫu sai | Tỷ lệ lỗi (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + Logistic Regression)** | 92.04% | 92.03% | 6,995 | 605 | 7.96% |
| **Frozen Encoder (DistilBERT)** | 90.34% | 90.32% | 6,866 | 734 | 9.66% |
| **Fine-Tuning (DistilBERT)** | **94.22%** | **94.23%** | **7,161** | **439** | **5.78%** |

**Nhận xét tổng quan:**
- **Baseline (TF-IDF + Logistic Regression)** đạt độ chính xác **92.04%** (6,995/7,600 mẫu đúng, 605 mẫu sai). Lớp **Sports** đạt hiệu năng cao nhất (chỉ sai 41/1,900 mẫu), trong khi lớp **Business** có tỷ lệ sai cao nhất (sai 216/1,900 mẫu).
- **Frozen Encoder (DistilBERT)** có tỷ lệ lỗi cao hơn Baseline (**9.66%**, tương đương 734 mẫu sai). Nguyên nhân là toàn bộ trọng số tầng Transformer Encoder bị đóng băng, khiến biểu diễn đặc trưng từ mô hình tiền huấn luyện chưa thích nghi tối ưu với miền dữ liệu tin tức chuyên biệt của AG News.
- **Fine-Tuning (DistilBERT)** giảm mạnh tỷ lệ lỗi xuống chỉ còn **5.78%** (**439/7,600 mẫu sai**, đúng 7,161 mẫu) — giảm **166 mẫu lỗi (-27.4%)** so với Baseline và giảm **295 mẫu lỗi (-40.2%)** so với Frozen Encoder. Nhờ cập nhật toàn bộ 66.37M tham số của encoder, mô hình học được ranh giới ngữ nghĩa chuyên biệt cho từng chủ đề tin tức (lớp `Sports` đạt F1-score **98.69%**, chỉ sai 22/1,900 mẫu).

---

### 1.2. Các cặp lớp (Class Pairs) thường bị nhầm lẫn nhất

Trích xuất từ các phần tử ngoài đường chéo chính trên ma trận nhầm lẫn của từng mô hình:

| Cặp nhầm lẫn (`True → Predicted`) | Baseline | Frozen Encoder | Fine-Tuning | Mức độ phổ biến |
| :--- | :---: | :---: | :---: | :---: |
| **Sci/Tech → Business** | **117** | **197** | **107** | Cao nhất |
| **Business → Sci/Tech** | **143** | **155** | **136** | Cao nhất |
| **World → Business** | **71** | **98** | **46** | Cao thứ hai |
| **Business → World** | **57** | **69** | **31** | Cao thứ hai |
| **World → Sports** | **56** | **60** | **12** | Trung bình |
| **Sci/Tech → World** | **41** | **62** | **25** | Trung bình |
| **World → Sci/Tech** | **48** | **44** | **44** | Trung bình |
| **Sports** (nhầm sang các lớp khác) | **41** *(21+12+8)* | **27** *(14+8+5)* | **22** *(14+6+2)* | Thấp nhất |

**Phân tích xu hướng nhầm lẫn:**
- **Nhóm nhầm lẫn lớn nhất (`Business ↔ Sci/Tech`):** Chiếm tới **42.98%** tổng số lỗi của Baseline (`260/605` mẫu), **47.96%** tổng số lỗi của Frozen Encoder (`352/734` mẫu) và **55.35%** tổng số lỗi của Fine-Tuning (`243/439` mẫu). Đặc biệt ở Frozen Encoder, số mẫu `Sci/Tech` bị đoán nhầm thành `Business` tăng mạnh lên **197 mẫu** (so với 117 mẫu ở Baseline và 107 mẫu ở Fine-Tuning), kéo Recall của lớp `Sci/Tech` xuống còn **85.74%**.
- **Nhóm nhầm lẫn thứ hai (`World ↔ Business`):** Chiếm **21.16%** tổng số lỗi của Baseline (`128/605` mẫu) và **22.75%** tổng số lỗi của Frozen Encoder (`167/734` mẫu), nhưng đã giảm hơn một nửa ở Fine-Tuning xuống còn **77 mẫu** (`46 + 31`, chiếm **17.54%** tổng lỗi).
- **Lớp ít nhầm lẫn nhất (`Sports`):** Cả 3 mô hình đều nhận diện rất tốt lớp này, trong đó Fine-Tuning giảm số lỗi xuống mức thấp nhất là **22/1,900 mẫu** (`14` sang `World`, `6` sang `Business`, `2` sang `Sci/Tech`) và giảm mạnh tình trạng nhầm `World → Sports` từ **56–60 mẫu** xuống chỉ còn **12 mẫu**.

---

## 2. Phân tích chi tiết các mẫu dự đoán sai tiêu biểu

Các mẫu thực tế trích xuất từ tập Test (`results/metrics/*_predictions.csv`) minh họa 4 tình huống điển hình:

| STT | Đoạn trích văn bản tin tức (Test Set) | Nhãn thật | Baseline | Frozen Encoder | Fine-Tuning | Phân loại tình huống |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | *"After Wait, Google Set for Market Debut NEW YORK (Reuters) - Shares of Google Inc. will make their Nasdaq stock market debut on Thursday..."* *(Index 239)* | `Business` | `Sci/Tech` ❌ | `Sci/Tech` ❌ | `Business` ✅ | Fine-Tuning khắc phục lỗi từ khóa |
| **2** | *"Oil Eases as Iraq Resumes Exports By Andrew Mitchell LONDON (Reuters) - High-flying oil prices eased for a second session on Monday as Iraq resumed exports..."* *(Index 448)* | `Business` | `World` ❌ | `World` ❌ | `World` ❌ | Lỗi chung cả 3 mô hình (Mơ hồ nhãn) |
| **3** | *"Venezuela Prepares for Chavez Recall Vote Supporters and rivals warn of possible fraud; government says Chavez's defeat could produce turmoil in world oil market."* *(Index 36)* | `World` | `Business` ❌ | `Business` ❌ | `Business` ❌ | Nhiễu từ khóa kinh tế trong tin chính trị |
| **4** | *"Mars Rovers Relay Images Through Mars Express European Space Agency -- ESAs Mars Express has relayed pictures from one of NASA's Mars rovers..."* *(Index 17)* | `Sci/Tech` | `Sci/Tech` ✅ | `World` ❌ | `Sci/Tech` ✅ | Frozen Encoder mất tín hiệu đặc trưng |

### Giải thích cơ chế lỗi qua các trường hợp:
1. **Lỗi thiên kiến từ khóa bề mặt (`Business ↔ Sci/Tech`):**
  - Các bản tin về tập đoàn công nghệ lớn (như Google, IBM, Intel, Apple, Oracle...) thường chứa dày đặc tên sản phẩm và thuật ngữ kỹ thuật. **Baseline** dựa trên trọng số TF-IDF của từng từ đơn lẻ nên dễ gán nhãn `Sci/Tech`, dù trọng tâm câu nói về niêm yết sàn Nasdaq, doanh thu hoặc sáp nhập doanh nghiệp (`Business`). **Fine-Tuning** khắc phục tốt trường hợp này nhờ hiểu ngữ cảnh tài chính của toàn câu.
2. **Lỗi giao thoa ngữ cảnh vĩ mô (`World ↔ Business`):**
  - Các bản tin về giá dầu thô gắn với tình hình địa chính trị (như xuất khẩu dầu tại Iraq hay trưng cầu dân ý tại Venezuela ảnh hưởng thị trường dầu mỏ) mang đặc trưng của cả hai lớp. Khi cụm từ *"world oil market"* hoặc tên quốc gia xuất hiện nổi bật, mô hình dễ bị kéo lệch giữa sự kiện chính trị quốc tế (`World`) và biến động hàng hóa (`Business`).
3. **Suy giảm tín hiệu từ khóa chuyên ngành ở Frozen Encoder (`Sci/Tech → World/Business`):**
  - Ở một số mẫu có từ khóa công nghệ/vũ trụ đặc thù (như NASA, Mars Express, ESA), **Baseline** bắt chính xác nhờ trọng số từ vựng cao, trong khi **Frozen Encoder** chỉ dùng vector biểu diễn tổng quát chưa qua fine-tune nên lớp Classification Head tuyến tính ở cuối không đủ năng lực tách biệt ranh giới quyết định (khiến 197 mẫu `Sci/Tech` bị đẩy sang `Business` và 62 mẫu sang `World`).

---

## 3. Nguyên nhân gây nhầm lẫn & Đặc điểm các mẫu khó phân loại

1. **Sự chồng chéo chủ đề tự nhiên (Topic Overlap):**
  - Bốn chủ đề của AG News không hoàn toàn độc lập về mặt ngữ nghĩa. Ranh giới giữa **kinh doanh công nghệ (`Business` vs `Sci/Tech`)** và **kinh tế – chính trị quốc tế (`World` vs `Business`)** là nguồn phát sinh lỗi lớn nhất của mọi mô hình.
2. **Thiên kiến từ khóa (Keyword Bias / Lexical Overlap):**
  - Các từ mang tính ẩn dụ hoặc đa nghĩa xuất hiện trong ngữ cảnh khác dễ gây nhiễu (ví dụ: các từ như *"race"*, *"champion"*, *"victory"* xuất hiện trong bài báo cạnh tranh thị phần kinh doanh hoặc bầu cử dễ kéo dự đoán lệch sang `Sports` — thể hiện qua 56 mẫu `World → Sports` ở Baseline và 60 mẫu ở Frozen Encoder, trước khi được Fine-Tuning giảm xuống còn 12 mẫu).
3. **Bản tin ngắn, giới hạn độ dài token (`max_len = 64`) & từ viết tắt:**
  - Nhiều mẫu trong AG News chỉ gồm tiêu đề ngắn gọn hoặc bị cắt ở 64 tokens đầu tiên, chứa tên viết tắt của tổ chức, mã chứng khoán hoặc tên riêng địa phương ít xuất hiện trong tập huấn luyện, khiến mô hình thiếu thông tin ngữ cảnh phía sau để suy luận.
4. **Tính tương đối trong nhãn gốc (Ambiguous Ground-Truth):**
  - Dữ liệu AG News được tổng hợp từ các chuyên mục của nhiều hãng tin khác nhau; cùng một sự kiện tài chính của công ty công nghệ (như tuyển dụng tại IBM ở Index 20 hay IPO của Google ở Index 23) có thể được nguồn này xếp vào `Sci/Tech` nhưng nguồn khác lại xếp vào `Business`. Đây là ngưỡng giới hạn tự nhiên (Bayes error) của tập dữ liệu.

---

## 4. Hạn chế của từng phương pháp rút ra từ Error Analysis

| Phương pháp | Hạn chế chính | Biểu hiện qua kết quả thực nghiệm |
| :--- | :--- | :--- |
| **1. TF-IDF + Logistic Regression (Baseline)** | • Biểu diễn túi từ (Bag-of-Words), bỏ qua trật tự từ và cấu trúc cú pháp.<br>• Không phân biệt được từ đa nghĩa theo ngữ cảnh. | Đạt **92.04% Accuracy (605 lỗi)** nhờ bắt từ khóa tốt, nhưng dễ nhầm ở các câu chứa từ khóa công nghệ trong bối cảnh tài chính (**143** mẫu `Business → Sci/Tech`) hoặc ẩn dụ thể thao (**56** mẫu `World → Sports`). |
| **2. Frozen Encoder (DistilBERT)** | • Toàn bộ 66.36 triệu tham số của DistilBERT bị đóng băng, chỉ huấn luyện 3,076 tham số ở Classification Head.<br>• Không gian đặc trưng chưa được điều chỉnh cho bài toán phân loại tin tức. | Có số lỗi cao nhất (**734 lỗi**, Accuracy **90.34%**), đặc biệt suy giảm mạnh khả năng tách biệt `Sci/Tech` và `Business` (**197** mẫu `Sci/Tech → Business`, **155** mẫu `Business → Sci/Tech`). |
| **3. Fine-Tuning (DistilBERT)** | • Chi phí tính toán và kích thước lưu trữ mô hình lớn (~253.17 MB, cần GPU để huấn luyện trong ~39.2 phút).<br>• Vẫn chịu giới hạn ở các mẫu có ranh giới nhãn gốc mơ hồ hoặc vượt quá `max_len = 64`. | Đạt hiệu năng vượt trội nhất (**94.22% Accuracy**, chỉ còn **439 lỗi** — giảm **27.4%** lỗi so với Baseline và **40.2%** lỗi so với Frozen Encoder), nhưng **55.35%** số lỗi còn lại (`243/439` mẫu) vẫn tập trung ở cặp giao thoa `Business ↔ Sci/Tech`. |