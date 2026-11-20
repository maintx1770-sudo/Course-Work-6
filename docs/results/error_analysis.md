# ERROR ANALYSIS — NEWS TOPIC CLASSIFICATION (AG NEWS)

## 1. Tổng quan tỷ lệ lỗi & Các cặp lớp thường bị nhầm lẫn

### 1.1. Thống kê lỗi tổng thể trên tập Test (7,600 mẫu)

Dựa trên kết quả đánh giá chuẩn hóa từ `results/metrics/evaluation_results.csv` và các biểu đồ ma trận nhầm lẫn (`results/figures/confusion_matrix_*.png`), tỷ lệ dự đoán sai của 3 phương pháp trên tập Test (7,600 mẫu; mỗi lớp 1,900 mẫu) được tổng hợp như sau:

| Phương pháp | Accuracy (%) | Macro F1-score (%) | Tổng số mẫu đúng | Tổng số mẫu sai | Tỷ lệ lỗi (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + Logistic Regression)** | 91.80% | 91.78% | 6,977 | 623 | 8.20% |
| **Frozen Encoder (DistilBERT)** | 90.34% | 90.32% | 6,866 | 734 | 9.66% |
| **Fine-Tuning (DistilBERT)** | *[Điền %]* | *[Điền %]* | *[Điền số]* | *[Điền số]* | *[Điền %]* |

**Nhận xét tổng quan:**
* **Baseline (TF-IDF + Logistic Regression)** đạt độ chính xác **91.80%** (6,977/7,600 mẫu đúng, 623 mẫu sai). Lớp **Sports** đạt hiệu năng cao nhất (F1-score **96.53%**, chỉ sai 38/1,900 mẫu), trong khi lớp **Business** có tỷ lệ sai cao nhất (F1-score **88.64%**, sai 226/1,900 mẫu).
* **Frozen Encoder (DistilBERT)** có tỷ lệ lỗi cao hơn Baseline (**9.66%**, tương đương 734 mẫu sai). Nguyên nhân là toàn bộ trọng số tầng Transformer Encoder bị đóng băng, khiến biểu diễn đặc trưng từ mô hình tiền huấn luyện chưa thích nghi tối ưu với miền dữ liệu tin tức chuyên biệt của AG News.
* **Fine-Tuning (DistilBERT)** *(cập nhật nhận xét giảm lỗi sau khi điền số liệu full 7,600 mẫu của Fine-Tuning)*.

---

### 1.2. Các cặp lớp (Class Pairs) thường bị nhầm lẫn nhất

Trích xuất từ các phần tử ngoài đường chéo chính trên ma trận nhầm lẫn của từng mô hình:

| Cặp nhầm lẫn (`True → Predicted`) | Baseline | Frozen Encoder | Fine-Tuning | Mức độ phổ biến |
| :--- | :---: | :---: | :---: | :---: |
| **Sci/Tech → Business** | **118** | **197** | *[Điền số]* | Cao nhất |
| **Business → Sci/Tech** | **151** | **155** | *[Điền số]* | Cao nhất |
| **World → Business** | **75** | **98** | *[Điền số]* | Cao thứ hai |
| **Business → World** | **56** | **69** | *[Điền số]* | Cao thứ hai |
| **World → Sports** | **61** | **60** | *[Điền số]* | Trung bình |
| **Sci/Tech → World** | **46** | **62** | *[Điền số]* | Trung bình |
| **World → Sci/Tech** | **43** | **44** | *[Điền số]* | Trung bình |
| **Sports** (nhầm sang các lớp khác) | **38** *(18+10+10)* | **27** *(14+8+5)* | *[Điền số]* | Thấp nhất |

**Phân tích xu hướng nhầm lẫn:**
* **Nhóm nhầm lẫn lớn nhất (`Business ↔ Sci/Tech`):** Chiếm tới **43.18%** tổng số lỗi của Baseline (`269/623` mẫu) và **47.96%** tổng số lỗi của Frozen Encoder (`352/734` mẫu). Đặc biệt ở Frozen Encoder, số mẫu `Sci/Tech` bị đoán nhầm thành `Business` tăng mạnh lên **197 mẫu** (so với 118 mẫu ở Baseline), kéo Recall của lớp `Sci/Tech` xuống còn **85.74%**.
* **Nhóm nhầm lẫn thứ hai (`World ↔ Business`):** Chiếm **21.03%** tổng số lỗi của Baseline (`131/623` mẫu) và **22.75%** tổng số lỗi của Frozen Encoder (`167/734` mẫu).
* **Lớp ít nhầm lẫn nhất (`Sports`):** Cả Baseline và Frozen Encoder đều nhận diện rất tốt lớp này (chỉ nhầm lần lượt **38** và **27** mẫu trên tổng số 1,900 mẫu), thậm chí Frozen Encoder còn giảm được số lỗi trên lớp `Sports` tốt hơn Baseline nhờ khả năng nhận diện thực thể thể thao theo ngữ cảnh.

---

## 2. Phân tích chi tiết các mẫu dự đoán sai tiêu biểu

*(Phối hợp với Trúc Chi trích xuất 4–5 câu thực tế từ tập Test của dự án điền vào bảng dưới đây)*

| STT | Đoạn trích văn bản tin tức (Test Set) | Nhãn thật | Baseline | Frozen Encoder | Fine-Tuning | Phân loại tình huống |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **1** | *[Điền câu trích từ tập Test]* | `Business` | `Sci/Tech` ❌ | `Sci/Tech` ❌ | `Business` ✅ | Fine-Tuning khắc phục lỗi từ khóa |
| **2** | *[Điền câu trích từ tập Test]* | `Business` | `World` ❌ | `World` ❌ | `World` ❌ | Lỗi chung cả 3 mô hình (Mơ hồ nhãn) |
| **3** | *[Điền câu trích từ tập Test]* | `World` | `Business` ❌ | `Business` ❌ | `World` ✅ | Fine-Tuning hiểu ngữ cảnh vĩ mô |
| **4** | *[Điền câu trích từ tập Test]* | `Sci/Tech` | `Sci/Tech` ✅ | `World` ❌ | `Sci/Tech` ✅ | Frozen Encoder mất tín hiệu đặc trưng |

### Giải thích cơ chế lỗi qua các trường hợp:
1. **Lỗi thiên kiến từ khóa bề mặt (`Business ↔ Sci/Tech`):**
   * Các bản tin về tập đoàn công nghệ lớn (như Apple, Microsoft, Google, IBM...) thường chứa dày đặc tên sản phẩm và thuật ngữ kỹ thuật. **Baseline** dựa trên trọng số TF-IDF của từng từ đơn lẻ nên dễ gán nhãn `Sci/Tech`, dù trọng tâm câu nói về doanh thu, giá cổ phiếu hoặc sáp nhập doanh nghiệp (`Business`).
2. **Lỗi giao thoa ngữ cảnh vĩ mô (`World ↔ Business`):**
   * Các bản tin về chính sách thương mại quốc tế, giá dầu thô, lệnh trừng phạt kinh tế hoặc biến động tỷ giá mang đặc trưng của cả hai lớp. Khi trọng số Transformer chưa được tinh chỉnh (**Frozen Encoder**), mô hình dễ nhầm lẫn giữa sự kiện chính trị quốc tế (`World`) và tác động tài chính (`Business`).
3. **Suy giảm tín hiệu từ khóa chuyên ngành ở Frozen Encoder (`Sci/Tech → World/Business`):**
   * Ở một số mẫu có từ khóa công nghệ/khoa học đặc thù, **Baseline** bắt chính xác nhờ trọng số từ vựng cao, trong khi **Frozen Encoder** chỉ dùng vector biểu diễn tổng quát chưa qua fine-tune nên lớp Classification Head tuyến tính ở cuối không đủ năng lực tách biệt ranh giới quyết định (khiến 197 mẫu `Sci/Tech` bị đẩy sang `Business`).

---

## 3. Nguyên nhân gây nhầm lẫn & Đặc điểm các mẫu khó phân loại

1. **Sự chồng chéo chủ đề tự nhiên (Topic Overlap):**
   * Bốn chủ đề của AG News không hoàn toàn độc lập về mặt ngữ nghĩa. Ranh giới giữa **kinh doanh công nghệ (`Business` vs `Sci/Tech`)** và **kinh tế – chính trị quốc tế (`World` vs `Business`)** là nguồn phát sinh lỗi lớn nhất của mọi mô hình.
2. **Thiên kiến từ khóa (Keyword Bias / Lexical Overlap):**
   * Các từ mang tính ẩn dụ hoặc đa nghĩa xuất hiện trong ngữ cảnh khác dễ gây nhiễu (ví dụ: các từ như *"race"*, *"champion"*, *"victory"* xuất hiện trong bài báo cạnh tranh thị phần kinh doanh hoặc bầu cử dễ kéo dự đoán lệch sang `Sports` — thể hiện qua 61 mẫu `World → Sports` ở Baseline và 60 mẫu ở Frozen Encoder).
3. **Bản tin ngắn, thiếu ngữ cảnh hoặc nhiều từ viết tắt:**
   * Nhiều mẫu trong AG News chỉ gồm tiêu đề ngắn gọn, chứa tên viết tắt của tổ chức, mã chứng khoán hoặc tên riêng địa phương ít xuất hiện trong tập huấn luyện, khiến mô hình thiếu thông tin ngữ cảnh để suy luận.
4. **Tính tương đối trong nhãn gốc (Ambiguous Ground-Truth):**
   * Dữ liệu AG News được tổng hợp từ các chuyên mục của nhiều hãng tin khác nhau; cùng một sự kiện kinh tế – chính trị quốc tế có thể được nguồn này xếp vào `World` nhưng nguồn khác lại xếp vào `Business`. Đây là ngưỡng giới hạn tự nhiên (Bayes error) của tập dữ liệu.

---

## 4. Hạn chế của từng phương pháp rút ra từ Error Analysis

| Phương pháp | Hạn chế chính | Biểu hiện qua kết quả thực nghiệm |
| :--- | :--- | :--- |
| **1. TF-IDF + Logistic Regression (Baseline)** | • Biểu diễn túi từ (Bag-of-Words), bỏ qua trật tự từ và cấu trúc cú pháp.<br>• Không phân biệt được từ đa nghĩa theo ngữ cảnh. | Đạt 91.80% Accuracy (623 lỗi) nhờ bắt từ khóa tốt, nhưng dễ nhầm ở các câu chứa từ khóa công nghệ trong bối cảnh tài chính (`151` mẫu `Business → Sci/Tech`). |
| **2. Frozen Encoder (DistilBERT)** | • Toàn bộ 66.36 triệu tham số của DistilBERT bị đóng băng, chỉ huấn luyện 3,076 tham số ở Classification Head.<br>• Không gian đặc trưng chưa được điều chỉnh cho bài toán phân loại tin tức. | Có số lỗi cao nhất (**734 lỗi**, Accuracy 90.34%), đặc biệt suy giảm mạnh khả năng tách biệt `Sci/Tech` và `Business` (`197` mẫu `Sci/Tech → Business`). |
| **3. Fine-Tuning (DistilBERT)** | • Chi phí tính toán và kích thước lưu trữ mô hình lớn (~253.17 MB).<br>• Vẫn chịu giới hạn ở các mẫu có ranh giới nhãn gốc mơ hồ. | *(Điền số liệu tổng lỗi và nhận xét mức độ cải thiện của Fine-Tuning sau khi cập nhật kết quả full 7,600 mẫu)*. |