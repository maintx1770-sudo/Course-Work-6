# Define – News Topic Classification Using Transformer-Based Models

## 1. Tên dự án

### Phân loại chủ đề tin tức bằng mô hình dựa trên Transformer

Xây dựng hệ thống phân loại văn bản sử dụng các mô hình dựa trên Transformer để tự động phân loại bài báo vào bốn danh mục của bộ dữ liệu AG News: **World, Sports, Business và Sci/Tech**.

Dự án tập trung so sánh ba phương pháp:

- **TF-IDF + Logistic Regression (Baseline)** – mô hình tham chiếu.
- **Frozen Transformer Encoder** – đóng băng Transformer Encoder và chỉ huấn luyện classification head.
- **Fine-Tuned Transformer** – tinh chỉnh toàn bộ Transformer và classification head.

Việc bổ sung Baseline giúp nhóm có một mốc tham chiếu để xác định liệu việc sử dụng Transformer có thực sự mang lại cải thiện về hiệu năng hay không.

---

## 2. Vấn đề

Các nền tảng tin tức trực tuyến đăng tải một lượng lớn bài báo mỗi ngày. Việc gán chủ đề thủ công tốn nhiều thời gian và nhân lực.

Dự án hướng tới việc tự động dự đoán chủ đề của một bài báo dựa trên nội dung văn bản.

### Câu hỏi nghiên cứu

> Việc sử dụng Transformer theo hai cách tiếp cận **Frozen Encoder** và **Fine-Tuning** khác nhau như thế nào về hiệu năng phân loại và chi phí tính toán so với một mô hình Baseline trong bài toán phân loại chủ đề AG News?

---

## 3. Yêu cầu nghiệp vụ

Hệ thống cần đáp ứng các yêu cầu chính:

1. Tự động phân loại một bài báo vào một trong bốn chủ đề.
2. Giảm sự phụ thuộc vào việc phân loại thủ công.
3. Có khả năng cung cấp xác suất dự đoán của các lớp để hỗ trợ đánh giá kết quả.
4. Cho phép so sánh các phương pháp khác nhau để lựa chọn mô hình phù hợp.
5. Cân nhắc đồng thời **độ chính xác phân loại và chi phí tính toán**, thay vì chỉ lựa chọn mô hình có Accuracy cao nhất.

### Quyết định cần hỗ trợ

Kết quả thực nghiệm phải giúp nhóm trả lời:

> **Mô hình nào phù hợp nhất để sử dụng cho bài toán phân loại AG News khi cân bằng giữa hiệu năng và chi phí tính toán?**

---

## 4. Đối tượng sử dụng

- Các nền tảng tin tức trực tuyến
- Các tờ báo điện tử
- Các nhóm quản lý nội dung
- Các nhà phân tích dữ liệu tin tức
- Nhà nghiên cứu và sinh viên

---

## 5. Đầu vào / Đầu ra

### Đầu vào

- Tiêu đề bài báo
- Mô tả bài báo
- Hoặc tiêu đề và mô tả kết hợp

### Đầu ra

- Danh mục được dự đoán
- Xác suất của từng lớp

### Các danh mục

| Nhãn | Danh mục |
|---|---|
| 0 | World (Thế giới) |
| 1 | Sports (Thể thao) |
| 2 | Business (Kinh doanh) |
| 3 | Sci/Tech (Khoa học/Công nghệ) |

---

## 6. Tính năng chính

### Tính năng 1 – Phân loại chủ đề tin tức

Tự động phân loại một bài báo vào một trong bốn danh mục của AG News.

### Tính năng 2 – Baseline

Sử dụng **TF-IDF + Logistic Regression** làm mô hình tham chiếu.

Baseline được sử dụng để xác định mức hiệu năng có thể đạt được bằng một phương pháp machine learning truyền thống trước khi sử dụng Transformer.

### Tính năng 3 – Frozen Encoder

Sử dụng Transformer đã được pretrained nhưng **đóng băng các tham số của Encoder** và chỉ huấn luyện classification head.

Mục tiêu là đánh giá khả năng sử dụng trực tiếp các biểu diễn đã học mà không cần điều chỉnh toàn bộ Transformer.

### Tính năng 4 – Fine-Tuning

Huấn luyện đồng thời Transformer pretrained và classification head trên dữ liệu AG News.

Mục tiêu là đánh giá khả năng thích nghi của mô hình pretrained với bài toán phân loại chủ đề tin tức cụ thể.

### Tính năng 5 – Đánh giá mô hình

Đánh giá các phương pháp bằng:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

### Tính năng 6 – Phân tích lỗi

Phân tích các mẫu dự đoán sai theo:

- Class pair
- Nội dung văn bản
- Mức độ giao nhau giữa các chủ đề
- Văn bản ngắn hoặc thiếu ngữ cảnh
- Khả năng mơ hồ/nhiễu của nhãn

Kết quả Error Analysis được sử dụng để giải thích nguyên nhân lỗi và hỗ trợ quyết định lựa chọn mô hình.

### Tính năng 7 – So sánh mô hình

So sánh ba phương pháp dựa trên:

- Hiệu năng phân loại
- Thời gian huấn luyện
- Thời gian suy luận
- Số tham số có thể huấn luyện
- Kích thước mô hình

---

## 7. Ràng buộc kỹ thuật

- Hạn chế về CPU/GPU và bộ nhớ.
- Độ dài chuỗi tối đa của Transformer.
- Chi phí tính toán của Fine-Tuning.
- Bộ dữ liệu AG News với bốn danh mục định sẵn.
- Sử dụng cùng cách chia dữ liệu và tiền xử lý để đảm bảo so sánh công bằng.
- Sử dụng cùng tập test để đánh giá các phương pháp.
- Ghi lại random seed và hyperparameters để đảm bảo khả năng tái lập.
- Không sử dụng kết quả Quick Test làm kết quả thực nghiệm chính thức.

---

## 8. Tiêu chí đánh giá

### 8.1. Hiệu năng phân loại

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

### 8.2. Hiệu năng tính toán

- Thời gian huấn luyện
- Thời gian suy luận
- Số tham số có thể huấn luyện
- Kích thước mô hình

### 8.3. Phân tích lỗi

Ngoài các metric tổng hợp, nhóm phân tích:

- Lớp nào khó phân loại.
- Các cặp lớp thường bị nhầm.
- Các mẫu có confidence cao nhưng dự đoán sai.
- Nguyên nhân lỗi đến từ mô hình hay từ đặc điểm dữ liệu.

---

## 9. Tiêu chí thành công

Dự án được xem là đạt yêu cầu khi:

1. Phân loại được bài báo AG News vào bốn danh mục.
2. Cài đặt được Baseline, Frozen Encoder và Fine-Tuning.
3. Đánh giá ba phương pháp trên cùng tập test.
4. So sánh được hiệu năng và chi phí tính toán.
5. Phân tích được các lỗi chính của mô hình.
6. Đưa ra được quyết định lựa chọn mô hình dựa trên bằng chứng thực nghiệm.
7. Ghi lại các thiết lập để có thể tái lập thí nghiệm.

---

## 10. Quy trình ra quyết định

Quá trình thực hiện dự án được tổ chức theo chuỗi:

**Problem Analysis → Requirements → EDA → Model Selection → Training → Training Monitoring → Evaluation → Error Analysis → Model Comparison → Final Decision**

Ở mỗi giai đoạn, nhóm thực hiện:

**Quan sát dữ liệu/kết quả → Phân tích → Đưa ra quyết định → Kiểm chứng bằng thực nghiệm**

Ví dụ:

- **EDA:** phân tích dữ liệu và độ dài văn bản → xác định cách preprocessing và cấu hình đầu vào.
- **Model Selection:** cần một mốc tham chiếu và hai cách sử dụng Transformer → chọn Baseline, Frozen Encoder và Fine-Tuning.
- **Training Monitoring:** theo dõi validation performance → xác định checkpoint/epoch phù hợp.
- **Evaluation:** so sánh Accuracy, Precision, Recall và Macro F1 → xác định mô hình có hiệu năng tốt hơn.
- **Error Analysis:** phân tích class pair và mẫu sai → xác định nguyên nhân lỗi.
- **Model Comparison:** kết hợp hiệu năng và chi phí tính toán → đưa ra quyết định mô hình cuối cùng.

### Quyết định cuối cùng

Mô hình cuối cùng không được lựa chọn chỉ dựa trên một metric đơn lẻ mà dựa trên:

**Classification Performance + Computational Cost + Error Analysis + Experimental Evidence**

Điều này đảm bảo quyết định kỹ thuật của nhóm có cơ sở từ dữ liệu và kết quả thực nghiệm.
