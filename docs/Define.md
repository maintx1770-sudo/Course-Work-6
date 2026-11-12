## 1. Tên dự án

### Phân loại chủ đề tin tức bằng mô hình dựa trên Transformer

Xây dựng hệ thống phân loại văn bản sử dụng các mô hình dựa trên Transformer để tự động phân loại bài báo vào bốn danh mục của bộ dữ liệu AG News: World, Sports, Business và Sci/Tech.

Dự án so sánh hai hướng tiếp cận:
- Frozen Transformer Encoder (đóng băng bộ mã hóa)
- Fine-Tuned Transformer (tinh chỉnh toàn bộ mô hình)

## 2. Vấn đề

Các nền tảng tin tức trực tuyến đăng tải một lượng lớn bài báo mỗi ngày. Việc gán chủ đề thủ công tốn nhiều thời gian và nhân lực.

Dự án hướng tới việc tự động dự đoán chủ đề của một bài báo bằng mô hình Transformer đã được huấn luyện trước (pretrained).

### Câu hỏi nghiên cứu

Việc fine-tuning một Transformer đã huấn luyện trước so với việc dùng Transformer encoder đóng băng (frozen) khác nhau như thế nào về hiệu năng phân loại và chi phí tính toán trong bài toán phân loại chủ đề AG News?

## 3. Đối tượng sử dụng

- Các nền tảng tin tức trực tuyến
- Các tờ báo điện tử
- Các nhóm quản lý nội dung
- Các nhà phân tích dữ liệu tin tức
- Nhà nghiên cứu và sinh viên

## 4. Đầu vào / Đầu ra

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

## 5. Tính năng chính

### Tính năng 1 – Phân loại chủ đề tin tức
Tự động phân loại một bài báo vào một trong bốn danh mục.

### Tính năng 2 – Phân loại dựa trên Transformer
Sử dụng BERT hoặc DistilBERT để học thông tin ngữ cảnh từ văn bản tin tức.

### Tính năng 3 – Frozen Encoder
Đóng băng Transformer đã huấn luyện trước và chỉ huấn luyện lớp phân loại (classification head).

### Tính năng 4 – Fine-Tuning
Huấn luyện cả Transformer đã huấn luyện trước và lớp phân loại.

### Tính năng 5 – Đánh giá mô hình
Đánh giá bằng:
- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

### Tính năng 6 – So sánh mô hình
So sánh Frozen Encoder và Fine-Tuning dựa trên:
- Hiệu năng phân loại
- Thời gian huấn luyện
- Thời gian suy luận (inference)
- Số tham số có thể huấn luyện
- Kích thước mô hình

## 6. Ràng buộc kỹ thuật

- Hạn chế về CPU/GPU và bộ nhớ
- Độ dài chuỗi tối đa của Transformer
- Chi phí tính toán của fine-tuning
- Bộ dữ liệu AG News với bốn danh mục định sẵn
- Dùng cùng cách chia dữ liệu và tiền xử lý để so sánh công bằng
- Ghi lại random seed và các siêu tham số (hyperparameters) để có thể tái lập kết quả

## 7. Tiêu chí đánh giá

### Hiệu năng phân loại
- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix

### Hiệu năng tính toán
- Thời gian huấn luyện
- Thời gian suy luận
- Số tham số có thể huấn luyện
- Kích thước mô hình

### Tiêu chí thành công
1. Phân loại được bài báo AG News vào bốn danh mục.
2. Cài đặt được cả Frozen Encoder và Fine-Tuning.
3. Đánh giá cả hai hướng tiếp cận trên cùng một tập test.
4. So sánh hiệu năng và chi phí tính toán.
5. Ghi lại các thiết lập để có thể tái lập thí nghiệm.
