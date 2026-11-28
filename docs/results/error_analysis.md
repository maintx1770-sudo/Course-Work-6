# Phân tích lỗi (Error Analysis)

> **Phạm vi:** Phân tích lỗi của ba phương pháp: Baseline (TF-IDF + Logistic Regression), Frozen Encoder (DistilBERT) và Fine-Tuning (DistilBERT).  
> **Dữ liệu:** Tập test AG News gồm 7.600 mẫu, cân bằng 1.900 mẫu cho mỗi lớp: World, Sports, Business và Sci/Tech.  
> **Nguyên tắc:** Error Analysis được thực hiện theo quy trình **Confusion Matrix → Class Pair → Representative Samples → Cause → Decision**.

---

## 1. Tổng quan

Ba phương pháp được đánh giá trên cùng tập test 7.600 mẫu. Kết quả tổng hợp cho thấy Fine-Tuning đạt hiệu năng cao nhất, tiếp theo là Baseline và cuối cùng là Frozen Encoder.

| Mô hình | Accuracy | Macro F1 | Trainable Parameters |
|---|---:|---:|---:|
| Baseline | 0,9196 | 0,9194 | 200.004 |
| Frozen Encoder | 0,9029 | 0,9028 | 3.076 |
| **Fine-Tuning** | **0,9422** | **0,9423** | **66.365.956** |

Fine-Tuning cải thiện Accuracy khoảng 2,26 điểm phần trăm so với Baseline và khoảng 3,93 điểm phần trăm so với Frozen Encoder. Kết quả này cho thấy việc cập nhật toàn bộ tham số của DistilBERT giúp mô hình thích nghi tốt hơn với đặc trưng của bài toán phân loại tin tức AG News.

Tuy nhiên, Accuracy tổng thể không đủ để giải thích nguyên nhân lỗi. Vì vậy, nhóm tiếp tục phân tích theo từng lớp, từng cặp nhãn và các mẫu dự đoán sai tiêu biểu.

---

## 2. Hiệu năng theo từng lớp

Kết quả Classification Report của Baseline và Frozen Encoder cho thấy Sports là lớp dễ phân biệt nhất, trong khi Business và Sci/Tech khó hơn.

| Lớp | Baseline F1 | Frozen Encoder F1 |
|---|---:|---:|
| World | 0,9218 | 0,9039 |
| Sports | 0,9670 | 0,9708 |
| Business | 0,8879 | 0,8617 |
| Sci/Tech | 0,9011 | 0,8748 |

### Interpretation

- **Sports** có F1 cao nhất ở cả Baseline và Frozen Encoder. Các bài viết thể thao thường chứa những từ khóa và ngữ cảnh đặc trưng như tên vận động viên, đội bóng, giải đấu hoặc kết quả thi đấu.
- **Business** và **Sci/Tech** khó phân biệt hơn vì nhiều bài viết có nội dung giao thoa giữa công nghệ, doanh nghiệp, sản phẩm và thị trường.
- Frozen Encoder giảm hiệu năng đáng kể ở Business và Sci/Tech so với Baseline. Điều này cho thấy biểu diễn DistilBERT khi không được điều chỉnh theo AG News chưa đủ phù hợp để tách các lớp có nội dung giao nhau.

### Decision

Nhóm không đánh giá mô hình chỉ dựa trên Accuracy tổng thể mà tiếp tục kiểm tra riêng các lớp Business và Sci/Tech trong Error Analysis. Đây là các lớp cần được ưu tiên khi đánh giá khả năng cải thiện của Fine-Tuning.

---

## 3. Các cặp lớp hay bị nhầm

Đối với Baseline và Frozen Encoder, các cặp nhầm lẫn nổi bật gồm:

| Nhầm (thật → đoán) | Baseline | Frozen Encoder |
|---|---:|---:|
| Business → Sci/Tech | 147 | 164 |
| Sci/Tech → Business | 120 | 180 |
| World → Business | 70 | 103 |
| World → Sports | 56 | 56 |
| Business → World | 58 | 67 |
| Sci/Tech → World | 42 | 68 |

Cặp **Business ↔ Sci/Tech** là pattern lỗi nổi bật nhất.

- Baseline có 267 lỗi ở hai chiều Business ↔ Sci/Tech.
- Frozen Encoder có 344 lỗi ở hai chiều này.
- Frozen Encoder đặc biệt yếu ở chiều **Sci/Tech → Business**, với 180 lỗi so với 120 lỗi của Baseline.

### Interpretation

Kết quả cho thấy lỗi không phân bố ngẫu nhiên mà tập trung ở những cặp lớp có nội dung ngữ nghĩa giao nhau. Do đó, việc chỉ báo cáo số mẫu sai tổng cộng sẽ không giải thích được nguyên nhân mô hình thất bại.

### Decision

Nhóm sử dụng **class pair analysis** thay vì chỉ dựa trên Accuracy để đánh giá mô hình. Đối với Fine-Tuning, cặp Business ↔ Sci/Tech được xem là pattern lỗi cần kiểm tra đầu tiên để xác định liệu việc mở khóa Transformer có thực sự giải quyết được vấn đề này hay không.

---

## 4. Phân tích nguyên nhân lỗi

### 4.1. Chủ đề giao nhau: Business ↔ Sci/Tech

Tin tức về các công ty công nghệ thường đồng thời chứa thông tin về sản phẩm, doanh thu, lợi nhuận hoặc thị trường. Vì vậy, một bài viết có thể hợp lý với cả Business và Sci/Tech.

Ví dụ:

- “Intel to delay product aimed for high-definition TVs…” — nhãn Business nhưng bị đoán Sci/Tech.
- “Storage, servers bruise HP earnings update…” — nội dung liên quan đồng thời đến công nghệ và kết quả kinh doanh.

Đây là trường hợp **topic-overlap**, không thể tự động xem là lỗi hoàn toàn do kiến trúc mô hình.

---

### 4.2. World là lớp có phạm vi rộng

World có thể chứa các sự kiện quốc tế liên quan đến kinh tế, thể thao hoặc khoa học.

Ví dụ:

- “India's Tata expands regional footprint via NatSteel buyout…” — World → Business.
- “Producer Price Surge Fuels Inflation Fears…” — World → Business.
- “Argentina Basketball Coach Magnano Quits…” — World → Sports.
- “Mars water tops science honours…” — World → Sci/Tech.

Những trường hợp này cho thấy một bài viết có thể chứa tín hiệu mạnh của một lớp khác mặc dù nhãn gốc thuộc World.

---

### 4.3. Nhãn gốc có thể không hoàn toàn phản ánh nội dung

Một số mẫu bị cả hai mô hình dự đoán sai với độ tin cậy cao nhưng nội dung thực tế lại phù hợp với nhãn dự đoán.

Ví dụ, một bài viết được gán Business nhưng tập trung nhiều vào giày thể thao và vận động viên có thể bị mô hình dự đoán Sports.

Đây là trường hợp cần xem xét khả năng **label ambiguity/noise** thay vì kết luận ngay rằng mô hình yếu.

---

### 4.4. Văn bản ngắn và thiếu ngữ cảnh

Một số bài báo chỉ có tiêu đề hoặc rất ít nội dung. Khi đó, cả TF-IDF và Transformer đều có thể thiếu đủ thông tin để xác định chủ đề.

Đây là nguyên nhân thuộc nhóm **short/low-context**.

---

### 4.5. Nhiễu định dạng

Một số văn bản chứa các thành phần như:

- HTML entities (`#39;`, `quot;`);
- ký tự đặc biệt;
- thông tin nguồn như `(Reuters) Reuters -`.

`clean_text` đã xử lý phần lớn nhiễu nhưng một số dấu vết định dạng vẫn có thể ảnh hưởng nhẹ đến biểu diễn văn bản.

---

## 5. Quy trình phân loại nguyên nhân lỗi

Để tránh quy toàn bộ lỗi cho mô hình, nhóm phân loại từng mẫu sai theo thứ tự:

1. **Model-driven:** nội dung có tín hiệu rõ cho nhãn thật nhưng mô hình chọn nhãn khác.
2. **Topic-overlap:** nội dung hợp lý với nhiều lớp, đặc biệt Business ↔ Sci/Tech.
3. **Short/low-context:** văn bản quá ngắn hoặc thiếu thông tin.
4. **Possible label ambiguity/noise:** nhãn gốc có thể không phản ánh hoàn toàn nội dung.

Chỉ sau bước phân loại này nhóm mới đưa ra quyết định về mô hình hoặc dữ liệu.

Cách tiếp cận này giúp phân biệt giữa:

```text
Model Error
```

và

```text
Data / Task Difficulty
```

thay vì xem mọi prediction sai đều là bằng chứng cho thấy kiến trúc mô hình chưa tốt.

---

## 6. Phân tích Fine-Tuning

Fine-Tuning đạt:

- **Accuracy = 0,9422**
- **Macro Precision = 0,9424**
- **Macro Recall = 0,9422**
- **Macro F1 = 0,9423**

Kết quả này cao hơn cả Baseline và Frozen Encoder.

Trong `05_comparison.ipynb`, Fine-Tuning được sử dụng để tiếp tục phân tích các prediction sai và các cặp nhãn nhầm lẫn. Kết quả tổng hợp cho thấy:

- Tổng thể Fine-Tuning có ít lỗi hơn hai phương pháp còn lại.
- Business ↔ Sci/Tech vẫn là pattern lỗi quan trọng.
- Theo phân tích trong notebook, Baseline có 267 lỗi ở cặp Business ↔ Sci/Tech, trong khi Fine-Tuning có 243 lỗi, tương ứng giảm khoảng **9,0%**.
- Các cặp lỗi khác giảm từ 344 xuống 196, tương ứng khoảng **43%**.

Tuy nhiên, tỷ trọng của Business ↔ Sci/Tech trong tổng số lỗi tăng từ khoảng 44% lên khoảng **55%** khi Fine-Tuning làm giảm mạnh các nhóm lỗi khác. Vì vậy, việc tỷ trọng tăng không có nghĩa Fine-Tuning làm cặp lỗi này tệ hơn; số lỗi tuyệt đối vẫn giảm.

> **Lưu ý:** số liệu pair-count của Fine-Tuning (243 và 196) lấy từ lần chạy notebook 05 trước đó; repository hiện chưa lưu `fine_tuning_predictions.csv` nên chưa thể tái lập. Tổng 243 + 196 = 439 lỗi khớp với Accuracy chính thức 94,22%. Số liệu Baseline và Frozen Encoder ở trên được tính lại từ file prediction chính thức.


### Interpretation

Fine-Tuning không loại bỏ hoàn toàn lỗi Business ↔ Sci/Tech. Điều này phù hợp với bản chất của AG News, trong đó một số bài viết có chủ đề giao thoa và có thể hợp lý với nhiều nhãn.

Mặt khác, việc giảm các nhóm lỗi khác cho thấy Fine-Tuning cải thiện khả năng phân loại tổng thể thay vì chỉ tối ưu một cặp lớp.

### Decision

Nhóm lựa chọn Fine-Tuning làm mô hình cuối cùng vì:

1. đạt Accuracy và Macro F1 cao nhất;
2. giảm tổng số lỗi so với Baseline và Frozen Encoder;
3. giảm lỗi ở nhiều class pair khác;
4. cải thiện khả năng biểu diễn ngữ cảnh nhờ cập nhật toàn bộ Transformer.

Đối với Business ↔ Sci/Tech, nhóm xem đây chủ yếu là **giới hạn của dữ liệu và sự giao thoa chủ đề**, thay vì tiếp tục tăng độ phức tạp kiến trúc chỉ để xử lý riêng một pattern lỗi.

---

## 7. Confidence và các mẫu dự đoán sai

Một số mẫu sai có confidence cao. Điều này cho thấy confidence không nên được sử dụng như bằng chứng duy nhất để khẳng định prediction là đúng.

Đặc biệt, các mẫu thuộc nhóm:

- topic-overlap;
- label ambiguity/noise;
- short/low-context

có thể khiến mô hình đưa ra prediction với confidence cao mặc dù nhãn dự đoán khác nhãn gốc.

### Decision

Nhóm không sử dụng confidence đơn lẻ để tự động loại bỏ hoặc chấp nhận prediction. Các prediction có confidence cao nhưng sai cần được xem xét cùng với:

```text
Prediction
→ True Label
→ Text
→ Class Pair
→ Cause
```

Cách này phù hợp hơn với mục tiêu Error Analysis vì confidence chỉ phản ánh mức độ chắc chắn của mô hình đối với prediction, không chứng minh rằng nhãn gốc chắc chắn đúng.

---

## 8. Error Analysis → Interpretation → Decision

### Interpretation

Error Analysis cho thấy lỗi tập trung vào một số pattern có hệ thống, đặc biệt là **Business ↔ Sci/Tech**. Baseline và Frozen Encoder đều gặp khó khăn ở hai lớp này, trong đó Frozen Encoder có số lỗi Sci/Tech → Business cao hơn đáng kể.

Fine-Tuning giảm số lỗi tổng thể và đạt hiệu năng cao nhất. Đồng thời, Fine-Tuning cũng giảm lỗi ở Business ↔ Sci/Tech so với Baseline, mặc dù pattern này vẫn là một trong những nguồn lỗi chính.

Các lỗi còn lại có thể xuất phát từ:

- sự giao thoa giữa các chủ đề;
- văn bản ngắn hoặc thiếu ngữ cảnh;
- nhãn gốc có khả năng mơ hồ;
- nhiễu định dạng văn bản.

### Evidence → Decision

Các quyết định được đưa ra dựa trên:

1. Accuracy và Macro F1 trên cùng tập test;
2. Classification Report theo từng lớp;
3. Confusion Matrix;
4. số lượng lỗi theo class pair;
5. các mẫu prediction sai tiêu biểu;
6. nguyên nhân của từng nhóm lỗi.

### Final Decision

**Fine-Tuning được chọn làm mô hình cuối cùng.**

Baseline được giữ làm **phương án dự phòng khi tài nguyên tính toán hạn chế**, vì mô hình có hiệu năng khá cao với chi phí huấn luyện và kích thước mô hình thấp.

Frozen Encoder **không được lựa chọn** vì đạt hiệu năng thấp hơn Baseline nhưng vẫn phải sử dụng DistilBERT và có chi phí trích xuất đặc trưng lớn hơn.

---

## 9. Hạn chế của Error Analysis

Error Analysis hiện tại vẫn có một số hạn chế:

- Một số nguyên nhân lỗi được suy luận từ nội dung mẫu nên chưa thể khẳng định tuyệt đối.
- Topic-overlap và label ambiguity cần được kiểm tra thủ công trên nhiều mẫu hơn để xác nhận.
- Confidence của các mô hình được tạo từ các cơ chế khác nhau nên không nên so sánh trực tiếp như một thước đo calibration nếu chưa thực hiện calibration analysis.
- Chi phí training của ba mô hình được đo trên các thiết bị khác nhau, vì vậy không dùng thời gian training tuyệt đối để kết luận mô hình nào “nhanh hơn” nếu không kiểm soát cùng phần cứng.

Do đó, kết luận chính được dựa trên **hiệu năng trên cùng tập test, pattern lỗi và trade-off giữa chất lượng và chi phí**, thay vì chỉ dựa trên một chỉ số riêng lẻ.
