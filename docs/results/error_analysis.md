# Phân tích lỗi (Error Analysis)

> **Phạm vi:** Baseline (TF-IDF + LR) và Frozen Encoder (DistilBERT). Phần Fine-Tuning sẽ bổ sung sau khi có `fine_tuning_predictions.csv` chính thức (mục 6).
> Dữ liệu: dự đoán trên tập test AG News (7.600 mẫu). Baseline dùng `baseline_test_predictions.csv`, khớp với `baseline_metrics.csv`.

## 1. Tổng quan

| Mô hình | Số mẫu sai | Accuracy | Độ tin cậy TB (đúng) | Độ tin cậy TB (sai) | Sai nhưng tin cậy > 0,9 |
|---|---:|---:|---:|---:|---:|
| Baseline | 623 | 0,9180 | 0,877 | 0,653 | 87 |
| Frozen Encoder | 734 | 0,9034 | 0,925 | 0,713 | 163 |

- 464 mẫu bị **cả hai** mô hình đoán sai; 159 mẫu chỉ Baseline sai; 270 mẫu chỉ Frozen Encoder sai. Phần lớn lỗi là các mẫu "khó thật" chứ không phải lỗi riêng của một phương pháp.
- Frozen Encoder đưa ra nhiều dự đoán sai với độ tin cậy rất cao (163 mẫu > 0,9), nghĩa là độ tin cậy của nó kém hiệu chỉnh hơn Baseline.

## 2. Hiệu năng theo từng lớp (F1)

| Lớp | Baseline | Frozen Encoder |
|---|---:|---:|
| World | 0,9201 | 0,9073 |
| Sports | 0,9653 | 0,9717 |
| Business | 0,8864 | 0,8612 |
| Sci/Tech | 0,8996 | 0,8728 |

Sports dễ nhất ở cả hai mô hình (từ vựng thể thao rất đặc trưng; Frozen Encoder thậm chí nhỉnh hơn Baseline). Business và Sci/Tech khó nhất.

## 3. Các cặp lớp hay bị nhầm

| Nhầm (thật → đoán) | Baseline | Frozen Encoder |
|---|---:|---:|
| Business → Sci/Tech | 151 | 155 |
| Sci/Tech → Business | 118 | 197 |
| World → Business | 75 | 98 |
| World → Sports | 61 | 60 |
| Business → World | 56 | 69 |
| Sci/Tech → World | 46 | 62 |

Cặp **Business ↔ Sci/Tech** chiếm tỉ lệ lỗi lớn nhất ở cả hai mô hình (Baseline: 269/623 ≈ 43%; Frozen Encoder: 352/734 ≈ 48%). Frozen Encoder tệ hơn rõ rệt ở chiều Sci/Tech → Business (197 so với 118).

## 4. Nguyên nhân có thể và ví dụ

1. **Chủ đề giao nhau (Business ↔ Sci/Tech).** Tin về công ty công nghệ vừa là tin kinh doanh vừa là tin công nghệ.
   - "Intel to delay product aimed for high-definition TVs…" (nhãn Business, bị đoán Sci/Tech, tin cậy 0,94).
   - "Storage, servers bruise HP earnings update…" (nhãn Sci/Tech, bị đoán Business, tin cậy 0,69).
2. **World bị kéo sang Business/Sports/Sci/Tech.** World là lớp "rộng": tin quốc tế có thể nói về kinh tế, thể thao hay khoa học.
   - "India's Tata expands regional footprint via NatSteel buyout…" (World → Business).
   - "Producer Price Surge Fuels Inflation Fears…" (World → Business, tin cậy 0,997).
   - "Argentina Basketball Coach Magnano Quits…" (World → Sports, tin cậy 0,996).
   - "Mars water tops science honours…" (World → Sci/Tech, tin cậy 0,998).
3. **Nhãn gốc mơ hồ hoặc nhiễu.** Nhiều mẫu bị cả hai mô hình sai với độ tin cậy gần 1 mà nội dung thật sự khớp với nhãn dự đoán hơn (ví dụ tin về Phil Knight/Nike gán Business nhưng nói về giày thể thao và vận động viên, bị đoán Sports). Đây là giới hạn của dữ liệu, không hẳn là lỗi mô hình.
4. **Văn bản ngắn/thiếu ngữ cảnh** (chỉ tiêu đề + một câu) làm cả TF-IDF lẫn encoder thiếu tín hiệu.
5. **Nhiễu định dạng:** các thực thể HTML (`#39;`, `quot;`), dấu `\`, nguồn tin trong ngoặc như "(Reuters) Reuters -" có thể gây nhiễu nhẹ dù `clean_text` đã xử lý phần lớn.

## 5. Hạn chế của từng phương pháp

- **Baseline (TF-IDF + LR):** không hiểu ngữ nghĩa/ngữ cảnh, chỉ dựa trên sự xuất hiện của từ và cụm hai từ; dễ sai khi bài báo dùng từ vựng của lớp khác (ví dụ tin công ty công nghệ dùng nhiều từ tài chính). Ưu điểm là nhanh và gọn.
- **Frozen Encoder:** đặc trưng DistilBERT không được điều chỉnh cho miền tin tức và chỉ có head tuyến tính nhỏ, nên không tách tốt các lớp giao nhau; tốn thời gian trích xuất đặc trưng; nhiều dự đoán sai tin cậy cao.

## 6. Việc cần bổ sung (Fine-Tuning)

Sau khi có `results/metrics/fine_tuning_predictions.csv` (chạy chính thức), chạy lại `05_comparison.ipynb` rồi bổ sung: số mẫu sai, ma trận nhầm lẫn mới (`confusion_matrix_fine_tuning.png`), các cặp lớp còn nhầm và so sánh với hai phương pháp trên. Cần kiểm tra giả thuyết: fine-tuning có giảm lỗi Business ↔ Sci/Tech hay không, hay đây chủ yếu là nhiễu nhãn của dữ liệu.


## 7.1. Cách nhóm phân biệt nguyên nhân lỗi

Khi xem một mẫu sai, nhóm không được kết luận ngay rằng mô hình yếu. Mẫu được phân loại theo thứ tự:

1. **Model-driven:** nội dung có tín hiệu rõ cho nhãn thật nhưng mô hình chọn nhãn khác.
2. **Topic-overlap:** nội dung hợp lý với cả hai lớp, đặc biệt `Business ↔ Sci/Tech`.
3. **Short/low-context:** văn bản quá ngắn hoặc thiếu ngữ cảnh để phân biệt chủ đề.
4. **Possible label ambiguity/noise:** nhãn gốc có thể không phản ánh hoàn toàn nội dung.

Chỉ sau bước này nhóm mới đưa ra decision về kiến trúc hoặc dữ liệu. Điều này giúp tránh quy toàn bộ lỗi cho mô hình khi nguyên nhân thực tế nằm ở độ giao nhau của chủ đề hoặc chất lượng nhãn.

## 7. Error Analysis → Interpretation → Decision

### Interpretation

Kết quả cho thấy lỗi tập trung mạnh ở cặp **Business ↔ Sci/Tech**. Với Baseline, hai chiều này tạo ra 269/623 lỗi; với Frozen Encoder là 352/734 lỗi. Điều này cho thấy đây không chỉ là một số ít outlier mà là một pattern có hệ thống trong bài toán. Sports là lớp dễ phân biệt nhất, trong khi Business và Sci/Tech khó hơn.

Ngoài ra, việc có các mẫu sai với confidence rất cao cho thấy không nên dùng confidence của mô hình như một bằng chứng duy nhất để khẳng định dự đoán đúng. Các mẫu có nội dung giao nhau hoặc nhãn gốc mơ hồ cần được kiểm tra bằng nội dung thực tế.

### Evidence → Decision

Các quyết định dưới đây phải dựa trực tiếp trên số lỗi theo class pair và các mẫu sai đại diện, thay vì chỉ dựa trên Accuracy tổng.

### Decision

1. Khi đánh giá Fine-Tuning, nhóm phải kiểm tra riêng cặp Business ↔ Sci/Tech để xác định fine-tuning có thực sự giảm pattern lỗi này hay không.
2. Khi phân tích một mô hình, không chỉ báo cáo số lỗi tổng mà phải xem **class pair + mẫu văn bản + nguyên nhân**.
3. Các lỗi do chủ đề giao nhau hoặc nhãn mơ hồ được xem là giới hạn của dữ liệu; không tự động quy kết toàn bộ lỗi cho kiến trúc mô hình.
4. Nếu Fine-Tuning giảm đáng kể các lỗi Business ↔ Sci/Tech mà không làm tăng mạnh lỗi ở các lớp khác, đó là bằng chứng hỗ trợ quyết định sử dụng Fine-Tuning. Nếu không, nhóm phải cân nhắc lại trade-off giữa hiệu năng và chi phí tính toán.

### 8. Quy tắc bổ sung sau Fine-Tuning chính thức

Sau khi có `fine_tuning_predictions.csv`, cập nhật lại toàn bộ bảng lỗi và thực hiện cùng một quy trình: **Confusion Matrix → Class Pair → Representative Samples → Cause → Decision**. Không dùng kết quả Quick Test 0,25 làm kết quả nghiên cứu chính thức.
