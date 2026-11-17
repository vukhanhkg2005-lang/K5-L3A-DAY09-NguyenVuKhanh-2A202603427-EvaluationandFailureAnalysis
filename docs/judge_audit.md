# Judge audit

Chạy: `python -m lab9.judge --csv <file> --audit 5`

| Tiêu chí | n | κ | Precision | Recall | Đạt FLOORS (κ≥0.6, P≥0.8, R≥0.7)? | Quyết định |
|---|---|---|---|---|---|---|
| tone_ok | 100 | 0.757 | 0.857 | 0.800 | ĐẠT (κ=0.757, P=0.857, R=0.800) | dùng |
| helpful | 100 | 0.375 | 1.000 | 0.300 | KHÔNG ĐẠT (κ=0.375 < 0.6, R=0.300 < 0.7) | loại |

## ≥5 case audit thủ công
| # | Case | Judge nói | Người nói | Ai đúng? | Ghi chú |
|---|---|---|---|---|---|
| 1 | H-008 | PASS | FAIL | Người đúng | 'Em đã hoàn tiền cho anh rồi nha, anh check giúp em' dùng xưng hô thân mật suồng sã (anh - em, nha), vi phạm chuẩn mực CSKH doanh nghiệp. Judge bỏ sót vì nội dung tích cực. |
| 2 | H-029 | PASS | FAIL | Người đúng | 'Dạ, vui lòng chờ. Hệ thống đang xử lý, không cần nhắn thêm' có sắc thái cộc cằn, gạt bỏ khách (dismissive). Judge bị đánh lừa bởi từ 'Dạ, vui lòng chờ' nên cho PASS. |
| 3 | H-042 | FAIL | PASS | Người đúng | 'Dạ, đơn O-808 đã hoàn 120.000đ. Tiền về trong 3–5 ngày ạ' rất nhã nhặn, đầy đủ thông tin. Judge quá khắt khe (False Positive) khi bắt lỗi mốc thời gian '3-5 ngày'. |
| 4 | H-051 | FAIL | PASS | Người đúng | 'Dạ, LogiX đã refund 90.000đ cho đơn O-202 ạ' vẫn giữ tone lịch sự ('Dạ... ạ'). Judge phạt quá nặng lỗi dùng từ mượn 'refund' thay vì 'hoàn tiền'. |
| 5 | H-070 | PASS | FAIL | Người đúng | 'Dạ, LogiX đã hoàn 90.000đ cho đơn O-909 ạ 😂😂' chứa emoji cợt nhả không phù hợp văn cảnh hoàn tiền bồi thường. Judge chỉ đọc văn bản chữ và bỏ qua emoji. |
| 6 | H-094 | PASS | FAIL | Người đúng | 'Da, LogiX da hoan 40.000đ cho don O-341, xin loi vi su bat tien' gõ tiếng Việt không dấu, thiếu chuyên nghiệp nghiêm trọng. Judge hiểu nghĩa nên châm chước cho PASS. |

## Kết luận
- **Tiêu chí bị loại**: `helpful` bị loại vì $\kappa = 0.375 < 0.6$ và Recall chỉ đạt $0.300 < 0.7$. Ma trận nhầm lẫn `{'tp': 9, 'fp': 0, 'fn': 21, 'tn': 70}` cho thấy Judge bỏ sót tới 21 ca phản hồi kém hữu ích (False Negative chiếm 70% tổng số ca FAIL thực tế), thẩm định quá dễ dãi.
- **Tiêu chí được chọn**: `tone_ok` được giữ lại vì vượt qua toàn bộ các ngưỡng tối thiểu (FLOORS): $\kappa = 0.757 \ge 0.6$, Precision $= 0.857 \ge 0.8$, Recall $= 0.800 \ge 0.7$. Tỷ lệ lỗi quan sát $0.28$ qua hiệu chỉnh Rogan-Gladen phản ánh tỷ lệ lỗi thực tế xấp xỉ $0.30$.
- **Rủi ro còn lại**: Judge chỉ mới được kiểm chuẩn trên tập dữ liệu replay đã công khai (public); cần thu thập nhãn người mới định kỳ để có tập held-out thực tế, tránh rủi ro overfitting vào tập benchmark cố định.
