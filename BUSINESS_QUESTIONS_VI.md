# Câu hỏi nghiệp vụ cho FHIR Agent

Tài liệu này là ngân hàng câu hỏi tham khảo cho bác sĩ và nhân viên y tế.
Thay các giá trị trong dấu `[]` bằng thông tin thực tế. Việc hệ thống trả lời
được câu hỏi phụ thuộc vào dữ liệu FHIR hiện có.

## 1. Tìm kiếm và nhận diện bệnh nhân

1. Tìm bệnh nhân có tên `[họ tên]`.
2. Bệnh nhân có mã `[mã bệnh nhân]` là ai?
3. Hiển thị thông tin hành chính cơ bản của bệnh nhân `[bệnh nhân]`.
4. Bệnh nhân `[bệnh nhân]` có những mã định danh nào?
5. Có bao nhiêu bệnh nhân trong hệ thống?
6. Liệt kê các bệnh nhân đang có dữ liệu trong hệ thống.
7. Có bệnh nhân nào trùng tên hoặc trùng thông tin định danh không?
8. Những bệnh nhân nào thiếu thông tin nhận diện quan trọng?

## 2. Hồ sơ và tổng quan lâm sàng

1. Tóm tắt hồ sơ lâm sàng hiện có của bệnh nhân `[bệnh nhân]`.
2. Các vấn đề sức khỏe chính của bệnh nhân `[bệnh nhân]` là gì?
3. Hiện có những chẩn đoán nào được ghi nhận cho bệnh nhân `[bệnh nhân]`?
4. Chẩn đoán nào của bệnh nhân `[bệnh nhân]` còn hiệu lực?
5. Bệnh nhân `[bệnh nhân]` có bệnh nền hoặc bệnh mạn tính nào?
6. Có thông tin lâm sàng nào đang mâu thuẫn trong hồ sơ không?
7. Những dữ liệu quan trọng nào còn thiếu trong hồ sơ của bệnh nhân?
8. Những thay đổi đáng chú ý gần đây trong hồ sơ bệnh nhân là gì?

## 3. Lượt khám và tiếp nhận

1. Liệt kê các lượt khám của bệnh nhân `[bệnh nhân]`.
2. Lượt khám gần nhất của bệnh nhân diễn ra khi nào?
3. Bệnh nhân đã đến những khoa hoặc cơ sở nào?
4. Trạng thái hiện tại của lượt khám `[mã lượt khám]` là gì?
5. Lý do tiếp nhận của lượt khám `[mã lượt khám]` là gì?
6. Ai là người hoặc đơn vị phụ trách lượt khám?
7. Những chẩn đoán nào gắn với lượt khám này?
8. Những chỉ định, kết quả và thủ thuật nào phát sinh trong lượt khám?
9. Có lượt khám nào chưa kết thúc hoặc thiếu thông tin kết thúc không?
10. Tóm tắt toàn bộ hoạt động trong lượt khám theo thứ tự thời gian.

## 4. Chẩn đoán và vấn đề sức khỏe

1. Liệt kê các chẩn đoán của bệnh nhân `[bệnh nhân]`.
2. Chẩn đoán `[mã hoặc tên bệnh]` được ghi nhận khi nào?
3. Chẩn đoán này liên quan đến lượt khám nào?
4. Trạng thái xác nhận và trạng thái lâm sàng của chẩn đoán là gì?
5. Những bệnh nhân nào được ghi nhận chẩn đoán `[mã hoặc tên bệnh]`?
6. Có bao nhiêu bệnh nhân có chẩn đoán thuộc nhóm `[nhóm bệnh]`?
7. Chẩn đoán nào xuất hiện nhiều nhất trong `[khoảng thời gian]`?
8. Có chẩn đoán nào bị ghi lặp hoặc có mã và tên không nhất quán không?
9. Diễn biến chẩn đoán của bệnh nhân thay đổi như thế nào qua các lượt khám?
10. Những chẩn đoán nào chưa có bằng chứng hoặc kết quả liên quan trong hồ sơ?

## 5. Dị ứng và cảnh báo an toàn

1. Bệnh nhân `[bệnh nhân]` có dị ứng nào được ghi nhận?
2. Tác nhân và biểu hiện của từng dị ứng là gì?
3. Mức độ nghiêm trọng và trạng thái xác nhận của dị ứng là gì?
4. Có thuốc đang dùng nào cần lưu ý dựa trên dữ liệu dị ứng không?
5. Có cảnh báo an toàn nào liên quan đến bệnh nhân không?
6. Những bệnh nhân nào chưa có dữ liệu dị ứng?
7. Có bản ghi dị ứng nào thiếu tác nhân hoặc trạng thái không?

## 6. Thuốc và điều trị

1. Bệnh nhân `[bệnh nhân]` đang được chỉ định những thuốc nào?
2. Những thuốc nào đã được sử dụng trong lượt khám `[mã lượt khám]`?
3. Liều dùng, đường dùng và thời gian dùng của thuốc `[tên thuốc]` là gì?
4. Trạng thái của từng chỉ định thuốc hiện tại là gì?
5. Thuốc nào đã ngừng và lý do ngừng có được ghi nhận không?
6. Lịch sử dùng thuốc của bệnh nhân thay đổi như thế nào?
7. Những thuốc nào liên quan đến chẩn đoán hoặc lượt khám đang xem?
8. Có chỉ định thuốc nào thiếu liều, đơn vị hoặc đường dùng không?
9. Có dữ liệu nào cho thấy các chỉ định thuốc bị trùng lặp không?
10. Tổng hợp các thuốc theo từng lượt khám của bệnh nhân.

## 7. Xét nghiệm và quan sát lâm sàng

1. Liệt kê các xét nghiệm đã thực hiện cho bệnh nhân `[bệnh nhân]`.
2. Kết quả gần nhất của xét nghiệm `[tên xét nghiệm]` là gì?
3. Giá trị, đơn vị và khoảng tham chiếu của kết quả là gì?
4. Kết quả nào được đánh dấu bất thường?
5. Những kết quả nào chưa có giá trị hoặc chưa hoàn tất?
6. Diễn biến của chỉ số `[tên chỉ số]` trong `[khoảng thời gian]` ra sao?
7. Các xét nghiệm nào được thực hiện trong lượt khám `[mã lượt khám]`?
8. Xét nghiệm nào liên quan trực tiếp đến chẩn đoán đang xem?
9. Có kết quả nào thiếu đơn vị hoặc khoảng tham chiếu không?
10. So sánh kết quả mới nhất với kết quả trước đó của bệnh nhân.

## 8. Chẩn đoán hình ảnh

1. Bệnh nhân `[bệnh nhân]` đã thực hiện những dịch vụ chẩn đoán hình ảnh nào?
2. Kết quả chẩn đoán hình ảnh gần nhất là gì?
3. Kết luận chính của báo cáo `[mã báo cáo]` là gì?
4. Báo cáo này liên quan đến chỉ định và lượt khám nào?
5. Có báo cáo nào chưa có kết luận hoặc chưa hoàn tất không?
6. Những hình ảnh hoặc tài liệu nào gắn với báo cáo?
7. So sánh các kết luận hình ảnh theo thời gian.

## 9. Chỉ định, yêu cầu dịch vụ và thủ thuật

1. Các chỉ định đang chờ thực hiện của bệnh nhân `[bệnh nhân]` là gì?
2. Ai đã tạo chỉ định và chỉ định được tạo khi nào?
3. Trạng thái hiện tại của chỉ định `[mã chỉ định]` là gì?
4. Chỉ định đã tạo ra kết quả hoặc báo cáo nào?
5. Những thủ thuật nào đã được thực hiện cho bệnh nhân?
6. Thủ thuật diễn ra khi nào, ở đâu và trong lượt khám nào?
7. Có chỉ định nào đã hoàn thành nhưng chưa có kết quả không?
8. Có kết quả nào không tìm thấy chỉ định nguồn không?
9. Dựng chuỗi từ chỉ định đến thực hiện và trả kết quả.

## 10. Hành trình chăm sóc

1. Dựng lại hành trình chăm sóc của bệnh nhân `[bệnh nhân]` theo thời gian.
2. Tóm tắt hành trình từ tiếp nhận đến kết thúc lượt khám.
3. Nêu các mốc chẩn đoán, chỉ định, thực hiện và trả kết quả.
4. Những chuyên khoa, cơ sở và nhân viên y tế nào đã tham gia?
5. Có khoảng thời gian nào trong hành trình thiếu dữ liệu không?
6. Những sự kiện nào thay đổi hướng xử trí hoặc điều trị?
7. So sánh hai lượt khám gần nhất của bệnh nhân.
8. Tóm tắt hành trình ở mức tổng quan, chỉ mở rộng các sự kiện có ý nghĩa.

## 11. Chăm sóc liên tục và chuyển tuyến

1. Bệnh nhân có kế hoạch chăm sóc nào đang hoạt động?
2. Các mục tiêu chăm sóc và hoạt động dự kiến là gì?
3. Có yêu cầu chuyển tuyến hoặc hội chẩn nào?
4. Bệnh nhân đã được chuyển đến đơn vị hoặc cơ sở nào?
5. Trạng thái của yêu cầu chuyển tuyến hiện tại là gì?
6. Có tài liệu nào được gửi kèm trong quá trình chuyển tuyến?
7. Những nhiệm vụ chăm sóc nào chưa hoàn thành?
8. Ai đang chịu trách nhiệm cho bước chăm sóc tiếp theo?

## 12. Lịch hẹn và theo dõi

1. Bệnh nhân `[bệnh nhân]` có lịch hẹn nào sắp tới?
2. Lịch hẹn thuộc chuyên khoa, cơ sở hoặc dịch vụ nào?
3. Trạng thái của lịch hẹn `[mã lịch hẹn]` là gì?
4. Có lịch hẹn nào bị hủy hoặc bệnh nhân không đến không?
5. Sau lượt khám gần nhất có kế hoạch tái khám nào?
6. Những bệnh nhân nào cần theo dõi nhưng chưa có lịch hẹn?
7. Có lịch hẹn nào thiếu người phụ trách hoặc địa điểm không?

## 13. Nhân viên y tế, tổ chức và cơ sở

1. Ai tham gia chăm sóc bệnh nhân `[bệnh nhân]`?
2. Nhân viên y tế `[tên hoặc mã]` thuộc tổ chức nào?
3. Những lượt khám nào do nhân viên y tế này phụ trách?
4. Những cơ sở hoặc địa điểm nào xuất hiện trong hành trình bệnh nhân?
5. Khoa hoặc đơn vị nào tiếp nhận nhiều lượt khám nhất trong `[khoảng thời gian]`?
6. Những loại dịch vụ nào được thực hiện tại `[cơ sở]`?
7. Có hồ sơ nhân viên, tổ chức hoặc địa điểm nào thiếu thông tin liên kết không?

## 14. Thanh toán và bảo hiểm

1. Lượt khám `[mã lượt khám]` phát sinh những yêu cầu thanh toán nào?
2. Trạng thái của yêu cầu thanh toán hiện tại là gì?
3. Những dịch vụ hoặc chi phí nào được ghi nhận trong yêu cầu?
4. Bệnh nhân có thông tin bảo hiểm hoặc quyền lợi nào liên quan?
5. Yêu cầu thanh toán liên kết với lượt khám và dịch vụ nào?
6. Có yêu cầu nào thiếu kết quả xử lý hoặc thông tin người chi trả không?
7. Dựng chuỗi từ dịch vụ đã thực hiện đến yêu cầu và kết quả thanh toán.

## 15. Báo cáo tổng hợp và chất lượng dữ liệu

1. Có bao nhiêu resource thuộc từng loại trong hệ thống?
2. Những loại dữ liệu nào đang có nhiều bản ghi nhất?
3. Có bao nhiêu lượt khám trong `[khoảng thời gian]`?
4. Những chẩn đoán nào thường xuất hiện cùng nhau?
5. Những trường dữ liệu quan trọng nào thường bị thiếu?
6. Có Reference nào không phân giải được đến resource đích không?
7. Có Coding nào chưa xác định được ý nghĩa không?
8. Có bản ghi nào trùng mã định danh hoặc liên kết không nhất quán không?
9. Những lượt khám nào thiếu chẩn đoán, chỉ định hoặc kết quả?
10. Báo cáo phạm vi dữ liệu hiện có và các giới hạn cần lưu ý.

## 16. Kịch bản request kiểm thử

Tất cả các conversation dưới đây phải dùng chung một tài khoản. Mỗi nhóm tạo
một conversation riêng và gửi câu hỏi theo đúng thứ tự trong nhóm.

### Conversation 1 - Tổng quan hệ thống

1. Có bao nhiêu bệnh nhân trong hệ thống?
2. Liệt kê đầy đủ các bệnh nhân và cho biết bệnh nhân nào có dữ liệu chẩn đoán.

### Conversation 2 - Hồ sơ một bệnh nhân

1. Tìm bệnh nhân có tên NAM VŨ và cho biết mã bệnh nhân.
2. Liệt kê các chẩn đoán đã được ghi nhận cho bệnh nhân này.

### Conversation 3 - Hành trình chăm sóc

1. Dựng lại hành trình chăm sóc của bệnh nhân NAM VŨ theo thứ tự thời gian.
2. Chỉ tóm tắt các mốc chẩn đoán, chỉ định, thực hiện và trả kết quả.

### Conversation 4 - Lượt khám và dữ liệu liên quan

1. Liệt kê các lượt khám của bệnh nhân NAM VŨ.
2. Với lượt khám gần nhất, cho biết các chẩn đoán và kết quả liên quan.

### Conversation 5 - Thuốc và xét nghiệm

1. Bệnh nhân NAM VŨ có dữ liệu thuốc nào được ghi nhận?
2. Liệt kê các xét nghiệm và kết quả hiện có của bệnh nhân này.

### Conversation 6 - Chất lượng và độ đầy đủ

1. Những dữ liệu lâm sàng quan trọng nào đang có trong hồ sơ bệnh nhân NAM VŨ?
2. Trong hồ sơ bệnh nhân này, những nhóm dữ liệu lâm sàng quan trọng nào
   không tìm thấy hoặc đang thiếu trường cần thiết? Chỉ kết luận thiếu sau khi
   đã kiểm tra nguồn dữ liệu liên quan.
