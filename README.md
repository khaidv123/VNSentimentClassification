# VNSentimentClassification

## Project goals
Phân loại cảm xúc của các bình luận trên mạng xã hội, Tiktok, Facebook,...

## Task description
- input một câu bình luận tiếng Việt
- output : xác suất đầu ra thuộc lớp positive, negative

## Mô tả các giai đoạn/modules 
1. Crawl data từ tiktok,face,...
2. Xây dựng embedding vector từ hơn 8000 câu (không có label)
3. Tiền xử lý, làm sạch dữ liệu
4. Xây dựng annotation guideline ⇒ tiến hành gán nhãn (2500 câu) 
   - Nếu Inter-annotation < 80% ⇒ kiểm tra lỗi ⇒ Update guideline ⇒ gán nhãn lại
   - Nếu Inter-annotation > 80% ⇒ tạo bộ data 
5. Visualize data
6. Chọn model (RNN, CNN với số lượng các lớp,số chiều ẩn khác nhau)
7. Tiến hành training, đánh giá model
8. Tuning Hyper-parameter
9. Xây dựng giao diện, tích hợp model

![Modules](https://github.com/khaidv123/VNSentimentClassification/assets/111173070/ce7e426d-f645-42c6-b695-5a166e28d549)

## Demo
![Demo](https://github.com/khaidv123/VNSentimentClassification/assets/111173070/155027e4-90c5-4b5d-8e60-7e995f4e91c0)

