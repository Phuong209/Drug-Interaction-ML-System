# 🏥 Hệ Sinh Thái Y Tế Đa Phương Thức (AI Drug Analyzer)
Hệ thống AI y tế nội bộ sử dụng Mô hình NLP (BioBERT) và Computer Vision (EasyOCR) để tự động hóa việc sàng lọc đơn thuốc và cảnh báo tương tác thuốc dựa trên 2 triệu bản ghi từ cơ sở dữ liệu DrugBank.

## 🚀 Tính Năng Chính
1. **OCR Đơn Thuốc (Thị giác máy tính)**: Nhận biết chữ viết trên ảnh chụp giấy khám bệnh, vỏ hộp thuốc bằng thuật toán Deep Learning.
2. **Khai Phá Bệnh Án Thực Tế (NLP - NER)**: Trích xuất chính xác cấu trúc hóa học và biệt dược (Drug Named Entities) khỏi văn bản y khoa phức tạp.
3. **Tra cứu tương tác thời gian thực**: Thuật toán quét O(1) rà soát database với cảnh báo rủi ro từ thư viện quốc gia Canada (DrugBank).

## 🛠 Hướng Dẫn Cài Đặt & Triển Khai Server

### 1. Chuẩn bị môi trường
Hệ thống tương thích tốt nhất với `Python 3.12`.
Mở Terminal, trỏ đến thư mục mã nguồn và cài đặt:
```bash
pip install -r requirements.txt
```

### 2. Tải Dữ liệu & Cấu Trúc Model (Rất Quan Trọng)
> **Lưu ý:** Github giới hạn dung lượng nên không thể đẩy thư mục AI (nặng 1GB) lên Repo này. Để Server của bạn chạy được Code, bạn phải TỰ TẠO thủ công 2 thư mục `Data` và `models` nhé.

Cấu trúc cây dữ liệu bắt buộc phải có để Load Web:
```
.
├── Data/
│   └── processed/
│       ├── drugbank_drugs.csv
│       ├── drugbank_interactions.csv
├── models/
│   └── ner_model/
│       ├── config.json
│       ├── model.safetensors
│       └── ... (các file bộ não của Model kéo từ Kaggle về)
├── src/
│   ├── inference.py
│   ├── interaction.py
│   ├── ocr_engine.py
├── app.py
├── requirements.txt
└── .gitignore
```

### 3. Ra mắt Server Giao Diện Web
Sau khi hoàn tất cài đặt, gõ lệnh sao để bật Web Server:
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động nhảy lên trình duyệt tại cổng `http://localhost:8501`.
