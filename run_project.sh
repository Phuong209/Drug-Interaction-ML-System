#!/bin/bash
# Mã khóa đường dẫn cứng để ép máy Mac nhận diện 100% PyTorch
export PATH="/opt/miniconda3/envs/ml_env_312/bin:$PATH"

echo "📡 Kết nối ép buộc vào lõi AI (ml_env_312)..."
echo "🟢 Hệ thống PyTorch nội bộ đã khóa mục tiêu. Bắt đầu tải Web..."

# Ép hệ thống dùng chính xác lõi Python của môi trường 3.12 để bật Web
/opt/miniconda3/envs/ml_env_312/bin/python -m streamlit run app.py
