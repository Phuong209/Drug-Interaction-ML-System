import easyocr
import numpy as np
from PIL import Image

class MedicalOCREngine:
    def __init__(self, use_gpu: bool = False):
        print("Mở mắt thần EasyOCR (Nhận Diện Thuốc)...")
        # Sử dụng mô hình nhận diện Tiếng Anh (chuẩn quốc tế cho tên thuốc)
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)

    def extract_text(self, image_input) -> str:
        """
        Đọc ảnh UploadedFile từ Streamlit, chuyển thành văn bản Text.
        """
        # Streamlit trả về file dạng st.UploadedFile (giống bytes IO), ta chuyển nó sang mảng của PIL / Numpy
        if hasattr(image_input, 'read'):
            image = Image.open(image_input).convert('RGB')
            img_array = np.array(image)
        else:
            img_array = image_input
            
        # OCR quét toàn bộ hình ảnh, detail=0 chỉ lấy chữ ko lấy toạ độ box
        result = self.reader.readtext(img_array, detail=0)
        
        # Nối các cụm từ rời rạc tìm được thành một câu dài liên kết đầy đủ
        text = " ".join(result)
        return text
