import streamlit as st
import os
import sys
import time

# Chèn thư mục src/ vào hệ thống để load code
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from inference import DrugNERPipeline
from interaction import DrugInteractionChecker
from ocr_engine import MedicalOCREngine

# Config trang Web hiện đại
st.set_page_config(page_title="Hệ Sinh Thái Y Tế AI", page_icon="🏥", layout="wide")

# CSS
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #1A2980, #26D0CE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-header {
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .drug-badge {
        background-color: #26D0CE;
        color: #fff;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 5px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .ocr-box {
        background-color: #f1f3f5;
        border-left: 5px solid #26D0CE;
        padding: 10px;
        border-radius: 5px;
        color: #333;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN ĐỘNG CHUẨN ĐỂ DEPLOY SERVER
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "ner_model")

@st.cache_resource
def load_systems():
    # Cache lại toàn bộ Model nặng (CPU/GPU) để app siêu mượt
    _ner = DrugNERPipeline(MODEL_DIR)
    _checker = DrugInteractionChecker(DATA_DIR)
    _ocr = MedicalOCREngine(use_gpu=False)  # Để false cho mọi máy Mac đều chạy được an toàn
    return _ner, _checker, _ocr

st.markdown('<p class="main-header">🏥 HỆ SINH THÁI Y TẾ ĐA PHƯƠNG THỨC</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Bóc Tách Thị Giác (OCR) ➔ Phân Tích NLP (BioBERT) ➔ Cảnh Báo Tương Tác</p>', unsafe_allow_html=True)
st.markdown("---")

# Kiểm tra thư mục trước khi Load
if not os.path.exists(MODEL_DIR):
    st.error("🚨 CHƯA CÓ MODEL AI! Vui lòng đặt thư mục `ner_model` tải từ Kaggle vào mục `models/`.")
    st.stop()

try:
    with st.spinner("⏳ Đang nạp cụm siêu máy tính Deep Learning (BioBERT & EasyOCR)..."):
        ner_engine, db_engine, ocr_engine = load_systems()
except Exception as e:
    st.error(f"Xảy ra lỗi khi nạp hệ thống: {e}")
    st.stop()

# ==========================================
# CẤU TRÚC ĐA PHƯƠNG THỨC (MULTI-MODAL)
# ==========================================
tabs = st.tabs(["✍️ Soạn Bệnh Án Bằng Tay", "📸 Quét Toa Thuốc Bằng Ảnh (OCR)"])
user_text_to_analyze = ""

# TAB 1: NHẬP VĂN BẢN
with tabs[0]:
    st.markdown("#### 📝 Điền thông tin bệnh án, đơn thuốc hoặc lời dặn của bác sĩ:")
    sample_text = "Bệnh nhân có sử dụng Aspirin. Sáng nay có dấu hiệu cảm nên dùng chung với Ibuprofen."
    text_input = st.text_area("Văn bản báo cáo", value=sample_text, height=150, label_visibility="collapsed")
    
    if st.button("🔍 PHÂN TÍCH VĂN BẢN BẰNG AI", type="primary", use_container_width=True):
        user_text_to_analyze = text_input

# TAB 2: QUÉT ẢNH (CAMERA / UPLOAD)
with tabs[1]:
    st.markdown("#### 📸 Tải lên hình ảnh Toa thuốc, Bệnh án hoặc chụp Vỏ Hộp Thuốc:")
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("📂 Chọn ảnh từ máy tính (JPG, PNG)", type=['jpg', 'jpeg', 'png'])
    with col2:
        camera_file = st.camera_input("📷 Hoặc dùng Camera chụp trực tiếp")
        
    img_to_process = uploaded_file if uploaded_file else camera_file
    
    if img_to_process:
        st.image(img_to_process, caption="Ảnh Đã Nạp", width=350)
        if st.button("👁️ QUÉT ẢNH VÀ XỬ LÝ (OCR)", type="primary", use_container_width=True):
            with st.spinner("Đôi mắt AI đang chép lại nội dung từ bức ảnh..."):
                time.sleep(1) # Chờ animation
                try:
                    raw_text = ocr_engine.extract_text(img_to_process)
                    st.success("✅ Chép chữ hoàn tất!")
                    st.markdown(f'<div class="ocr-box">"{raw_text}"</div>', unsafe_allow_html=True)
                    st.write("")
                    user_text_to_analyze = raw_text
                except Exception as e:
                    st.error(f"Lỗi khi đọc ảnh: {e}")

# ==========================================
# 🧠 BỘ NÃO XỬ LÝ CHUNG (NLP PIPELINE)
# ==========================================
if user_text_to_analyze.strip() != "":
    st.markdown("---")
    st.markdown("### 🧬 BÁO CÁO CỦA TRÍ TUỆ NHÂN TẠO")
    
    # BƯỚC 1: XÉT NGHIỆM NER
    with st.spinner("AI NLP đang đọc hiểu ngôn ngữ và trích xuất chất hoá học..."):
        extracted_drugs = ner_engine.extract_drugs(user_text_to_analyze)
        
    if not extracted_drugs:
        st.info("💡 Không phát hiện tên thuốc nào (hoặc các từ không mang dược lý) trong nội dung trên.")
    else:
        # Trang trí lại tên thuốc
        st.markdown(f"**💊 Các loại thuốc AI bắt được ({len(extracted_drugs)} chất):**")
        html_badges = "".join([f'<span class="drug-badge">💊 {drug.title()}</span>' for drug in extracted_drugs])
        st.markdown(html_badges, unsafe_allow_html=True)
        st.write("")
            
        # BƯỚC 2: TRA CỨU DRUGBANK
        st.markdown("#### 🚨 RÀ SOÁT TƯƠNG TÁC THUỐC (Theo Viện DrugBank Canada)")
        with st.spinner("Đang truy vấn cơ sở dữ liệu 2 triệu bản ghi..."):
            conflicts = db_engine.check_list(extracted_drugs)
            
        if conflicts:
            st.error(f"⚠️ CẢNH BÁO ĐỎ: Phát hiện **{len(conflicts)}** tương tác thuốc nguy hiểm trong tổ hợp này!")
            
            for c in conflicts:
                with st.expander(f"🔴 Tương tác: {c['drug1']} 💥 {c['drug2']}", expanded=True):
                    st.markdown(f"**Mức độ nguy hiểm:** Cần báo bác sĩ lập tức.")
                    st.markdown(f"**Cơ chế:** {c['description']}")
        else:
            st.balloons()
            st.success("✅ Tuyệt vời! Theo chuẩn DrugBank, các loại thuốc kể trên hoàn toàn an toàn khi kết hợp với nhau.")
