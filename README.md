# 🏥 MedSafe AI — Multi-Modal Clinical Pharmacovigilance & Medication Assistant

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![NLP Model](https://img.shields.io/badge/NLP-BioBERT%20NER-green.svg)](https://huggingface.co/)
[![OCR Engine](https://img.shields.io/badge/OCR-EasyOCR-orange.svg)](https://github.com/JaidedAI/EasyOCR)
[![LLM Engine](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4.svg)](https://deepmind.google/technologies/gemini/)

An end-to-end multi-modal medical AI assistant designed for prescription text recognition (EasyOCR), precise clinical named entity extraction (BioBERT NER), real-time drug interaction auditing (DrugBank database), and AI-driven clinical medication scheduling.

*Hệ thống y tế đa phương thức hỗ trợ nhận diện đơn thuốc (OCR), trích xuất tự động danh mục dược phẩm (BioBERT NER), kiểm tra tương tác thuốc thời gian thực (DrugBank) và tự động lập lịch dùng thuốc thông minh theo dược lý lâm sàng.*

---

## 🌟 Key Technical Features / Tính Năng Nổi Bật

### 1. Multi-Modal AI Pipeline (Hệ thống Trí tuệ Nhân tạo Đa phương thức)
*   **Computer Vision (EasyOCR)**: Scans and extracts text from medical prescriptions, clinical letters, or medication packages with automatic preprocessing.
    *(Nhận diện chữ viết từ hình ảnh đơn thuốc, bệnh án bằng thị giác máy tính với xử lý ảnh tối ưu.)*
*   **State-of-the-Art NLP Extraction (BioBERT NER)**: Fine-tuned BioBERT model on Kaggle GPUs, extracting precise active ingredients and brand names with a **99.69% F1-score**.
    *(Mô hình NLP chuyên sâu trích xuất chính xác tên hoạt chất và biệt dược từ văn bản y khoa phức tạp.)*
*   **Clinical LLM Reasoning (Google Gemini)**: Generates clinical medication schedules with drug-class-specific intelligence.
    *(Tích hợp LLM phân tích lập lịch uống thuốc cá nhân hóa thông minh.)*

### 2. Enterprise Data Engineering (Kỹ thuật Dữ liệu & Tra cứu Tối ưu)
*   **O(1) Search Complexity**: Parsed and indexed **2M+ records** from the **DrugBank XML database** into optimized in-memory key-value lookups for instant interaction auditing.
    *(Cấu trúc dữ liệu tối ưu hóa thời gian tra cứu O(1) trên cơ sở dữ liệu hơn 2 triệu bản ghi DrugBank.)*
*   **Strict Pydantic Guardrails**: Implements rigorous runtime data validation for the drug database to enforce absolute medical-grade data integrity.
    *(Rào chắn dữ liệu nghiêm ngặt sử dụng Pydantic bảo vệ tính toàn vẹn của dữ liệu y khoa.)*

### 3. Intelligent Pharmacovigilance & Recommendations (Cảnh báo Tương tác & Đề xuất Thay thế)
*   **Severity Risk Scoring**: Automatically classifies interactions (High, Moderate, Low) based on clinical description analysis.
    *(Đánh giá rủi ro và phân loại mức độ nghiêm trọng của tương tác thuốc.)*
*   **Safe Alternative Suggestions**: Suggests non-interacting pharmacological alternatives when a contraindication is detected.
    *(Gợi ý các loại thuốc thay thế an toàn hơn khi phát hiện có tương tác nguy hại.)*
*   **Safety Interlocks**: Dynamically blocks automated schedule creation when critical drug interactions are discovered.
    *(Tự động khóa chức năng tạo lịch nếu phát hiện tương tác nguy hiểm.)*

### 4. Advanced Clinical Scheduling (Lập Lịch Theo Dược Lý Lâm Sàng)
*   **Prescription-First Priority**: Directly honors exact prescription instructions parsed from OCR text first.
    *(Ưu tiên số một theo đúng chỉ dẫn của bác sĩ trong đơn thuốc.)*
*   **Pharmacology-Driven Fallback**: Auto-assigns optimal times based on drug properties when doctor's instructions are missing:
    *   *Anticoagulants / Statins* $\rightarrow$ Scheduled for evening (optimal therapeutic window).
    *   *NSAIDs* $\rightarrow$ Set for after-meal slots (minimizes gastrointestinal adverse effects).
    *   *Diuretics* $\rightarrow$ Kept out of late-evening slots to prevent sleep disruption.
    *(Tự động phân bổ khung giờ thông minh theo tính chất dược lý lâm sàng nếu đơn thuốc không ghi rõ giờ uống.)*

### 5. Premium Mobile-First UI/UX (Giao Diện Di Động Cao Cấp)
*   Inspired by premium apps like **Medisafe** and **Apple Health**, built using custom CSS glassmorphism in Streamlit:
    *   **Interactive Splash Screen**: Animated launch screen for a premium first impression.
    *   **Bottom Navigation**: Sticky bottom navigation bar for high-fidelity mobile experience.
    *   **Interactive Checkboxes**: "Mark as taken" medication tracker with immediate completion feedback.
    *   **Time-of-Day Greetings**: Dynamic personalized greetings depending on the user's current local hour.
    *(Thiết kế giao diện kính mờ sang trọng, thanh điều hướng chân trang giống ứng dụng di động, splash screen sinh động và theo dõi uống thuốc tương tác.)*

---

## ⚡ v1 Release vs Legacy Version Comparison

| Aspect | Legacy version (v0) | Premium Release (v1) |
| :--- | :--- | :--- |
| **App Startup** | All AI models loaded at startup (30-60s delay) | **Lazy-loading architecture** — near-instant startup, models load only on demand |
| **Interface** | Standard out-of-the-box Streamlit widgets | **Premium Dark Glassmorphism**, custom mobile bottom nav, animated splash screen |
| **State Retention** | App resets and loses data upon checking boxes | **Persistent Session State** — data persists smoothly across interactive check-ins |
| **Scheduling** | Simple LLM text output block | **Visual Timeline Cards** with custom icons, color codes, and checkmarks |
| **Interaction Check** | Simple alert block | **Enterprise Schema Guardrails**, Risk Severity Scoring, and Alternative Suggestions |
| **Safety Controls** | Generates schedule despite hazards | **Interactive safety interlocking** (blocks schedule creation when severe interactions found) |

---

## 🛠️ Architecture & Project Directory / Cấu Trúc Dự Án

```directory
.
├── Data/
│   └── processed/
│       ├── drugbank_drugs.csv
│       └── drugbank_interactions.csv
├── models/
│   └── ner_model/
│       ├── config.json
│       ├── model.safetensors
│       └── ... (BioBERT NER weights)
├── src/
│   ├── inference.py              # BioBERT Named Entity Recognition pipeline
│   ├── interaction.py            # Drug Interaction checking and alternative generator
│   ├── ocr_engine.py             # Computer Vision prescription text scanner
│   ├── data_validators.py        # Pydantic schemas for data validation
│   └── generate_ner_dataset.py   # Training dataset generation scripts
├── tests/
│   └── test_components.py        # Automated test suites for safety validators
├── app.py                        # Streamlit main application & UI logic
├── requirements.txt              # Production dependency packages
├── run_project.sh                # Hardlocked system launch script
└── .gitignore                    # Managed ignore paths
```

---

## 🧪 Validation & Automated Testing
To run the automated test suite confirming data schema integrity and interactions:
*(Chạy kiểm thử tự động để xác nhận tính toàn vẹn của mô hình dữ liệu y khoa):*

```bash
pytest tests/
```
