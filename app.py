import streamlit as st
import os
import sys
import time
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from inference import DrugNERPipeline
from interaction import DrugInteractionChecker
from ocr_engine import MedicalOCREngine

# ==========================================
# ⚙️ APP CONFIG
# ==========================================
st.set_page_config(
    page_title="MedSafe — Medication Safety",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 🎨 PREMIUM HEALTHCARE CSS
# ==========================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,1,0" rel="stylesheet">
<style>
    /* === RESET & GLOBAL === */
    *, *::before, *::after { box-sizing: border-box; }
    .stApp { font-family: 'Inter', -apple-system, sans-serif; }
    
    /* === HIDE STREAMLIT CHROME === */
    #MainMenu, footer, header, .stDeployButton { display: none !important; }
    .block-container { 
        padding: 1rem 1rem 6rem 1rem;
        max-width: 860px;
    }

    /* === SPLASH SCREEN (FULLSCREEN) === */
    .splash-overlay {
        background: linear-gradient(160deg, #064e3b 0%, #065f46 30%, #047857 60%, #059669 100%);
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        z-index: 999999;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 40px;
    }
    .splash-overlay::before {
        content: '';
        position: absolute;
        top: 10%; left: 5%;
        width: 120px; height: 120px;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 50%;
        pointer-events: none;
    }
    .splash-overlay::after {
        content: '';
        position: absolute;
        bottom: 5%; right: 8%;
        width: 80px; height: 80px;
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 12px;
        transform: rotate(30deg);
        pointer-events: none;
    }
    .splash-shield {
        font-size: 4rem;
        margin-bottom: 12px;
        filter: drop-shadow(0 4px 12px rgba(0,0,0,0.2));
    }
    .splash-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0 0 6px 0;
        letter-spacing: -0.5px;
    }
    .splash-sub {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.55);
        margin: 0 0 20px 0;
        font-weight: 400;
    }
    .splash-divider {
        width: 60px;
        height: 2px;
        background: rgba(255,255,255,0.2);
        margin: 0 auto 24px auto;
        border-radius: 2px;
    }
    .splash-features {
        display: flex;
        justify-content: center;
        gap: 16px;
        flex-wrap: wrap;
    }
    .splash-feat {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.1);
        color: rgba(255,255,255,0.8);
        padding: 8px 20px;
        border-radius: 30px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .splash-pills {
        font-size: 2.5rem;
        margin-top: 28px;
    }
    .splash-version {
        color: rgba(255,255,255,0.25);
        font-size: 0.7rem;
        margin-top: 20px;
        letter-spacing: 1px;
    }

    /* === GREETING CARD === */
    .greeting-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #1e1b4b 100%);
        border-radius: 24px;
        padding: 32px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(99,102,241,0.2);
    }
    .greeting-card::after {
        content: '';
        position: absolute;
        top: -60%;
        right: -15%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        border-radius: 50%;
        pointer-events: none;
    }
    .greeting-time {
        font-size: 0.8rem;
        font-weight: 600;
        color: rgba(165,180,252,0.7);
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 6px;
    }
    .greeting-text {
        font-size: 1.8rem;
        font-weight: 800;
        color: #e0e7ff;
        margin: 0 0 4px 0;
        line-height: 1.2;
    }
    .greeting-sub {
        font-size: 0.95rem;
        color: rgba(165,180,252,0.6);
        margin: 0;
        font-weight: 400;
    }
    .greeting-status {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16,185,129,0.12);
        border: 1px solid rgba(16,185,129,0.25);
        color: #34d399;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-top: 14px;
    }

    /* === STAT CARDS === */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 24px;
    }
    .stat-card {
        background: rgba(30,27,75,0.5);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 16px;
        padding: 18px 16px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .stat-card:hover {
        border-color: rgba(99,102,241,0.35);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99,102,241,0.08);
    }
    .stat-icon { font-size: 1.6rem; margin-bottom: 6px; }
    .stat-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #e0e7ff;
        margin: 0;
    }
    .stat-label {
        font-size: 0.72rem;
        color: rgba(165,180,252,0.5);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin: 0;
    }

    /* === SECTION TITLES === */
    .section-title {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 28px 0 16px 0;
    }
    .section-title-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px; height: 36px;
        border-radius: 12px;
        font-size: 1.1rem;
    }
    .section-title-icon.scan { background: rgba(99,102,241,0.15); }
    .section-title-icon.safety { background: rgba(239,68,68,0.12); }
    .section-title-icon.schedule { background: rgba(16,185,129,0.12); }
    .section-title-text {
        font-size: 1.15rem;
        font-weight: 700;
        color: #c7d2fe;
        margin: 0;
    }
    .section-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(99,102,241,0.2), transparent);
    }

    /* === DRUG PILLS === */
    .drug-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.08));
        border: 1px solid rgba(99,102,241,0.2);
        color: #a5b4fc;
        padding: 10px 20px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.88rem;
        margin: 4px;
        transition: all 0.25s ease;
    }
    .drug-pill:hover {
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(139,92,246,0.15));
        transform: scale(1.04);
        box-shadow: 0 4px 16px rgba(99,102,241,0.15);
    }

    /* === ALERT CARDS === */
    .alert-card {
        border-radius: 16px;
        padding: 20px 24px;
        margin: 12px 0;
        border-left: 4px solid;
        transition: all 0.3s ease;
    }
    .alert-card:hover {
        transform: translateX(4px);
    }
    .alert-high {
        background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(185,28,28,0.04));
        border-left-color: #ef4444;
        border: 1px solid rgba(239,68,68,0.15);
        border-left: 4px solid #ef4444;
    }
    .alert-moderate {
        background: linear-gradient(135deg, rgba(245,158,11,0.08), rgba(217,119,6,0.04));
        border-left-color: #f59e0b;
        border: 1px solid rgba(245,158,11,0.15);
        border-left: 4px solid #f59e0b;
    }
    .alert-low {
        background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(37,99,235,0.04));
        border-left-color: #3b82f6;
        border: 1px solid rgba(59,130,246,0.15);
        border-left: 4px solid #3b82f6;
    }
    .alert-title {
        font-size: 0.95rem;
        font-weight: 700;
        margin: 0 0 8px 0;
    }
    .alert-high .alert-title { color: #fca5a5; }
    .alert-moderate .alert-title { color: #fcd34d; }
    .alert-low .alert-title { color: #93c5fd; }
    .alert-body {
        font-size: 0.85rem;
        color: rgba(200,210,230,0.7);
        line-height: 1.6;
        margin: 0;
    }

    /* === TIMELINE SCHEDULE === */
    .timeline-container {
        display: grid;
        gap: 12px;
        margin: 16px 0;
    }
    .timeline-slot {
        background: rgba(30,27,75,0.4);
        border: 1px solid rgba(99,102,241,0.1);
        border-radius: 16px;
        padding: 20px 24px;
        display: flex;
        align-items: flex-start;
        gap: 16px;
        transition: all 0.3s ease;
    }
    .timeline-slot:hover {
        border-color: rgba(99,102,241,0.25);
        background: rgba(30,27,75,0.6);
    }
    .timeline-time {
        min-width: 64px;
        text-align: center;
    }
    .timeline-time-icon { font-size: 1.5rem; margin-bottom: 4px; }
    .timeline-time-label {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .timeline-time-hour {
        font-size: 0.75rem;
        color: rgba(165,180,252,0.5);
        font-weight: 500;
    }
    .slot-morning .timeline-time-label { color: #86efac; }
    .slot-noon .timeline-time-label { color: #93c5fd; }
    .slot-evening .timeline-time-label { color: #c4b5fd; }
    .timeline-divider {
        width: 2px;
        min-height: 40px;
        border-radius: 2px;
        align-self: stretch;
    }
    .slot-morning .timeline-divider { background: linear-gradient(180deg, #86efac, transparent); }
    .slot-noon .timeline-divider { background: linear-gradient(180deg, #93c5fd, transparent); }
    .slot-evening .timeline-divider { background: linear-gradient(180deg, #c4b5fd, transparent); }
    .timeline-content { flex: 1; }
    .timeline-drug {
        font-size: 0.88rem;
        font-weight: 600;
        color: #e0e7ff;
        margin-bottom: 2px;
    }
    .timeline-reason {
        font-size: 0.75rem;
        color: rgba(165,180,252,0.5);
    }

    /* === OCR BOX === */
    .ocr-box {
        background: rgba(30,27,75,0.4);
        border: 1px solid rgba(99,102,241,0.15);
        border-left: 3px solid #818cf8;
        padding: 16px 20px;
        border-radius: 12px;
        font-style: italic;
        color: rgba(165,180,252,0.6);
        line-height: 1.6;
        font-size: 0.9rem;
    }

    /* === SAFE RESULT === */
    .safe-card {
        background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(5,150,105,0.04));
        border: 1px solid rgba(16,185,129,0.2);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    .safe-icon { font-size: 2.5rem; margin-bottom: 8px; }
    .safe-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #6ee7b7;
        margin: 0 0 6px 0;
    }
    .safe-body {
        font-size: 0.85rem;
        color: rgba(110,231,183,0.6);
        margin: 0;
    }

    /* === BOTTOM NAV === */
    .bottom-nav {
        position: fixed;
        bottom: 0; left: 0; right: 0;
        background: rgba(15,12,41,0.96);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-top: 1px solid rgba(99,102,241,0.1);
        padding: 10px 0 14px 0;
        z-index: 99999;
        display: flex;
        justify-content: center;
        gap: 40px;
    }
    .nav-btn {
        text-align: center;
        color: rgba(165,180,252,0.35);
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        cursor: pointer;
        transition: all 0.2s;
    }
    .nav-btn.active { color: #818cf8; }
    .nav-btn:hover { color: #a5b4fc; }
    .nav-btn-icon {
        font-size: 1.25rem;
        display: block;
        margin-bottom: 3px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# ⚙️ BACKEND CONFIG
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models", "ner_model")

llm_base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
llm_model = "gemini-2.5-flash"
api_key_input = os.getenv("GOOGLE_API_KEY", "")

llm_client = None
if api_key_input:
    llm_client = OpenAI(base_url=llm_base_url, api_key=api_key_input)

@st.cache_resource
def load_drugbank():
    """Fast: just CSV data (~1 second)"""
    return DrugInteractionChecker(DATA_DIR)

@st.cache_resource
def load_ner():
    """Heavy: BioBERT model (load only when needed)"""
    return DrugNERPipeline(MODEL_DIR)

@st.cache_resource
def load_ocr():
    """Heavy: EasyOCR engine (load only when needed)"""
    return MedicalOCREngine(use_gpu=False)

# ==========================================
# 🏥 SMART GREETING DASHBOARD
# ==========================================
now = datetime.now()
hour = now.hour
if hour < 12:
    greeting = "Good morning"
    greeting_icon = "🌅"
elif hour < 17:
    greeting = "Good afternoon" 
    greeting_icon = "☀️"
else:
    greeting = "Good evening"
    greeting_icon = "🌙"

date_str = now.strftime("%A, %B %d, %Y")

st.markdown(f"""
<div class="greeting-card">
    <p class="greeting-time">{greeting_icon} {date_str}</p>
    <p class="greeting-text">{greeting}</p>
    <p class="greeting-sub">Let's make sure your medications are safe today.</p>
    <div class="greeting-status">🟢 All systems ready</div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SYSTEM BOOT (fast — only loads DrugBank CSV)
# ==========================================
if not os.path.exists(MODEL_DIR):
    st.error("🚨 AI Model not found. Please place the `ner_model` folder inside `/models/`.")
    st.stop()
if not os.path.exists(os.path.join(DATA_DIR, "drugbank_drugs.csv")):
    st.error("🚨 DrugBank data not found. Check your `/Data/processed` folder.")
    st.stop()

try:
    db_engine = load_drugbank()
except Exception as e:
    st.error(f"Failed to load: {e}")
    st.stop()

# ==========================================
# 🟢 WELCOME SPLASH (only on first open)
# ==========================================
if "app_started" not in st.session_state:
    st.session_state.app_started = False

if not st.session_state.app_started:
    splash = st.empty()
    splash.markdown("""
    <div class="splash-overlay">
        <div class="splash-shield">🛡️</div>
        <p class="splash-title">MedSafe</p>
        <p class="splash-sub">Drug Interaction Checker</p>
        <div class="splash-divider"></div>
        <div class="splash-features">
            <span class="splash-feat">Interaction Check</span>
            <span class="splash-feat">Drug Lookup</span>
            <span class="splash-feat">AI Alerts</span>
        </div>
        <div class="splash-pills">💊💊</div>
        <p class="splash-version">v2.0.0</p>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(3)
    splash.empty()
    st.session_state.app_started = True
    st.rerun()

# ==========================================
# 📋 INPUT SECTION
# ==========================================
st.markdown("""
<div class="section-title">
    <span class="section-title-icon scan">📋</span>
    <p class="section-title-text">Scan Your Prescription</p>
    <span class="section-line"></span>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["✍️ Type or Paste", "📸 Photo / Camera"])

with tabs[0]:
    sample_text = "The 60yo male patient with a history of heart disease is maintained on Nitroglycerin. Last night, the patient secretly consumed Sildenafil before intercourse."
    text_input = st.text_area(
        "prescription_input", value=sample_text, height=120,
        label_visibility="collapsed",
        placeholder="Paste your prescription, medication list, or clinical notes here..."
    )
    if st.button("🔍  Check My Medications", type="primary", use_container_width=True):
        st.session_state.run_analysis = text_input

with tabs[1]:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    with col2:
        camera_file = st.camera_input("Camera", label_visibility="collapsed")
    
    img_to_process = uploaded_file if uploaded_file else camera_file
    if img_to_process:
        st.image(img_to_process, caption="Preview", width=280)
        if st.button("👁️  Scan & Check", type="primary", use_container_width=True):
            with st.spinner("Reading prescription..."):
                time.sleep(1)
                try:
                    ocr_engine = load_ocr()
                    raw_text = ocr_engine.extract_text(img_to_process)
                    st.success("✅ Scanned!")
                    st.markdown(f'<div class="ocr-box">{raw_text}</div>', unsafe_allow_html=True)
                    st.session_state.run_analysis = raw_text
                except Exception as e:
                    st.error(f"Could not read image: {e}")

# ==========================================
# 🧠 AI ANALYSIS (runs once, results saved)
# ==========================================
if "run_analysis" in st.session_state and st.session_state.run_analysis:
    user_text = st.session_state.run_analysis
    
    # Only re-run analysis if text changed
    if st.session_state.get("last_analyzed_text") != user_text:
        with st.spinner("Loading AI model & analyzing..."):
            ner_engine = load_ner()
            extracted_drugs = ner_engine.extract_drugs(user_text)
        
        if not extracted_drugs:
            st.info("No medications detected. Try with a different text.")
            st.session_state.run_analysis = None
        else:
            with st.spinner("Checking safety..."):
                conflicts = db_engine.check_list(extracted_drugs)
            
            # Save results
            st.session_state.last_analyzed_text = user_text
            st.session_state.extracted_drugs = extracted_drugs
            st.session_state.conflicts = conflicts
            st.session_state.schedule_text = None  # will be filled below
            
            # Generate schedule if safe
            if not conflicts and llm_client:
                with st.spinner("Creating your schedule..."):
                    drugs_str = ", ".join(extracted_drugs)
                    sched_sys = """You are a clinical pharmacology scheduling AI. Output ONLY a valid JSON object. No greetings, warnings, disclaimers, markdown, or code fences. Just pure JSON."""
                    sched_usr = f"""Medications: {drugs_str}
Prescription text: "{user_text}"

INSTRUCTIONS:
1. FIRST check the prescription text for ANY timing info from the doctor (e.g. "twice daily", "after meals", "morning", "before bed", "5mg daily").
2. If the doctor specified timing → follow it exactly. Set reason to "(Doctor's order)".
3. If the doctor did NOT specify timing → use clinical pharmacology knowledge:
   - Anticoagulants (Warfarin, Heparin) → Evening
   - Blood pressure meds (ACE inhibitors, ARBs) → Morning
   - Statins (Atorvastatin, Rosuvastatin) → Evening
   - NSAIDs/Pain (Aspirin, Ibuprofen) → Noon (after meals)
   - Sedatives, sleep aids → Evening
   - Antibiotics → Morning + Evening (evenly spaced)
   - Vitamins → Morning
   Set reason to "(Clinical recommendation: [brief reason])".
4. EVERY drug MUST be assigned. ONLY include time slots that have drugs.

OUTPUT: Return ONLY this JSON (no markdown, no code block, no extra text):
{{"Morning": [{{"drug": "DrugName Dose", "reason": "reason text"}}], "Noon": [...], "Evening": [...]}}

If a time slot has no drugs, omit it entirely from the JSON. Example with only Morning:
{{"Morning": [{{"drug": "Aspirin 81mg", "reason": "(Doctor's order)"}}]}}"""
                    try:
                        sched_resp = llm_client.chat.completions.create(
                            model=llm_model,
                            messages=[{"role":"system","content":sched_sys},{"role":"user","content":sched_usr}],
                            max_tokens=1024
                        )
                        raw = sched_resp.choices[0].message.content.strip()
                        # Clean markdown code fences if AI adds them
                        if raw.startswith("```"):
                            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                        if raw.endswith("```"):
                            raw = raw[:-3].strip()
                        if raw.startswith("json"):
                            raw = raw[4:].strip()
                        
                        import json
                        st.session_state.schedule_data = json.loads(raw)
                        st.session_state.schedule_error = None
                    except json.JSONDecodeError:
                        # Fallback: store raw text as error
                        st.session_state.schedule_data = None
                        st.session_state.schedule_error = f"Could not parse schedule. Raw: {raw[:200]}"
                    except Exception as e:
                        st.session_state.schedule_data = None
                        st.session_state.schedule_error = str(e)
    
    # ==========================================
    # 📊 DISPLAY RESULTS (from session_state)
    # ==========================================
    if "extracted_drugs" in st.session_state:
        extracted_drugs = st.session_state.extracted_drugs
        conflicts = st.session_state.conflicts
        num_drugs = len(extracted_drugs)
        num_alerts = len(conflicts)
        status_text = "Safe" if num_alerts == 0 else f"{num_alerts} Alert{'s' if num_alerts > 1 else ''}"
        status_color = "#34d399" if num_alerts == 0 else "#f87171"
        
        # === QUICK STATS ===
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-icon">💊</div>
                <p class="stat-value">{num_drugs}</p>
                <p class="stat-label">Medications</p>
            </div>
            <div class="stat-card">
                <div class="stat-icon">⚠️</div>
                <p class="stat-value" style="color: {status_color}">{num_alerts}</p>
                <p class="stat-label">Interactions</p>
            </div>
            <div class="stat-card">
                <div class="stat-icon">{"✅" if num_alerts == 0 else "🔴"}</div>
                <p class="stat-value" style="color: {status_color}; font-size: 1.1rem;">{status_text}</p>
                <p class="stat-label">Status</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # === DETECTED DRUGS ===
        st.markdown("""
        <div class="section-title">
            <span class="section-title-icon scan">💊</span>
            <p class="section-title-text">Medications Found</p>
            <span class="section-line"></span>
        </div>
        """, unsafe_allow_html=True)
        
        pills_html = "".join([f'<span class="drug-pill">💊 {d.title()}</span>' for d in extracted_drugs])
        st.markdown(pills_html, unsafe_allow_html=True)

        # === SAFETY CHECK ===
        st.markdown("""
        <div class="section-title">
            <span class="section-title-icon safety">🛡️</span>
            <p class="section-title-text">Safety Check</p>
            <span class="section-line"></span>
        </div>
        """, unsafe_allow_html=True)
        
        if conflicts:
            for idx, c in enumerate(conflicts):
                severity = c.get('severity', 'Moderate')
                if severity == 'High':
                    cls, icon = "alert-high", "🔥"
                elif severity == 'Moderate':
                    cls, icon = "alert-moderate", "⚠️"
                else:
                    cls, icon = "alert-low", "ℹ️"
                
                with st.expander(f"{icon} {c['drug1']}  ⚡  {c['drug2']} — {severity.upper()}", expanded=True):
                    st.markdown(f"""
                    <div class="{cls}">
                        <p class="alert-title">{icon} {severity} Risk: {c['drug1']} + {c['drug2']}</p>
                        <p class="alert-body"><strong>What happens:</strong> {c['description']}</p>
                        <p class="alert-body" style="margin-top:8px"><strong>Alternatives:</strong> Ask your doctor about <em>{c['drug1_alternatives'][0]}</em> or <em>{c['drug2_alternatives'][0]}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    verified = st.checkbox("✅ I have reviewed this", key=f"v_{idx}")
                    if verified:
                        st.success("Acknowledged ✓")
                    
                    # LLM Harms
                    if llm_client:
                        sys_ctx = """You are a strict Pharmacovigilance AI. ONLY list specific clinical harms as bullet points. No mechanisms, no greetings, no disclaimers. Be concise."""
                        usr_ctx = f"[Context]: {c['description']}\n\nList ONLY the clinical harms of combining {c['drug1']} with {c['drug2']}. Bullet points."
                        try:
                            resp = llm_client.chat.completions.create(
                                model=llm_model,
                                messages=[{"role":"system","content":sys_ctx},{"role":"user","content":usr_ctx}],
                                max_tokens=400
                            )
                            st.markdown("**⚠️ Clinical Harms:**")
                            st.warning(resp.choices[0].message.content)
                        except Exception as e:
                            st.error(f"Error: {e}")
        else:
            st.balloons()
            st.markdown("""
            <div class="safe-card">
                <div class="safe-icon">✅</div>
                <p class="safe-title">All Clear!</p>
                <p class="safe-body">No dangerous interactions found between your medications.<br>You're safe to follow your prescription as directed.</p>
            </div>
            """, unsafe_allow_html=True)

        # === MEDICATION SCHEDULE ===
        st.markdown("""
        <div class="section-title">
            <span class="section-title-icon schedule">🗓️</span>
            <p class="section-title-text">Your Daily Schedule</p>
            <span class="section-line"></span>
        </div>
        """, unsafe_allow_html=True)
        
        if not llm_client:
            st.warning("Schedule unavailable — API key not configured.")
        elif num_alerts > 0:
            st.error("🚫 **Can't set schedule** — Dangerous drug interactions detected above. Please consult your doctor before taking these medications together.")
        else:
            schedule_data = st.session_state.get("schedule_data")
            schedule_err = st.session_state.get("schedule_error")
            
            if schedule_err:
                st.error(f"Schedule error: {schedule_err}")
            elif schedule_data:
                # --- Render Timeline Cards (matching mockup) ---
                slot_config = {
                    "Morning": {"icon": "🌅", "time": "8:00", "css": "slot-morning", "pill_bg": "rgba(16,185,129,0.25)", "pill_border": "rgba(16,185,129,0.5)"},
                    "Noon":    {"icon": "☀️", "time": "12:00", "css": "slot-noon", "pill_bg": "rgba(59,130,246,0.25)", "pill_border": "rgba(59,130,246,0.5)"},
                    "Evening": {"icon": "🌙", "time": "19:00", "css": "slot-evening", "pill_bg": "rgba(139,92,246,0.25)", "pill_border": "rgba(139,92,246,0.5)"},
                }
                
                active_slots_for_check = []
                
                st.markdown('<div class="timeline-container">', unsafe_allow_html=True)
                
                for slot_name, cfg in slot_config.items():
                    drugs_in_slot = schedule_data.get(slot_name, [])
                    if not drugs_in_slot:
                        continue
                    
                    active_slots_for_check.append(slot_name)
                    
                    # Build drug pills HTML
                    drugs_html = ""
                    for drug_info in drugs_in_slot:
                        d_name = drug_info.get("drug", "Unknown")
                        d_reason = drug_info.get("reason", "")
                        drugs_html += f"""
                        <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                            <span style="display:inline-block;width:28px;height:14px;border-radius:10px;background:{cfg['pill_bg']};border:1.5px solid {cfg['pill_border']};"></span>
                            <span class="timeline-drug">{d_name}</span>
                        </div>
                        <div class="timeline-reason" style="margin-left:38px;margin-bottom:8px;">{d_reason}</div>
                        """
                    
                    st.markdown(f"""
                    <div class="timeline-slot {cfg['css']}">
                        <div class="timeline-time">
                            <div class="timeline-time-icon">{cfg['icon']}</div>
                            <div class="timeline-time-label">{slot_name}</div>
                            <div class="timeline-time-hour">{cfg['time']}</div>
                        </div>
                        <div class="timeline-divider"></div>
                        <div class="timeline-content">
                            {drugs_html}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # --- Mark as Taken checkboxes ---
                if active_slots_for_check:
                    st.markdown("---")
                    st.markdown("**Mark as taken:**")
                    cols = st.columns(len(active_slots_for_check))
                    all_done = True
                    slot_labels = {"Morning": "🌅 Morning", "Noon": "☀️ Noon", "Evening": "🌙 Evening"}
                    for i, slot_name in enumerate(active_slots_for_check):
                        with cols[i]:
                            checked = st.checkbox(f"{slot_labels[slot_name]} done", key=f"{slot_name.lower()}_done")
                            if not checked:
                                all_done = False
                    
                    if all_done:
                        st.success("✅ All medications taken! Great job staying on track!")

# ==========================================
# 📱 BOTTOM NAVIGATION
# ==========================================
st.markdown("""
<div class="bottom-nav">
    <div class="nav-btn active">
        <span class="nav-btn-icon">🏠</span>
        Home
    </div>
    <div class="nav-btn">
        <span class="nav-btn-icon">📷</span>
        Scan
    </div>
    <div class="nav-btn">
        <span class="nav-btn-icon">🛡️</span>
        Safety
    </div>
    <div class="nav-btn">
        <span class="nav-btn-icon">🗓️</span>
        Schedule
    </div>
    <div class="nav-btn">
        <span class="nav-btn-icon">👤</span>
        Profile
    </div>
</div>
""", unsafe_allow_html=True)
