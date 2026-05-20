#!/usr/bin/env python
# coding: utf-8

# # DrugBank Parser Setup
# 
# This notebook installs necessary dependencies and runs the `parse_drugbank.py` script to generate dataset CSVs.

# In[8]:


pip install streamlit


# In[9]:


"""
Parse DrugBank XML to CSV
"""
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# UPDATED PATHS
INPUT_XML = "/Users/lydang/Downloads/ML Project/Data/full database.xml"
OUTPUT_DIR = "/Users/lydang/Downloads/ML Project/Data/processed"

def parse_drugbank(xml_path):
    """Parse DrugBank XML"""

    ns = {'db': 'http://www.drugbank.ca'}

    print(f"Loading: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    drugs = []
    interactions = []

    for drug in tqdm(root.findall('db:drug', ns)):
        id_elem = drug.find('db:drugbank-id[@primary="true"]', ns)
        if id_elem is None:
            continue

        drug_id = id_elem.text
        name_elem = drug.find('db:name', ns)
        name = name_elem.text if name_elem is not None else ""
        desc_elem = drug.find('db:description', ns)
        desc = desc_elem.text if desc_elem is not None else ""

        all_names = [name]
        syn_elem = drug.find('db:synonyms', ns)
        if syn_elem is not None:
            for syn in syn_elem.findall('db:synonym', ns):
                if syn.text:
                    all_names.append(syn.text.strip())

        drugs.append({
            'drugbank_id': drug_id,
            'name': name,
            'description': desc[:500] if desc else "",
            'all_names': '|'.join(list(set(all_names))[:20])
        })

        int_elem = drug.find('db:drug-interactions', ns)
        if int_elem is not None:
            for interaction in int_elem.findall('db:drug-interaction', ns):
                target_id = interaction.find('db:drugbank-id', ns)
                target_name = interaction.find('db:name', ns)
                desc = interaction.find('db:description', ns)

                if target_id is not None and desc is not None:
                    interactions.append({
                        'drug_1_id': drug_id,
                        'drug_1_name': name,
                        'drug_2_id': target_id.text,
                        'drug_2_name': target_name.text if target_name is not None else "",
                        'description': desc.text
                    })

    return pd.DataFrame(drugs), pd.DataFrame(interactions)

def main():
    print("="*60)
    print("DRUGBANK PARSER")
    print("="*60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    drugs_df, interactions_df = parse_drugbank(INPUT_XML)

    drugs_df.to_csv(f"{OUTPUT_DIR}/drugbank_drugs.csv", index=False)
    interactions_df.to_csv(f"{OUTPUT_DIR}/drugbank_interactions.csv", index=False)

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Drugs:        {len(drugs_df):,}")
    print(f"Interactions: {len(interactions_df):,}")
    print("="*60)

if __name__ == "__main__":
    main()


# In[1]:


import pandas as pd
import sys
import os
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

# ==========================================
# LỚP XỬ LÝ DỮ LIỆU TƯƠNG TÁC THUỐC (Từ Code 2)
# Khai báo trực tiếp trong Notebook để không bị lỗi Import
# ==========================================
class DrugInteractionChecker:
    def __init__(self, drugs_path, interactions_path):
        print("📂 Đang tải dữ liệu, vui lòng đợi vài giây...")
        self.drugs = pd.read_csv(drugs_path)
        self.interactions = pd.read_csv(interactions_path)
        self._build_lookups()

        print(f"✅ Đã tải {len(self.drugs):,} loại thuốc")
        print(f"✅ Đã tải {len(self.interactions):,} tương tác")

    def _build_lookups(self):
        self.name_to_id = {}
        for _, row in self.drugs.iterrows():
            if pd.isna(row['all_names']):
                continue
            for name in str(row['all_names']).split('|'):
                name_clean = name.lower().strip()
                if name_clean:
                    self.name_to_id[name_clean] = row['drugbank_id']

        self.interaction_map = {}
        for _, row in self.interactions.iterrows():
            if pd.isna(row['drug_1_id']) or pd.isna(row['drug_2_id']):
                continue
            key = tuple(sorted([str(row['drug_1_id']), str(row['drug_2_id'])]))
            if key not in self.interaction_map:
                self.interaction_map[key] = str(row['description'])

    def find_drug(self, drug_name):
        drug_name = drug_name.lower().strip()
        if drug_name in self.name_to_id:
            drug_id = self.name_to_id[drug_name]
            drug_info = self.drugs[self.drugs['drugbank_id'] == drug_id].iloc[0]

            # Xử lý lỗi NaN nếu không có mô tả
            desc = drug_info['description']
            if pd.isna(desc):
                desc = "Không có thông tin mô tả."

            return {
                'found': True,
                'drugbank_id': drug_id,
                'name': drug_info['name'],
                'description': desc
            }
        return {'found': False}

    def check_interaction(self, drug_name_1, drug_name_2):
        drug_1 = self.find_drug(drug_name_1)
        drug_2 = self.find_drug(drug_name_2)

        if not drug_1['found']:
            return {'error': f"Không tìm thấy thuốc: {drug_name_1}"}
        if not drug_2['found']:
            return {'error': f"Không tìm thấy thuốc: {drug_name_2}"}

        key = tuple(sorted([drug_1['drugbank_id'], drug_2['drugbank_id']]))

        if key in self.interaction_map:
            return {
                'has_interaction': True,
                'drug_1': drug_1['name'],
                'drug_2': drug_2['name'],
                'description': self.interaction_map[key]
            }
        else:
            return {
                'has_interaction': False,
                'drug_1': drug_1['name'],
                'drug_2': drug_2['name']
            }

    def check_multiple(self, drug_names):
        drugs = []
        not_found = []

        for name in drug_names:
            result = self.find_drug(name)
            if result['found']:
                drugs.append(result)
            else:
                not_found.append(name)

        if not_found:
            return {'error': f"Không tìm thấy các thuốc: {', '.join(not_found)}"}

        interactions = []
        for i, drug_1 in enumerate(drugs):
            for drug_2 in drugs[i+1:]:
                key = tuple(sorted([drug_1['drugbank_id'], drug_2['drugbank_id']]))
                if key in self.interaction_map:
                    interactions.append({
                        'drug_1': drug_1['name'],
                        'drug_2': drug_2['name'],
                        'description': self.interaction_map[key]
                    })

        return {
            'drugs_checked': [d['name'] for d in drugs],
            'interactions_found': len(interactions),
            'interactions': interactions
        }

# ==========================================
# KHỞI TẠO ỨNG DỤNG VÀ DỮ LIỆU
# ==========================================
DRUGS_PATH = "/Users/lydang/Downloads/ML Project/Data/processed/drugbank_drugs.csv"
INTERACTIONS_PATH = "/Users/lydang/Downloads/ML Project/Data/processed/drugbank_interactions.csv"

# Khai báo global checker để chỉ load dữ liệu 1 lần (tránh Jupyter bị đơ khi chạy lại Cell nhiều lần)
if 'checker' not in globals():
    checker = None

def load_data():
    global checker
    if checker is None:
        if os.path.exists(DRUGS_PATH) and os.path.exists(INTERACTIONS_PATH):
            checker = DrugInteractionChecker(
                drugs_path=DRUGS_PATH,
                interactions_path=INTERACTIONS_PATH
            )
            print("✅ Hệ thống lưu trữ sẵn sàng!\n")
        else:
            print(f"❌ KHÔNG TÌM THẤY DỮ LIỆU TẠI:\n- {DRUGS_PATH}\n- {INTERACTIONS_PATH}")
            print("💡 Bạn nhớ phải chạy thành công Code 1 (xử lý XML sang CSV) trước khi chạy Code này nhé!")
            return False
    return True

print("="*40)
print("🚀 KHỞI ĐỘNG CÔNG CỤ TRA CỨU THUỐC")
print("="*40)

is_loaded = False
try:
    is_loaded = load_data()
except Exception as e:
    print(f"❌ Lỗi khi tải dữ liệu: {e}")

# ==========================================
# GIAO DIỆN (CHỈ HIỂN THỊ KHI LOAD DATA THÀNH CÔNG)
# ==========================================
if is_loaded and checker is not None:
    # ------------------- GIAO DIỆN 1: KIỂM TRA 2 LOẠI THUỐC -------------------
    display(HTML("<h3>🔍 Kiểm Tra Tương Tác Giữa 2 Loại Thuốc</h3>"))

    drug1_input = widgets.Text(description='Thuốc 1:', placeholder='VD: Aspirin')
    drug2_input = widgets.Text(description='Thuốc 2:', placeholder='VD: Warfarin')
    check_btn = widgets.Button(description='🔍 Kiểm Tra', button_style='primary')
    out_pair = widgets.Output()

    def on_check_pair_clicked(b):
        with out_pair:
            clear_output(wait=True)
            d1, d2 = drug1_input.value.strip(), drug2_input.value.strip()
            if not d1 or not d2:
                display(HTML("<b style='color:orange;'>⚠️ Vui lòng nhập cả hai loại thuốc.</b>"))
                return

            result = checker.check_interaction(d1, d2)
            clear_output()

            if 'error' in result:
                display(HTML(f"<b style='color:red;'>❌ Lỗi: {result['error']}</b>"))
            elif result['has_interaction']:
                display(HTML("<h4 style='color:red;'>⚠️ PHÁT HIỆN TƯƠNG TÁC</h4>"))
                display(HTML(f"<b>{result['drug_1']}</b> ↔️ <b>{result['drug_2']}</b>"))
                display(HTML(f"<div style='background-color:#ffe6e6; padding:10px; border-left: 5px solid red;'>"
                             f"{result['description']}</div>"))
            else:
                display(HTML(f"<b style='color:green;'>✅ Không tìm thấy tương tác đã biết giữa {result['drug_1']} và {result['drug_2']}</b>"))

    check_btn.on_click(on_check_pair_clicked)
    display(widgets.HBox([drug1_input, drug2_input]), check_btn, out_pair)
    display(HTML("<hr>"))

    # ------------------- GIAO DIỆN 2: KIỂM TRA NHIỀU THUỐC CÙNG LÚC -------------------
    display(HTML("<h3>📋 Kiểm Tra Danh Sách Thuốc</h3>"))

    drugs_area = widgets.Textarea(
        placeholder='Nhập tên thuốc, mỗi dòng 1 thuốc.\n\nVD:\nAspirin\nWarfarin\nAcetaminophen\nMetformin',
        description='Danh sách:',
        layout={'height': '150px', 'width': '350px'}
    )
    check_multi_btn = widgets.Button(description='🔍 Kiểm Tra Tất Cả', button_style='warning')
    out_multi = widgets.Output()

    def on_check_multi_clicked(b):
        with out_multi:
            clear_output(wait=True)
            text_val = drugs_area.value
            drugs = [d.strip() for d in text_val.split('\n') if d.strip()]

            if len(drugs) < 2:
                display(HTML("<b style='color:orange;'>⚠️ Vui lòng nhập ít nhất 2 loại thuốc.</b>"))
                return

            result = checker.check_multiple(drugs)
            clear_output()

            if 'error' in result:
                display(HTML(f"<b style='color:red;'>❌ {result['error']}</b>"))
            else:
                display(HTML(f"<span style='color:green;'>✅ Đã kiểm tra: <b>{', '.join(result['drugs_checked'])}</b></span>"))

                if result['interactions_found'] > 0:
                    display(HTML(f"<h4 style='color:red;'>⚠️ Tìm thấy {result['interactions_found']} tương tác:</h4>"))
                    for idx, interaction in enumerate(result['interactions'], 1):
                        display(HTML(f"<details><summary><b>{idx}. {interaction['drug_1']} ↔️ {interaction['drug_2']}</b></summary>"
                                     f"<div style='padding:10px; background-color:#fff3cd; border-left:4px solid orange; margin-bottom:5px;'>"
                                     f"{interaction['description']}</div></details>"))
                else:
                    display(HTML("<b style='color:green;'>✅ Tuyệt vời! Không tìm thấy tương tác nào giữa các thuốc này.</b>"))

    check_multi_btn.on_click(on_check_multi_clicked)
    display(widgets.HBox([drugs_area, check_multi_btn]), out_multi)
    display(HTML("<hr>"))

    # ------------------- GIAO DIỆN 3: TÌM KIẾM THUỐC -------------------
    display(HTML("<h3>🔎 Tra Cứu Thông Tin Thuốc</h3>"))

    search_input = widgets.Text(description='Tên thuốc:', placeholder='VD: Aspirin')
    search_btn = widgets.Button(description='🔎 Cứu', button_style='success')
    out_search = widgets.Output()

    def on_search_clicked(b):
        with out_search:
            clear_output(wait=True)
            term = search_input.value.strip()
            if not term:
                return

            result = checker.find_drug(term)
            if result['found']:
                display(HTML(f"<b style='color:green;'>✅ {result['name']} (ID: {result['drugbank_id']})</b>"))
                if result['description']:
                    display(HTML(f"<details open><summary>📖 <b>Mô tả:</b></summary>"
                                 f"<div style='padding:10px; background-color:#f8f9fa; border-left:3px solid #ccc;'>"
                                 f"{result['description']}</div></details>"))
            else:
                display(HTML(f"<b style='color:red;'>❌ Không tìm thấy thuốc: {term}</b><br>"
                             f"<small style='color:gray;'>💡 Mẹo: Hãy thử nhập tên chung (generic) của thuốc.</small>"))

    search_btn.on_click(on_search_clicked)
    display(widgets.HBox([search_input, search_btn]), out_search)


# ADVANCED ANALYTICS

# 2. Chart and DRUG RECOMMENDATION SYSTEM

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
DRUGS_PATH = "/Users/lydang/Downloads/ML Project/Data/processed/drugbank_drugs.csv"
INTERACTIONS_PATH = "/Users/lydang/Downloads/ML Project/Data/processed/drugbank_interactions.csv"

# ==========================================
# 1. LỚP XỬ LÝ DỮ LIỆU TƯƠNG TÁC THUỐC
# ==========================================
class DrugInteractionChecker:
    def __init__(self, drugs_path, interactions_path):
        print("📂 Đang tải dữ liệu...")
        self.drugs = pd.read_csv(drugs_path)
        self.interactions = pd.read_csv(interactions_path)
        self._build_lookups()
        self._build_interaction_lookup()

    def _build_lookups(self):
        self.name_to_id = {}
        for _, row in self.drugs.iterrows():
            if pd.isna(row['all_names']):
                continue
            for name in str(row['all_names']).split('|'):
                name_clean = name.lower().strip()
                if name_clean:
                    self.name_to_id[name_clean] = row['drugbank_id']

    def _build_interaction_lookup(self):
        self.interaction_map = {}
        for _, row in self.interactions.iterrows():
             if pd.isna(row['drug_1_id']) or pd.isna(row['drug_2_id']):
                 continue
             key = tuple(sorted([str(row['drug_1_id']), str(row['drug_2_id'])]))
             self.interaction_map[key] = str(row['description'])

    def find_drug(self, drug_name):
        drug_name = drug_name.lower().strip()
        if drug_name in self.name_to_id:
            drug_id = self.name_to_id[drug_name]
            drug_info = self.drugs[self.drugs['drugbank_id'] == drug_id].iloc[0]
            desc = drug_info['description'] if not pd.isna(drug_info['description']) else ""
            return {
                'found': True,
                'drugbank_id': drug_id,
                'name': drug_info['name'],
                'description': desc
            }
        return {'found': False}

    def check_interaction(self, drug_name_1, drug_name_2):
        drug_1 = self.find_drug(drug_name_1)
        drug_2 = self.find_drug(drug_name_2)

        if not drug_1['found'] or not drug_2['found']:
            return {'has_interaction': False}

        key = tuple(sorted([drug_1['drugbank_id'], drug_2['drugbank_id']]))

        if key in self.interaction_map:
            return {
                'has_interaction': True,
                'drug_1': drug_1['name'],
                'drug_2': drug_2['name'],
                'description': self.interaction_map[key]
            }
        return {'has_interaction': False}

# ==========================================
# 2. LỚP GỢI Ý THUỐC THAY THẾ
# ==========================================
class DrugRecommender:
    def __init__(self, checker):
        self.checker = checker
        self.drugs = checker.drugs
        self.interactions = checker.interactions

    def find_alternatives(self, problematic_drug, safe_drugs):
        drug_info = self.checker.find_drug(problematic_drug)
        if not drug_info['found']:
            return {'error': f'Drug not found: {problematic_drug}'}

        desc = str(drug_info['description']).lower()
        keywords = set(desc.split()[:20])

        candidates = []
        for _, row in self.drugs.iterrows():
            if row['name'] == drug_info['name']:
                continue

            candidate_desc = str(row['description']).lower() if not pd.isna(row['description']) else ""
            candidate_keywords = set(candidate_desc.split()[:20])

            similarity = len(keywords & candidate_keywords) / max(len(keywords), 1)

            if similarity > 0.2:
                candidates.append({
                    'name': row['name'],
                    'id': row['drugbank_id'],
                    'similarity': similarity
                })

        safe_alternatives = []
        for candidate in sorted(candidates, key=lambda x: x['similarity'], reverse=True)[:20]:
            has_interaction = False
            for safe_drug in safe_drugs:
                result = self.checker.check_interaction(candidate['name'], safe_drug)
                if result.get('has_interaction'):
                    has_interaction = True
                    break

            if not has_interaction:
                safe_alternatives.append(candidate)

        return {
            'original_drug': drug_info['name'],
            'safe_alternatives': safe_alternatives[:5]
        }

# ==========================================
# 3. CHẠY PHÂN TÍCH VÀ GỢI Ý
# ==========================================

# Khởi tạo (tránh load data lại nếu đã có)
if 'checker' not in globals() or checker is None:
    checker = DrugInteractionChecker(DRUGS_PATH, INTERACTIONS_PATH)

recommender = DrugRecommender(checker)

print("="*60)
print("📊 DRUGBANK ANALYTICS (TỪ CODE 3)")
print("="*60)

# 1. Các thuốc có nhiều tương tác nhất
print("\n🔝 TOP 10 DRUGS WITH MOST INTERACTIONS:")
top_drugs = checker.interactions['drug_1_name'].value_counts().head(10)
for i, (drug, count) in enumerate(top_drugs.items(), 1):
    print(f"{i:2d}. {drug:35s} {count:,} tương tác")

# 2. Từ khoá tương tác phổ biến
print("\n💊 COMMON INTERACTION TYPES:")
all_descriptions = ' '.join(checker.interactions['description'].astype(str).values)
keywords = ['increase', 'decrease', 'reduce', 'enhance', 'inhibit', 'risk', 'toxicity', 'bleeding']

for keyword in keywords:
    count = all_descriptions.lower().count(keyword)
    print(f"   {keyword:15s}: {count:,} mentions")


print("\n" + "="*60)
print("💊 DRUG RECOMMENDATION SYSTEM (TỪ CODE 4)")
print("="*60)

print("\n🔍 Scenario: Patient on Warfarin needs pain relief")
print("   Problem: Aspirin interacts with Warfarin")
print("💡 Finding alternatives to Aspirin that don't interact with Warfarin...\n")

result = recommender.find_alternatives(
    problematic_drug="Aspirin",
    safe_drugs=["Warfarin"]
)

if 'error' not in result:
    print(f"✅ Found {len(result['safe_alternatives'])} safe alternatives:")
    for i, alt in enumerate(result['safe_alternatives'], 1):
        print(f"   {i}. {alt['name']} (similarity: {alt['similarity']:.2%})")


# 3. RISK SCORING SYSTEM

# In[6]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Cho phép biểu đồ hiển thị trực tiếp trên Jupyter Notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
DRUGS_PATH = "/Users/lydang/Downloads/ML Project/Data/processed/drugbank_drugs.csv"
INTERACTIONS_PATH = "/Users/lydang/Downloads/ML Project/Data/processed/drugbank_interactions.csv"

# ==========================================
# 1. LỚP XỬ LÝ DỮ LIỆU TƯƠNG TÁC THUỐC (Nền tảng)
# ==========================================
class DrugInteractionChecker:
    def __init__(self, drugs_path, interactions_path):
        print("📂 Đang tải dữ liệu...")
        self.drugs = pd.read_csv(drugs_path)
        self.interactions = pd.read_csv(interactions_path)
        self._build_lookups()
        self._build_interaction_lookup()

    def _build_lookups(self):
        self.name_to_id = {}
        for _, row in self.drugs.iterrows():
            if pd.isna(row['all_names']):
                continue
            for name in str(row['all_names']).split('|'):
                name_clean = name.lower().strip()
                if name_clean:
                    self.name_to_id[name_clean] = row['drugbank_id']

    def _build_interaction_lookup(self):
        self.interaction_map = {}
        for _, row in self.interactions.iterrows():
             if pd.isna(row['drug_1_id']) or pd.isna(row['drug_2_id']):
                 continue
             key = tuple(sorted([str(row['drug_1_id']), str(row['drug_2_id'])]))
             self.interaction_map[key] = str(row['description'])

    def find_drug(self, drug_name):
        drug_name = drug_name.lower().strip()
        if drug_name in self.name_to_id:
            drug_id = self.name_to_id[drug_name]
            drug_info = self.drugs[self.drugs['drugbank_id'] == drug_id].iloc[0]
            desc = drug_info['description'] if not pd.isna(drug_info['description']) else ""
            return {
                'found': True,
                'drugbank_id': drug_id,
                'name': drug_info['name'],
                'description': desc
            }
        return {'found': False}

    def check_interaction(self, drug_name_1, drug_name_2):
        drug_1 = self.find_drug(drug_name_1)
        drug_2 = self.find_drug(drug_name_2)

        if not drug_1['found'] or not drug_2['found']:
            return {'has_interaction': False}

        key = tuple(sorted([drug_1['drugbank_id'], drug_2['drugbank_id']]))

        if key in self.interaction_map:
            return {
                'has_interaction': True,
                'drug_1': drug_1['name'],
                'drug_2': drug_2['name'],
                'description': self.interaction_map[key]
            }
        return {'has_interaction': False}

    def check_multiple(self, drug_names):
        drugs = []
        not_found = []

        for name in drug_names:
            result = self.find_drug(name)
            if result['found']:
                drugs.append(result)
            else:
                not_found.append(name)

        if not_found:
            return {'error': f"Không tìm thấy các thuốc: {', '.join(not_found)}"}

        interactions = []
        for i, drug_1 in enumerate(drugs):
            for drug_2 in drugs[i+1:]:
                key = tuple(sorted([drug_1['drugbank_id'], drug_2['drugbank_id']]))
                if key in self.interaction_map:
                    interactions.append({
                        'drug_1': drug_1['name'],
                        'drug_2': drug_2['name'],
                        'description': self.interaction_map[key]
                    })

        return {
            'drugs_checked': [d['name'] for d in drugs],
            'interactions_found': len(interactions),
            'interactions': interactions
        }

# ==========================================
# 2. LỚP ĐÁNH GIÁ RỦI RO (Risk Scorer)
# ==========================================
class DrugRiskScorer:
    def __init__(self, checker):
        self.checker = checker

        # Keywords for severity classification
        self.severe_keywords = ['severe', 'fatal', 'death', 'life-threatening', 'contraindicated']
        self.moderate_keywords = ['increase', 'enhance', 'risk', 'toxic']
        self.mild_keywords = ['may', 'possible', 'minor', 'slight']

    def classify_severity(self, description):
        """Classify interaction severity"""
        desc_lower = str(description).lower()

        for keyword in self.severe_keywords:
            if keyword in desc_lower:
                return 'SEVERE', 10

        for keyword in self.moderate_keywords:
            if keyword in desc_lower:
                return 'MODERATE', 5

        return 'MILD', 2

    def calculate_risk_score(self, drug_names):
        """Calculate overall risk score for drug combination"""
        result = self.checker.check_multiple(drug_names)

        if 'error' in result:
            return result

        total_score = 0
        severity_breakdown = {'SEVERE': 0, 'MODERATE': 0, 'MILD': 0}

        for interaction in result['interactions']:
            severity, score = self.classify_severity(interaction['description'])
            total_score += score
            severity_breakdown[severity] += 1

        # Normalize score (0-100)
        max_possible_score = len(result['interactions']) * 10
        normalized_score = (total_score / max_possible_score * 100) if max_possible_score > 0 else 0

        return {
            'drugs': result['drugs_checked'],
            'total_interactions': result['interactions_found'],
            'risk_score': normalized_score,
            'severity_breakdown': severity_breakdown,
            'recommendation': self._get_recommendation(normalized_score, severity_breakdown)
        }

    def _get_recommendation(self, score, breakdown):
        """Provide recommendation based on risk score"""
        if breakdown['SEVERE'] > 0:
            return "⛔ CONTRAINDICATED (CHỐNG CHỈ ĐỊNH) - Cần hỏi ý kiến bác sĩ ngay lập tức"
        elif score > 70:
            return "🔴 HIGH RISK (RỦI RO CAO) - Cần có sự giám sát y tế"
        elif score > 40:
            return "🟡 MODERATE RISK (RỦI RO VỪA) - Khuyên dùng thận trọng"
        else:
            return "🟢 LOW RISK (RỦI RO THẤP) - Nhìn chung an toàn, nhưng vẫn cần theo dõi tác dụng phụ"

# ==========================================
# KHỞI TẠO (TRÁNH LOAD LẠI NHIỀU LẦN)
# ==========================================
if 'checker' not in globals() or checker is None:
    if os.path.exists(DRUGS_PATH) and os.path.exists(INTERACTIONS_PATH):
        checker = DrugInteractionChecker(DRUGS_PATH, INTERACTIONS_PATH)
    else:
        print("❌ LỖI: KHÔNG TÌM THẤY DỮ LIỆU CSV! Hãy chạy Code 1 (XML -> CSV) trước.")

# ==========================================
# 3. CHẠY TEST TỔNG HỢP: BIỂU ĐỒ ANALYTICS + TÍNH RỦI RO
# ==========================================
if 'checker' in globals() and checker is not None:

    # ------------------ PHẦN PHÂN TÍCH VÀ VẼ BIỂU ĐỒ ------------------
    print("\n" + "="*60)
    print("📊 PHẦN 1: DRUGBANK ANALYTICS (Đã sửa lỗi hiển thị)")
    print("="*60)

    # Chuẩn bị dữ liệu vẽ biểu đồ
    interactions = checker.interactions
    all_descriptions = ' '.join(interactions['description'].astype(str).values)
    keywords = ['increase', 'decrease', 'reduce', 'enhance', 'inhibit', 'risk', 'toxicity', 'bleeding']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    sns.set_theme(style="whitegrid") # Thêm theme cho đẹp

    # Vẽ biểu đồ 1
    ax1 = axes[0, 0]
    top_15 = interactions['drug_1_name'].value_counts().head(15)
    sns.barplot(x=top_15.values, y=top_15.index, ax=ax1, palette='viridis')
    ax1.set_title('Top 15 Thuốc Có Nhiều Tương Tác Nhất')
    ax1.set_xlabel('Số lượng tương tác')

    # Vẽ biểu đồ 2
    ax2 = axes[0, 1]
    interaction_counts = interactions['drug_1_name'].value_counts()
    ax2.hist(interaction_counts, bins=50, edgecolor='black', color='coral')
    ax2.set_title('Phân bố Số Lượng Tương Tác Của Từng Thuốc')
    ax2.set_xlabel('Số lượng tương tác')
    ax2.set_ylabel('Số lượng Thuốc')
    ax2.set_yscale('log')

    # Vẽ biểu đồ 3
    ax3 = axes[1, 0]
    desc_lengths = interactions['description'].str.len()
    desc_lengths = desc_lengths[desc_lengths.notna()] # Tránh lỗi NaN
    ax3.hist(desc_lengths, bins=50, edgecolor='black', color='lightgreen')
    ax3.set_title('Độ dài dòng Mô tả Tương tác')
    ax3.set_xlabel('Số ký tự')
    ax3.set_ylabel('Tần suất')

    # Vẽ biểu đồ 4
    ax4 = axes[1, 1]
    keyword_counts = {kw: all_descriptions.lower().count(kw) for kw in keywords}
    sns.barplot(x=list(keyword_counts.keys()), y=list(keyword_counts.values()), ax=ax4, palette='magma')
    ax4.set_title('Từ Khóa Tương Tác Phổ Biến')
    ax4.set_xlabel('Từ khóa')
    ax4.set_ylabel('Tần suất')
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show() # ✅ Dòng này vô cùng quan trọng để ép hình ảnh hiện ra NOTEBOOK

    # ------------------ PHẦN TÍNH ĐIỂM RỦI RO TƯƠNG TÁC ------------------
    scorer = DrugRiskScorer(checker)
    print("\n" + "="*60)
    print("🎯 PHẦN 2: DRUG COMBINATION RISK SCORING")
    print("="*60)

    test_combos = [
        ["Aspirin", "Warfarin", "Clopidogrel"],
        ["Metformin", "Lisinopril", "Atorvastatin"],
        ["Aspirin", "Acetaminophen"]
    ]

    for drugs in test_combos:
        print(f"\n📋 Đang đánh giá tổ hợp: {', '.join(drugs)}")
        result = scorer.calculate_risk_score(drugs)

        if 'error' not in result:
            print(f"   Tổng số tương tác tìm thấy: {result['total_interactions']}")
            if result['total_interactions'] > 0:
                print(f"   Điểm Rủi Ro (Risk Score): {result['risk_score']:.1f}/100")
                print(f"   Chi tiết mức độ   : {result['severity_breakdown']}")
            print(f"   Đề xuất         : {result['recommendation']}")
        else:
            print(f"   ❌ Lỗi: {result['error']}")


# In[ ]:




