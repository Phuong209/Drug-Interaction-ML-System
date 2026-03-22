import pandas as pd
from typing import List, Dict, Optional
import os

class DrugInteractionChecker:
    def __init__(self, data_dir: str):
        print("Loading DrugBank interaction database...")
        self.drugs_df = pd.read_csv(os.path.join(data_dir, "drugbank_drugs.csv"))
        self.interactions_df = pd.read_csv(os.path.join(data_dir, "drugbank_interactions.csv"))
        
        # Xây dựng từ điển để tra cứu thuốc cực nhanh O(1)
        self.name_to_id = {}
        for _, row in self.drugs_df.iterrows():
            if pd.notna(row['name']):
                self.name_to_id[row['name'].lower()] = row['drugbank_id']
            if pd.notna(row['all_names']):
                for n in str(row['all_names']).split('|'):
                    if n.strip():
                        self.name_to_id[n.strip().lower()] = row['drugbank_id']
                        
        self.id_to_name = dict(zip(self.drugs_df.drugbank_id, self.drugs_df.name))
        print(f"Loaded {len(self.name_to_id)} drug aliases and {len(self.interactions_df)} interactions.")

    def check_interaction(self, drug1: str, drug2: str) -> Optional[Dict]:
        """Kiểm tra tương tác giữa 2 thuốc độc lập"""
        d1_id = self.name_to_id.get(drug1.lower().strip())
        d2_id = self.name_to_id.get(drug2.lower().strip())
        
        if not d1_id or not d2_id:
            return None
            
        # Tìm bản ghi tương tác trong Database
        interaction = self.interactions_df[
            ((self.interactions_df['drug_1_id'] == d1_id) & (self.interactions_df['drug_2_id'] == d2_id)) |
            ((self.interactions_df['drug_1_id'] == d2_id) & (self.interactions_df['drug_2_id'] == d1_id))
        ]
        
        if not interaction.empty:
            return {
                'drug1': self.id_to_name[d1_id],
                'drug2': self.id_to_name[d2_id],
                'description': interaction.iloc[0]['description']
            }
        return None

    def check_list(self, drugs: List[str]) -> List[Dict]:
        """Quét tương tác chéo cho một danh sách thuốc tìm được"""
        found_interactions = []
        # Duyệt từng cặp thuốc một
        for i in range(len(drugs)):
            for j in range(i+1, len(drugs)):
                result = self.check_interaction(drugs[i], drugs[j])
                if result:
                    found_interactions.append(result)
        return found_interactions
