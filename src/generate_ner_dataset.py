import pandas as pd
import json
import re
from tqdm import tqdm
import os

DRUGS_PATH = "/Users/lydang/Downloads/ML Project/Data/processed/drugbank_drugs.csv"
INTERACTIONS_PATH = "/Users/lydang/Downloads/ML Project/Data/processed/drugbank_interactions.csv"
OUTPUT_PATH = "/Users/lydang/Downloads/ML Project/Data/processed/ddi_bio_format.json"

def tokenize(text):
    # Simple tokenization: split by spaces, keep punctuation as separate tokens
    tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
    return tokens

def main():
    if os.path.exists(OUTPUT_PATH):
        print(f"File {OUTPUT_PATH} already exists.")
        return

    print("Loading drugs...")
    drugs_df = pd.read_csv(DRUGS_PATH)
    drug_names = set()
    for _, row in drugs_df.iterrows():
        if pd.notna(row['name']):
            drug_names.add(row['name'].lower())
        if pd.notna(row['all_names']):
            for n in str(row['all_names']).split('|'):
                clean_name = n.strip().lower()
                if len(clean_name) > 2:
                    drug_names.add(clean_name)

    print("Loading interactions...")
    # Use a sample of 10,000 interactions to keep the dataset size manageable for local training
    interactions_df = pd.read_csv(INTERACTIONS_PATH).dropna(subset=['description']).sample(10000, random_state=42)

    bio_data = []

    print("Generating BIO formatted dataset...")
    for desc in tqdm(interactions_df['description']):
        tokens = tokenize(desc)
        tags = ['O'] * len(tokens)
        
        i = 0
        while i < len(tokens):
            found_drug = False
            # Max n-gram for drug names (check up to 4 words)
            for n_gram in range(min(4, len(tokens) - i), 0, -1):
                phrase = " ".join(tokens[i:i+n_gram]).lower()
                if phrase in drug_names:
                    tags[i] = 'B-DRUG'
                    for j in range(1, n_gram):
                        tags[i+j] = 'I-DRUG'
                    i += n_gram
                    found_drug = True
                    break
            
            if not found_drug:
                i += 1
                
        # Only add sentences that actually contain drug entities
        if 'B-DRUG' in tags:
            bio_data.append({
                'tokens': tokens,
                'tags': tags
            })

    print(f"Generated {len(bio_data)} annotated sentences.")

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(bio_data, f, indent=2, ensure_ascii=False)

    print(f"Saved BIO dataset to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
