import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from typing import List

class DrugNERPipeline:
    def __init__(self, model_dir: str):
        print(f"Loading AI Model from {model_dir}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()
        self.id2label = self.model.config.id2label

    def extract_drugs(self, text: str) -> List[str]:
        """Đưa văn bản thô vào để trích xuất danh sách tên thuốc"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        predictions = torch.argmax(outputs.logits, dim=2)[0]
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        
        drugs = []
        current_drug = ""
        
        for token, pred in zip(tokens, predictions):
            label = self.id2label[pred.item()]
            
            # Bỏ qua special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']:
                continue
                
            is_subword = token.startswith("##")
            clean_token = token.replace("##", "")
            
            if label == 'B-DRUG':
                # Model đôi khi gán B-DRUG cho từng ##subword thay vì I-DRUG
                if is_subword and current_drug:
                    current_drug += clean_token
                else:
                    if current_drug:
                        drugs.append(current_drug)
                    current_drug = clean_token
            elif label == 'I-DRUG':
                if not current_drug:
                    current_drug = clean_token
                else:
                    if is_subword:
                        current_drug += clean_token
                    else:
                        current_drug += " " + clean_token
            else: # label == 'O'
                # Đôi khi model gán O cho phần đuôi của subword
                if is_subword and current_drug:
                    current_drug += clean_token
                else:
                    if current_drug:
                        drugs.append(current_drug)
                        current_drug = ""
                        
        if current_drug:
            drugs.append(current_drug)
            
        # Format lại chữ, loại bỏ rác và lọc trùng
        seen = set()
        cleaned_drugs = []
        for d in drugs:
            d = d.replace("  ", " ").strip()
            # Giữ lại nếu là tên thuốc có nghĩa (Dài hơn 2 kí tự)
            if len(d) > 2 and d.lower() not in seen:
                cleaned_drugs.append(d)
                seen.add(d.lower())
                
        return cleaned_drugs
