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
        current_drug = []
        
        for token, pred in zip(tokens, predictions):
            label = self.id2label[pred.item()]
            
            # Tách bỏ các từ đặc biệt của Tokenizer
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            # Loại bỏ kỹ hiệu subword ##
            clean_token = token.replace("##", "")
            
            if label == 'B-DRUG':
                if current_drug:
                    # Nếu đang có thuốc trước đó thì đóng lại
                    drugs.append(" ".join(current_drug))
                    current_drug = []
                current_drug.append(clean_token)
            elif label == 'I-DRUG':
                current_drug.append(clean_token)
            else:
                if current_drug:
                    drugs.append(" ".join(current_drug))
                    current_drug = []
                    
        if current_drug:
            drugs.append(" ".join(current_drug))
            
        # Format lại chữ, bỏ khoảng trắng dư do ## tạo ra và lọc trùng
        seen = set()
        cleaned_drugs = []
        for d in drugs:
            d = d.replace("  ", " ").strip()
            # Loại bỏ các chữ rác vô nghĩa bị nhận nhầm
            if len(d) > 2 and d.lower() not in seen:
                cleaned_drugs.append(d)
                seen.add(d.lower())
                
        return cleaned_drugs
