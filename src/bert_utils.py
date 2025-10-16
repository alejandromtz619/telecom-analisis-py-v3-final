
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Modelo por defecto recomendado (5 clases)
DEFAULT_BERT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

LABEL_MAP_5 = {0: "muy negativo", 1: "negativo", 2: "neutro", 3: "positivo", 4: "muy positivo"}

def _collapse_5_to_3(label_5: str) -> str:
    if label_5 in {"muy negativo", "negativo"}: return "negativo"
    if label_5 in {"muy positivo", "positivo"}: return "positivo"
    return "neutro"

def bert_predict_3(texts: List[str], model_name: str = DEFAULT_BERT_MODEL, batch_size: int = 16) -> List[str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, truncation=True, padding=True, return_tensors="pt").to(device)
            out = model(**enc)
            labels = torch.argmax(out.logits, dim=1).cpu().numpy()
            preds.extend([LABEL_MAP_5[int(x)] for x in labels])
    return [_collapse_5_to_3(y) for y in preds]
