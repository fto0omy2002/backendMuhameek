from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# تحميل الموديل المحفوظ
MODEL_PATH = "bert_legal_model"  

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # مهم جدًا، وضع الموديل في وضع التقييم

class CaseRequest(BaseModel):
    case_text: str

class CaseResponse(BaseModel):
    case_type: str

app = FastAPI(title="Case Analyzer API")

# دالة تصنيف النص
def classify_case(text: str) -> str:
    # تحويل النص إلى tokens
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    # تحويل النتيجة إلى التصنيف النهائي
    # افترضنا أن عندك dict يحول رقم الفئة إلى اسم القضية:
    label_map = {
        0: "Personal Status",
        1: "Commercial",
        2: "General"
    }
    return label_map[predictions.item()]

@app.post("/analyze-case", response_model=CaseResponse)
async def analyze_case(request: CaseRequest):
    case_type = classify_case(request.case_text)
    return {"case_type": case_type}