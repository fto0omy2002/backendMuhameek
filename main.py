from fastapi import FastAPI
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from fastapi.middleware.cors import CORSMiddleware  # ✅ جديد

# =========================
# إعدادات عامة
# =========================
app = FastAPI(title="Case Analyzer API")

# ✅ إضافة CORS حتى Voiceflow يقدر يتصل بدون "Can't send request"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # لاحقًا يمكنك تخصيصها
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# تحميل الموديل
# =========================
MODEL_PATH = "bert_legal_model"

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # وضع التقييم

# =========================
# نماذج الطلب والرد
# =========================
class CaseRequest(BaseModel):
    case_text: str

class CaseResponse(BaseModel):
    case_type: str

# =========================
# دوال مساعدة
# =========================
def classify_case(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=1)
    label_map = {
        0: "Personal Status",
        1: "Commercial",
        2: "General"
    }
    return label_map[predictions.item()]

# =========================
# المسارات (Routes)
# =========================
@app.get("/")  # ✅ صفحة فحص بسيطة
def home():
    return {"message": "Server is up"}

@app.get("/analyze-case")  # ✅ توضيح في حال أحد فتح الرابط بالمتصفح
def hint():
    return {"hint": "Use POST /analyze-case with JSON body {'case_text': '...'}"}

@app.post("/analyze-case", response_model=CaseResponse)
async def analyze_case(request: CaseRequest):
    case_type = classify_case(request.case_text)
    return {"case_type": case_type}
