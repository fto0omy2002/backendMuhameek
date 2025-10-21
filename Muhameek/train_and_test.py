# ===========================
# 1️⃣ استيراد المكتبات
# ===========================
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# ===========================
# 2️⃣ تحميل الداتاست
# ===========================
df = pd.read_csv("legal_cases_dataset.csv")  # تأكد أن الملف موجود

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['Case Description'].tolist(),
    df['Category'].tolist(),
    test_size=0.2,
    random_state=42
)

# ===========================
# 3️⃣ ترميز النصوص
# ===========================
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=256)

# ===========================
# 4️⃣ تحويل الفئات إلى أرقام
# ===========================
label2id = {label: i for i, label in enumerate(df['Category'].unique())}
id2label = {i: label for label, i in label2id.items()}

train_labels_ids = [label2id[label] for label in train_labels]
test_labels_ids = [label2id[label] for label in test_labels]

# ===========================
# 5️⃣ تجهيز Dataset
# ===========================
class LegalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = LegalDataset(train_encodings, train_labels_ids)
test_dataset = LegalDataset(test_encodings, test_labels_ids)

# ===========================
# 6️⃣ تحميل نموذج BERT للتصنيف
# ===========================
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(label2id)
)

# ===========================
# 7️⃣ إعداد التدريب
# ===========================
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir="./logs",
    logging_steps=50,
    learning_rate=2e-5,
    do_eval=True,
    save_total_limit=2
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# ===========================
# 8️⃣ تدريب النموذج
# ===========================
print("🚀 بدء التدريب...")
trainer.train()
print("✅ التدريب انتهى!")

# ===========================
# 9️⃣ تقييم النموذج
# ===========================
results = trainer.evaluate()
print("نتائج التقييم:", results)

# ===========================
# 🔟 حفظ النموذج
# ===========================
model.save_pretrained("./bert_legal_model")
tokenizer.save_pretrained("./bert_legal_model")
print("💾 النموذج تم حفظه في مجلد bert_legal_model")

# ===========================
# 1️⃣1️⃣ تجربة تصنيف نص جديد
# ===========================
def classify_case(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = model(**inputs)
    pred_id = torch.argmax(torch.nn.functional.softmax(outputs.logits, dim=-1), dim=-1).item()
    return id2label[pred_id]

# مثال للتجربة
example_case = "I want divorse"
predicted_category = classify_case(example_case)
print(f"القضية: {example_case}\nالتصنيف المتوقع: {predicted_category}")

print("خلصنا بنجاح")

