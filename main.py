from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load file .env dari direktori utama
load_dotenv()

app = FastAPI(title="Shopee Sentiment Analysis API")

# 1. Loading Model and Tokenizer
MODEL_PATH = "nikenlarash22/indobert-shopee-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Konfigurasi Gemini API
gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key:
    genai.configure(api_key=gemini_key)

# 2. Schema Input Pydantic
class ReviewInput(BaseModel):
    text: str

class SummarizeInput(BaseModel):
    reviews: list[str]
    sentiment_type: str

@app.post("/predict")
def predict_sentiment(input_data: ReviewInput):
    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=1).item()
    
    label = "Positif" if prediction == 1 else "Negatif"
    confidence = torch.max(probs).item()
    
    # --- RULE-BASED FALLBACK FOR EDGE CASES ---
    # Jika model ragu (confidence < 0.75) dan mengandung kata kunci negatif yang jelas
    if confidence < 0.75:
        negative_keywords = ['lama', 'lambat', 'jelek', 'rusak', 'kecewa', 'parah', 'buruk', 'cacat', 'kapok']
        if any(word in input_data.text.lower() for word in negative_keywords):
            label = "Negatif"
    
    return {
        "text": input_data.text,
        "sentiment": label,
        "confidence": round(confidence, 4)
    }

# 4. Summarize Endpoint dengan Error Handling
@app.post("/summarize")
def summarize_reviews(input_data: SummarizeInput):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY tidak ditemukan di file .env!")
        
    combined_text = "\n- ".join(input_data.reviews[:30])
    
    prompt = f"""
Act like an E-Commerce Business Analyst.
Analyze the following Shopee customer reviews with sentiment '{input_data.sentiment_type}':

- {combined_text}

Provide a concise, professional summary in Bahasa Indonesia covering:
1. **Aspek Utama / Masalah Utama**: Apa yang paling dipuji atau dikeluhkan pengguna.
2. **Dampak terhadap Kepuasan**: Efeknya terhadap CSAT dan retensi pelanggan.
3. **Rekomendasi Aksi**: Langkah konkret untuk tim Produk & Operasional.

Format the output clearly using clean markdown bullet points.
"""
    
    try:
        gemini_model = genai.GenerativeModel('gemini-3.6-flash')
        response = gemini_model.generate_content(prompt)
        return {"summary": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memanggil Gemini API: {str(e)}")

# 5. Status Check
@app.get("/")
def home():
    return {"status": "API is running!"}