from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = FastAPI(title="Shopee Sentiment Analysis API")

# 1. Loading Model and Tokenizer when API is turned on
MODEL_PATH = "nikenlarash22/indobert-shopee-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# 2. Define input data format (Request Body)
class ReviewInput(BaseModel):
    text: str

# 3. Prediction Endpoint
@app.post("/predict")
def predict_sentiment(input_data: ReviewInput):
    # Tokenization Process
    inputs = tokenizer(input_data.text, return_tensors="pt", truncation=True, max_length=128).to(device)
    
    # Prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=1).item()
    
    label = "Positif" if prediction == 1 else "Negatif"
    confidence = torch.max(probs).item()
    
    return {
        "text": input_data.text,
        "sentiment": label,
        "confidence": round(confidence, 4)
    }

# Endpoint to check API status
@app.get("/")
def home():
    return {"status": "API is running!"}