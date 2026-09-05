# 🛒 Shopee AI Customer Insights Engine & Topic Analysis

An end-to-end NLP analytics platform designed to analyze user experience and reviews of the Shopee app from the Google Play Store. This project combines **IndoBERT** for fine-tuned sentiment classification, **BERTopic** for automated issue clustering, and **Gemini 3.6 Flash** for real-time executive AI summaries—all integrated via a microservice architecture.

## 🔗 Live Demo App
* **Streamlit Dashboard:** [SHOPEE Live App Streamlit](https://shopee-sentiment-analysis.streamlit.app/)

---

## 🌟 Key Features
- **Real-Time Sentiment Classification**: Fine-tuned IndoBERT model with custom fallback rules for edge-case short texts.
- **Topic Modeling Insights**: Automatically uncovers underlying themes and main customer complaints (e.g., ads, delivery, payment) using BERTopic.
- **Generative AI Executive Summaries**: Translates raw customer feedback into actionable business insights and operational recommendations via Gemini 3.6 Flash.
- **Microservices Architecture**: Decoupled system architecture separating the prediction engine (FastAPI) and the user interface (Streamlit) for high performance and scalability.

---

## 🛠️ Tech Stack & Tools
- **Data Collection**: `google-play-scraper` to fetch user reviews.
- **NLP & Machine Learning**: PyTorch, Hugging Face Transformers (**IndoBERT**), **BERTopic**, WordCloud.
- **Generative AI**: Google Gemini API (`google-generativeai` / `gemini-3.6-flash`).
- **Backend API**: **FastAPI**, Uvicorn, Pydantic for structured data validation.
- **Frontend Dashboard**: **Streamlit** for interactive visualizations, real-time model testing, and markdown rendering.

---

## 📊 Workflow & Pipeline
1. **Scraping & Preprocessing**: Extracting 1000+ raw review data, text cleaning, normalization, and stopword removal.
2. **Modeling & Topic Extraction**: Fine-tuning IndoBERT for binary classification (Positive/Negative) and clustering topics using BERTopic.
3. **Backend Serving (FastAPI)**: Serving both sentiment prediction (`/predict`) and generative summarization (`/summarize`) endpoints asynchronously.
4. **Interactive Visualization (Streamlit)**: Fetching backend API results to display real-time predictions, sentiment distributions, topic breakdowns, and Gemini AI insights.

---

## 📈 Visualization
**This project provides various visualizations, such as**:
- Sentiment Distribution Bar Chart.
- WordCloud to see keywords that frequently appear in positive vs. negative reviews.
- Sentiment per Topic Analysis to see which topics receive the most complaints.
---

## 📡 API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Health check endpoint |
| `POST` | `/predict` | Predicts sentiment label and confidence score for a single review |
| `POST` | `/summarize` | Generates AI business summary and actionable steps using Gemini 3.6 Flash |

---

## 🚀 How to Run Locally

### 1. Clone & Environment Setup
```bash
git clone [https://github.com/nikenlarash22/shopee-sentiment-analysis.git](https://github.com/nikenlarash22/shopee-sentiment-analysis.git)
cd shopee-sentiment-analysis
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

```

Created by Niken Larasati W
