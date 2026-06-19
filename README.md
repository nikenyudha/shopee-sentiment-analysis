# 🛒 Shopee Sentiment & Topic Analysis Dashboard

This project aims to analyze the user experience of the Shopee app on the Google Play Store. Using Natural Language Processing (NLP) techniques, the project automatically classifies user sentiment and identifies key topics from thousands of reviews.

## 🔗 Live Demo App
**Streamlit:** [SHOPEE Live App Streamlit](https://shopee-sentiment-analysis.streamlit.app/)


## 🌟 Key Features
- Real-time Prediction: Analyze review sentiment directly through an interactive dashboard.
- Topic Modeling: Automatically identify key issues (e.g., ads, delivery) using BERTopic.
- Microservices Architecture: Separation of the prediction engine (FastAPI) and the user interface (Streamlit) for performance efficiency.

## 🛠️ Tech Stack & Tools
- Data Collection: google-play-scraper to fetch the latest reviews from the Google Play Store.
- Model: IndoBERT (Fine-tuned) hosted on Hugging Face.
- Backend API: FastAPI to serve model predictions quickly and asynchronously.
- Frontend Dashboard: Streamlit for data visualization, WordCloud, and sentiment analysis by category.

## 📊 Workflow (Pipeline)
- Scraping: Extracting raw review data.
- Preprocessing: Text cleaning, normalization, and stopword removal.
- Modeling: Training an IndoBERT transformer model for binary classification (Positive/Negative).
- Deployment: The model is accessed through the /predict API endpoint on FastAPI. The Streamlit Dashboard calls this API to present visual insights to users.

## 📈 Visualization
This project provides various visualizations, such as:
Sentiment Distribution Bar Chart.
WordCloud to see keywords that frequently appear in positive vs. negative reviews.
Sentiment per Topic Analysis to see which topics receive the most complaints.

Created by Niken Larasati
