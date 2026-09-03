import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import requests  # Used to shoot to API

# --- GLOBAL CONFIG & API URL ---
API_URL = "http://127.0.0.1:8000"

# --- 1. PAGE SETTINGS ---
st.set_page_config(page_title="SHOPEE Sentiment Analysis", layout="wide")
st.title("🛒 Shopee AI Customer Insights Engine")
st.markdown("""
An end-to-end NLP analytics platform leveraging **IndoBERT** for deep sentiment classification, **BERTopic** for automated issue clustering, and **Gemini 3.6 Flash** for real-time executive summaries via **FastAPI**.
""")

# --- 2. FUNCTION TO CALL API ---
def get_prediction_from_api(text):
    """Function to send text to FastAPI and receive prediction results"""
    url = f"{API_URL}/predict"
    payload = {"text": text}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": "API Error"}
    except Exception as e:
        return {"error": str(e)}

# Load Data 
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, 'data', 'shopee_reviews_cleaned.csv')
df = pd.read_csv(csv_path)
df['Sentiment_Label'] = df['label'].map({1: 'Positif 😊', 0: 'Negatif 😡'})

# ---------------------------------------------------------
# SECTION 1: Single Review Sentiment Prediction
# ---------------------------------------------------------
st.header("1. Single-Review Sentiment Analysis")
user_review = st.text_area("Enter customer review:", "Fast delivery, product matches description!")

if st.button("Sentiment Prediction"):
    if user_review.strip():
        try:
            response = requests.post(
                f"{API_URL}/predict",
                json={"text": user_review}
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"**Sentimen:** {result['sentiment']} (Confidence: {result['confidence'] * 100:.2f}%)")
            else:
                st.error("Predict with API failed.")
        except Exception as e:
            st.error(f"Connection to the backend failed: {e}")
    else:
        st.warning("Please insert the review first")

st.divider()

# ---------------------------------------------------------
# SECTION 2: AI Business Insight Summary (Gemini)
# ---------------------------------------------------------
st.header("2. AI Business Insight Summarizer")
st.caption("Get an executive summary of a collection of reviews using Gemini AI")

# Input Kumpulan Ulasan & Tipe Sentimen
sentiment_option = st.selectbox("Pilih Tipe Sentimen Ulasan:", ["Negatif", "Positif"])
raw_reviews_input = st.text_area(
    "Masukkan kumpulan ulasan (pisahkan tiap ulasan dengan baris baru / Enter):",
    height=150,
    value="Pengiriman sangat lambat, butuh seminggu baru sampai\nBarang yang dikirim tidak sesuai warna\nRespon penjual sangat cuek dan lambat"
)

if st.button("Generate AI Insight"):
    # Split teks input berdasarkan baris menjadi list ulasan
    reviews_list = [line.strip() for line in raw_reviews_input.split("\n") if line.strip()]
    
    if reviews_list:
        with st.spinner("Gemini sedang menganalisis ulasan..."):
            try:
                payload = {
                    "reviews": reviews_list,
                    "sentiment_type": sentiment_option
                }
                
                res = requests.post(f"{API_URL}/summarize", json=payload)
                
                if res.status_code == 200:
                    summary_text = res.json().get("summary", "")
                    st.subheader("💡 Ringkasan Analisis Bisnis")
                    st.markdown(summary_text)
                else:
                    error_detail = res.json().get("detail", "Terjadi kesalahan pada server.")
                    st.error(f"Gagal memuat ringkasan: {error_detail}")
            except Exception as e:
                st.error(f"Tidak dapat terhubung ke server backend: {e}")
    else:
        st.warning("Mohon masukkan minimal satu ulasan.")

# --- 3. SIDEBAR: REAL-TIME TESTING (USING API) ---
st.sidebar.header("🔍 Real-Time Model Testing")
st.sidebar.info("Type your review below. The dashboard will query FastAPI for the results.")
user_input = st.sidebar.text_area("Enter review text:")

if user_input:
    with st.sidebar.status("Contacting the Model API...", expanded=True) as status:
        hasil = get_prediction_from_api(user_input)
        
        if "error" not in hasil:
            label = hasil['sentiment'].upper()
            conf_score = hasil['confidence'] * 100
            
            st.sidebar.markdown(f"### Result: **{label}**")
            st.sidebar.progress(conf_score / 100)
            st.sidebar.write(f"Model Confidence Level: {conf_score:.2f}%")
            status.update(label="Analysis Completed!", state="complete")
        else:
            st.sidebar.error(f"Failed to connect to API: {hasil['error']}")

# --- 4. MAIN DASHBOARD: VISUALISASI ---
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📈 Sentiment Summary")
    sentiment_count = df['Sentiment_Label'].value_counts()
    st.bar_chart(sentiment_count)
    total_data = len(df)
    pos_perc = (df['label'] == 1).sum() / total_data * 100
    st.metric("Total Review", total_data)
    st.metric("Sentimen Positive", f"{pos_perc:.1f}%")

with col2:
    st.subheader("📄 Latest Sample Data")
    st.dataframe(df[['content_cleaned', 'Sentiment_Label']].head(15), use_container_width=True)

# --- 5. TOPIC MODELING INSIGHT (BERTopic) ---
st.divider()
st.subheader("📌 Key Findings (Topic Modeling)")

try:
    topic_path = os.path.join(base_path, 'data', 'bertopic_results.csv')
    df_topics = pd.read_csv(topic_path)
    df_filtered = df_topics[df_topics['Topic'] != -1].head(3)

    if not df_filtered.empty:
        st.write(f"Based on BERTopic analysis, the following are the top {len(df_filtered)} topics that are most frequently discussed:")
        cols = st.columns(len(df_filtered))
        
        for i, row in df_filtered.reset_index().iterrows():
            with cols[i % len(df_filtered)]:
                topic_name = row['Name'].split('_')[1:]
                topic_name = " ".join(topic_name).title()
                
                st.info(f"### {topic_name}")
                st.write(f"**Jumlah Review:** {row['Count']}")
                st.caption(f"Kata kunci: {row['Representation']}")
    else:
        st.write("No topics identified yet.")

except FileNotFoundError:
    st.warning("File bertopic_results.csv not found. Please make sure you have uploaded the BERTopic results to the data folder.")

# --- 6. SENTIMENT ANALYSIS PER TOPIC ---
st.divider()
st.subheader("📊 Sentiment Analysis per Topic")

try:
    topic_sent_path = os.path.join(base_path, 'data', 'shopee_sentiment_per_topic.csv')
    df_sent_topic = pd.read_csv(topic_sent_path)
    df_sent_topic = df_sent_topic[df_sent_topic['Topic'] != -1]

    def clean_topic_name(name):
        parts = name.split('_')
        if len(parts) > 1:
            return " ".join(parts[1:]).title()
        return name

    df_sent_topic['Kategori'] = df_sent_topic['Name'].apply(clean_topic_name)
    df_plot = df_sent_topic.set_index('Kategori')[['Positif 😊', 'Negatif 😡']]
    
    st.bar_chart(df_plot)
    st.caption("This graph shows a comparison of the number of positive and negative sentiments for each main topic.")

except Exception as e:
    st.error(f"Failed to load sentiment chart per topic: {e}")
    st.info("Please make sure the file 'shopee_sentiment_per_topic.csv' is uploaded to the data folder.")

# --- 7. WORDCLOUD PER SENTIMEN ---
st.divider()
st.subheader("☁️ Word Cloud")

additional_stopwords = {
    'shopee', 'aplikasi', 'saya', 'yang', 'dan', 'di', 'ini', 'ada', 
    'untuk', 'dengan', 'banget', 'dah', 'sudah', 'bisa', 'aja', 'jadi',
    'kalau', 'sama', 'tapi', 'gak', 'ke', 'dari', 'lagi', 'buat'
}

def buat_wordcloud(data, color):
    text = " ".join(data.dropna())
    if len(text) > 10:
        wc = WordCloud(
            background_color='white', 
            max_words=100, 
            colormap=color,
            stopwords=additional_stopwords,
            width=800, 
            height=400
        ).generate(text)
        
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        return fig
    else:
        return None

col1, col2 = st.columns(2)

with col1:
    st.write("### Review Positif 😊")
    pos_data = df[df['label'] == 1]['content_cleaned']
    fig_pos = buat_wordcloud(pos_data, 'viridis') 
    if fig_pos:
        st.pyplot(fig_pos)
    else:
        st.write("Insufficient data for Wordcloud.")

with col2:
    st.write("### Review Negatif 😡")
    neg_data = df[df['label'] == 0]['content_cleaned']
    fig_neg = buat_wordcloud(neg_data, 'magma')
    if fig_neg:
        st.pyplot(fig_neg)
    else:
        st.write("Insufficient data for Wordcloud.")

# Footer
st.markdown(
    "<hr style='margin-top:50px;'>"
    "<center style='color: gray;'>© 2026 Niken Larasati — Shopee AI Customer Insights Engine 💗</center>",
    unsafe_allow_html=True
)