import streamlit as st
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import requests  # Used to shoot to API

# --- 1. PAGE SETTINGS --- ---
st.set_page_config(page_title="SHOPEE Sentiment Analysis", layout="wide")
st.title("🛒 SHOPEE Sentiment & Topic Analysis")
st.markdown("""
This dashboard analyzes user reviews of the **SHOPEE** app using the **IndoBERT** model via **FastAPI**.
""")

# --- 2. FUNCTION TO CALL API (REPLACE MODEL LOADING) ---
def get_prediction_from_api(text):
    """Function to send text to FastAPI and receive prediction results"""
    url = "http://127.0.0.1:8000/predict" # FastAPI address
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

# --- 3. SIDEBAR: REAL-TIME TESTING (USING API) ---
st.sidebar.header("🔍 Real-Time Model Testing")
st.sidebar.info("Type your review below. The dashboard will query FastAPI for the results.")
user_input = st.sidebar.text_area("Enter review text:")

if user_input:
    # We call the API function, instead of manually calculating using Torch anymore
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

# --- 4. MAIN DASHBOARD: VISUALISASI  ---
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

# --- 5. TOPIC MODELING INSIGHT (BERTopic) - OTOMATION ---
st.divider()
st.subheader("📌 Key Findings (Topic Modeling)")

try:
    # Load BERTopic results
    topic_path = os.path.join(base_path, 'data', 'bertopic_results.csv')
    df_topics = pd.read_csv(topic_path)

    # Filter topics (remove -1 topics as they are usually Outliers/Noise)
    df_filtered = df_topics[df_topics['Topic'] != -1].head(3)

    if not df_filtered.empty:
        st.write(f"Based on BERTopic analysis, the following are the top {len(df_filtered)} topics that are most frequently discussed:")
        
        cols = st.columns(len(df_filtered))
        
        for i, row in df_filtered.iterrows():
            with cols[i % len(df_filtered)]:
                # Display Topic Name (eg: 0_dana_cicil_bayar)
                topic_name = row['Name'].split('_')[1:] #Just take his word for it
                topic_name = " ".join(topic_name).title()
                
                st.info(f"### {topic_name}")
                st.write(f"**Jumlah Review:** {row['Count']}")
                # Displays the keywords representing the topic
                st.caption(f"Kata kunci: {row['Representation']}")
    else:
        st.write("No topics identified yet.")

except FileNotFoundError:
    st.warning("File bertopic_results.csv not found. Please make sure you have uploaded the BERTopic results to the data folder.")

# --- 6. SENTIMENT ANALYSIS PER TOPIC---
st.divider()
st.subheader("📊 Sentiment Analysis per Topic")

try:
    # Reading the data that was just created
    df_sent_topic = pd.read_csv('data/shopee_sentiment_per_topic.csv')
    
    # Removing rows if there are -1 topics (Outliers) that are included
    df_sent_topic = df_sent_topic[df_sent_topic['Topic'] != -1]

    # Cleaning topic names: from "0_iklan_video_ganggu" to "Iklan Video Ganggu"
    def clean_topic_name(name):
        parts = name.split('_')
        if len(parts) > 1:
            return " ".join(parts[1:]).title()
        return name

    df_sent_topic['Kategori'] = df_sent_topic['Name'].apply(clean_topic_name)
    
    # Fetching the sentiment columns that are already fixed in the CSV
    # Set 'Kategori' as the index so it appears on the X-axis of the chart
    df_plot = df_sent_topic.set_index('Kategori')[['Positif 😊', 'Negatif 😡']]
    
    # Displaying the Stacked Bar Chart
    # By default, Streamlit will stack columns if their index is the same
    st.bar_chart(df_plot)
    
    st.caption("This graph shows a comparison of the number of positive and negative sentiments for each main topic.")

except Exception as e:
    st.error(f"Failed to load sentiment chart per topic: {e}")
    st.info("Please make sure the file 'shopee_sentiment_per_topic.csv' is uploaded to the data folder.")

# --- 7. WORDCLOUD PER SENTIMEN ---
st.divider()
st.subheader("☁️ Word Cloud")

# Function for creating WordCloud
def buat_wordcloud(data, color):
    # Combine all text into one large string
    text = " ".join(data.dropna())
    if len(text) > 10:
        wc = WordCloud(
            background_color='white', 
            max_words=100, 
            colormap=color,
            width=800, 
            height=400
        ).generate(text)
        
        fig, ax = plt.subplots()
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        return fig
    else:
        return None

# Add a list of words to exclude
additional_stopwords = {
    'shopee', 'aplikasi', 'saya', 'yang', 'dan', 'di', 'ini', 'ada', 
    'untuk', 'dengan', 'banget', 'dah', 'sudah', 'bisa', 'aja', 'jadi',
    'kalau', 'sama', 'tapi', 'gak', 'ke', 'dari', 'lagi', 'buat'
}


# Load the cleaned review data (main file)
# Assuming the variable 'df' is the dataframe resulting from loading reviews_cleaned.csv
col1, col2 = st.columns(2)

with col1:
    st.write("### Review Positif 😊")
    # Filter label 1 for positive reviews
    pos_data = df[df['label'] == 1]['content_cleaned']
    fig_pos = buat_wordcloud(pos_data, 'viridis') 
    if fig_pos:
        st.pyplot(fig_pos)
    else:
        st.write("Insufficient data for Wordcloud.")

with col2:
    st.write("### Review Negatif 😡")
    # Filter label 0 for negative reviews
    neg_data = df[df['label'] == 0]['content_cleaned']
    fig_neg = buat_wordcloud(neg_data, 'magma') # Warna merah-jingga
    if fig_neg:
        st.pyplot(fig_neg)
    else:
        st.write("Insufficient data for Wordcloud.")


st.markdown(
    "<hr style='margin-top:50px;'>"
    "<center style='color: gray;'>© 2026 Niken Larasati — SHOPEE Sentiment and Topic Analysis💗</center>",
    unsafe_allow_html=True
)


