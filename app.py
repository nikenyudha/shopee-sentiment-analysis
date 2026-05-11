import streamlit as st
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from wordcloud import WordCloud
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
import torch
import os

# --- 1. SETTING HALAMAN ---
st.set_page_config(page_title="Shopee Sentiment Analysis", layout="wide")
st.title("🛒 Shopee Sentiment & Topic Analysis")
st.markdown("""
Dashboard ini menganalisis review user aplikasi **Shopee** menggunakan model **IndoBERT** yang telah di-fine-tune. 
Proyek ini membantu mengidentifikasi kepuasan pengguna dan area yang perlu diperbaiki.
""")

# --- 2. LOAD MODEL & DATA ---
@st.cache_resource
def load_model():
    model_path = "nikenlarash22/indobert-shopee-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    return tokenizer, model

tokenizer, model = load_model()

# Menggunakan path yang aman agar tidak error di server
base_path = os.path.dirname(__file__)
csv_path = os.path.join(base_path, 'data', 'shopee_reviews_cleaned.csv')
df = pd.read_csv(csv_path)

# Mapping Label (Hanya dilakukan sekali di sini)
df['Sentiment_Label'] = df['label'].map({1: 'Positif 😊', 0: 'Negatif 😡'})

# --- 3. SIDEBAR: UJI REAL-TIME ---
st.sidebar.header("🔍 Uji Model Real-Time")
st.sidebar.info("Ketik review di bawah untuk melihat bagaimana model AI mengklasifikasikannya.")
user_input = st.sidebar.text_area("Masukkan teks review:")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        # Mendapatkan Probabilitas (Confidence Score)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=1).item()
        conf_score = torch.max(probs).item() * 100
    
    label = "POSITIF 😊" if prediction == 1 else "NEGATIF 😡"
    st.sidebar.markdown(f"### Hasil: **{label}**")
    st.sidebar.progress(conf_score / 100)
    st.sidebar.write(f"Tingkat Keyakinan Model: {conf_score:.2f}%")

# --- 4. MAIN DASHBOARD: VISUALISASI ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📈 Ringkasan Sentimen")
    sentiment_count = df['Sentiment_Label'].value_counts()
    st.bar_chart(sentiment_count)
    
    # Tambahan: Statistik Sederhana
    total_data = len(df)
    pos_perc = (df['label'] == 1).sum() / total_data * 100
    st.metric("Total Review", total_data)
    st.metric("Sentimen Positif", f"{pos_perc:.1f}%")

with col2:
    st.subheader("📄 Sampel Data Terbaru")
    st.dataframe(
        df[['content_cleaned', 'Sentiment_Label']].head(15),
        use_container_width=True
    )

# --- 5. TOPIC MODELING INSIGHT (BERTopic) - OTOMATIS ---
st.divider()
st.subheader("📌 Temuan Utama (Topic Modeling)")

try:
    # Load hasil BERTopic
    topic_path = os.path.join(base_path, 'data', 'bertopic_results.csv')
    df_topics = pd.read_csv(topic_path)

    # Filter topik (buang topik -1 karena itu biasanya Outliers/Noise)
    df_filtered = df_topics[df_topics['Topic'] != -1].head(3)

    if not df_filtered.empty:
        st.write(f"Berdasarkan analisis BERTopic, berikut adalah {len(df_filtered)} topik utama yang paling sering dibahas:")
        
        cols = st.columns(len(df_filtered))
        
        for i, row in df_filtered.iterrows():
            with cols[i % len(df_filtered)]:
                # Menampilkan Nama Topik (misal: 0_dana_cicil_bayar)
                topic_name = row['Name'].split('_')[1:] # Ambil kata-katanya saja
                topic_name = " ".join(topic_name).title()
                
                st.info(f"### {topic_name}")
                st.write(f"**Jumlah Review:** {row['Count']}")
                # Menampilkan kata kunci representasi topik tersebut
                st.caption(f"Kata kunci: {row['Representation']}")
    else:
        st.write("Belum ada topik yang teridentifikasi.")

except FileNotFoundError:
    st.warning("File bertopic_results.csv tidak ditemukan. Pastikan sudah mengunggah hasil BERTopic ke folder data.")

# --- 6. ANALISIS SENTIMEN PER TOPIK ---
st.divider()
st.subheader("📊 Analisis Sentimen per Kategori Masalah")

try:
    # Membaca data yang baru saja kamu buat
    df_sent_topic = pd.read_csv('data/shopee_sentiment_per_topic.csv')
    
    # Menghapus baris jika ada topik -1 (Outliers) yang terbawa
    df_sent_topic = df_sent_topic[df_sent_topic['Topic'] != -1]

    # Membersihkan nama topik: dari "0_iklan_video_ganggu" menjadi "Iklan Video Ganggu"
    def clean_topic_name(name):
        parts = name.split('_')
        if len(parts) > 1:
            return " ".join(parts[1:]).title()
        return name

    df_sent_topic['Kategori'] = df_sent_topic['Name'].apply(clean_topic_name)
    
    # Mengambil kolom sentimen yang sudah fix ada di CSV 
    #set 'Kategori' sebagai index agar muncul di sumbu X grafik
    df_plot = df_sent_topic.set_index('Kategori')[['Positif 😊', 'Negatif 😡']]
    
    # Menampilkan Stacked Bar Chart
    # Secara default, Streamlit akan menumpuk (stack) kolom jika indexnya sama
    st.bar_chart(df_plot)
    
    st.caption("Grafik ini menunjukkan perbandingan jumlah sentimen positif dan negatif untuk setiap topik utama.")

except Exception as e:
    st.error(f"Gagal memuat grafik sentimen per topik: {e}")
    st.info("Pastikan file 'shopee_sentiment_per_topic.csv' sudah di-upload ke folder data.")

# --- 7. WORDCLOUD PER SENTIMEN ---
st.divider()
st.subheader("☁️ Awan Kata (Word Cloud)")

# Fungsi untuk membuat WordCloud
def buat_wordcloud(data, color):
    # Gabungkan semua teks menjadi satu string besar
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

# Tambahkan daftar kata yang ingin dibuang
additional_stopwords = {
    'shopee', 'aplikasi', 'saya', 'yang', 'dan', 'di', 'ini', 'ada', 
    'untuk', 'dengan', 'banget', 'dah', 'sudah', 'bisa', 'aja', 'jadi',
    'kalau', 'sama', 'tapi', 'gak', 'ke', 'dari', 'lagi', 'buat'
}


# Load data review yang sudah di-clean (file utama dashboard-mu)
# Asumsi variabel 'df' adalah dataframe hasil load reviews_cleaned.csv
col1, col2 = st.columns(2)

with col1:
    st.write("### Review Positif 😊")
    # Filter label 1 untuk positif
    pos_data = df[df['label'] == 1]['content_cleaned']
    fig_pos = buat_wordcloud(pos_data, 'viridis') # Warna hijau-biru
    if fig_pos:
        st.pyplot(fig_pos)
    else:
        st.write("Data tidak cukup untuk Wordcloud.")

with col2:
    st.write("### Review Negatif 😡")
    # Filter label 0 untuk negatif
    neg_data = df[df['label'] == 0]['content_cleaned']
    fig_neg = buat_wordcloud(neg_data, 'magma') # Warna merah-jingga
    if fig_neg:
        st.pyplot(fig_neg)
    else:
        st.write("Data tidak cukup untuk Wordcloud.")


st.markdown(
    "<hr style='margin-top:50px;'>"
    "<center style='color: gray;'>© 2026 Niken Larasati — Shopee Sentiment and Topic Analysis💗</center>",
    unsafe_allow_html=True
)


