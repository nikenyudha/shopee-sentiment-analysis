import pandas as pd
import re

# 1. Load data yang sudah di-scrape tadi
df = pd.read_csv('reviews_shopee.csv')

def clean_text(text):
    # Ubah ke string dan kecilkan huruf
    text = str(text).lower()
    # Hapus URL/Link
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Hapus mention (@) dan hashtag (#)
    text = re.sub(r'@\w+|#\w+', '', text)
    # Hapus angka dan tanda baca
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Hapus spasi berlebih
    text = text.strip()
    return text

# 2. Jalankan fungsi pembersih
df['content_cleaned'] = df['content'].apply(clean_text)

# 3. Buat Label Sentimen Sederhana (Pseudo-labeling)
# Skor 4-5 = Positif (1), Skor 1-2 = Negatif (0), Skor 3 = Netral ( abaikan dulu agar fokus)
def create_label(score):
    if score >= 4:
        return 1
    elif score <= 2:
        return 0
    else:
        return None # Netral

df['label'] = df['score'].apply(create_label)

# Hapus yang netral agar model belajar perbedaan yang kontras
df = df.dropna(subset=['label'])

# 4. Simpan data yang sudah bersih
df[['content_cleaned', 'label']].to_csv('shopee_reviews_cleaned.csv', index=False)

print("Data sudah bersih dan siap diolah!")
print(df[['content_cleaned', 'label']].head())