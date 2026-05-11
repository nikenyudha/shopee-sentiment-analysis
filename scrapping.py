from google_play_scraper import Sort, reviews
import pandas as pd

# 1. Get ID
app_id = 'com.shopee.id'

print(f"Sedang mengambil data ulasan untuk aplikasi dengan ID: {app_id}")

#2.Get data 
result, _ = reviews(
    app_id,
    lang='id', # Bahasa Indonesia
    country='id', # Indonesia
    sort=Sort.NEWEST, # Urutkan berdasarkan yang terbaru
    count=1000, # Ambil 1000 ulasan
)

#3. Check data 
if not result:
    print("data kosong, cek koneksi internet")
else:
  df = pd.DataFrame(result)
    
    # 4. Tampilkan daftar kolom yang tersedia 
print(f"Kolom yang tersedia di data ini adalah: {df.columns.tolist()}")
    
    # 5. Filter kolom yang dibutuhkan
df = df[['userName', 'score', 'at', 'content']]
    
    # 6. Simpan
df.to_csv('reviews_shopee.csv', index=False)
print(f"Berhasil! {len(df)} review telah disimpan ke reviews_shopee.csv")
    
    # Intip 5 data teratas
print(df.head())
