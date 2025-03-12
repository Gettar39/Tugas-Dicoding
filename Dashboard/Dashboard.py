import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gdown

# Judul Aplikasi
st.title("Analisis Data Penyewaan Sepeda")

# Menampilkan Deskripsi Aplikasi
st.write("""
Aplikasi ini menganalisis data penyewaan sepeda berdasarkan berbagai faktor seperti cuaca, suhu, dan hari kerja.
""")

# Gathering Data
url = "https://drive.google.com/file/d/1JWDPQz_8Ndu3MD1owAKL_lZ8Fxf8TIBi/view"
output = 'data.csv'
gdown.download(url=url, output=output, quiet=False, fuzzy=True)
data_bicyle = pd.read_csv(output)

# Menampilkan Data
st.subheader("Data Penyewaan Sepeda")
st.write(data_bicyle.head())

# Data Wrangling
st.subheader("Data Wrangling")
st.write("Informasi Dataset:")
st.write(data_bicyle.info())

st.write("Jumlah Nilai yang Hilang:")
st.write(data_bicyle.isna().sum())

st.write("Jumlah Duplikasi:")
st.write(data_bicyle.duplicated().sum())

# Exploratory Data Analysis (EDA)
st.subheader("Exploratory Data Analysis (EDA)")

# Visualisasi Jumlah Sepeda yang Disewa Berdasarkan Kondisi Cuaca
st.write("### Jumlah Sepeda yang Disewa Berdasarkan Kondisi Cuaca")
data_bicyle['Weather_category'] = pd.cut(data_bicyle['weathersit'], bins=4, labels=['Clear','Mist','Light Snow','Heavy Rain'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Weather_category', y='cnt', data=data_bicyle, errorbar=None, palette='viridis', ax=ax)
ax.set_title('Jumlah Sepeda yang Disewa Berdasarkan Kondisi Cuaca')
ax.set_xlabel('Kondisi Cuaca')
ax.set_ylabel('Rata-rata Jumlah Sepeda yang Disewa (cnt)')
st.pyplot(fig)

# Visualisasi Jumlah Sepeda yang Disewa Berdasarkan Suhu
st.write("### Jumlah Sepeda yang Disewa Berdasarkan Suhu")
data_bicyle['temp_category'] = pd.cut(data_bicyle['temp'], bins=5, labels=['Sangat Dingin', 'Dingin', 'Sedang', 'Hangat', 'Panas'])
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='temp_category', y='cnt', data=data_bicyle, errorbar=None, palette='coolwarm', ax=ax)
ax.set_title('Jumlah Sepeda yang Disewa Berdasarkan Suhu')
ax.set_xlabel('Kategori Suhu')
ax.set_ylabel('Rata-rata Jumlah Sepeda yang Disewa (cnt)')
st.pyplot(fig)

# Visualisasi Jumlah Sepeda yang Disewa Berdasarkan Hari Kerja
st.write("### Jumlah Sepeda yang Disewa Berdasarkan Hari Kerja")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='workingday', y='cnt', data=data_bicyle, errorbar=None, palette='gist_ncar', ax=ax)
ax.set_title('Jumlah Sepeda yang Disewa Berdasarkan Hari Kerja')
ax.set_xlabel('Hari Kerja (0: Bukan Hari Kerja, 1: Hari Kerja)')
ax.set_ylabel('Rata-rata Jumlah Sepeda yang Disewa (cnt)')
st.pyplot(fig)

# Kategorisasi Jam Berdasarkan Pola Penyewaan Sepeda
st.write("### Kategorisasi Jam Berdasarkan Pola Penyewaan Sepeda")
avg_cnt_per_hour = data_bicyle.groupby('hr')['cnt'].mean().reset_index()
percentile_25 = avg_cnt_per_hour['cnt'].quantile(0.25)
percentile_75 = avg_cnt_per_hour['cnt'].quantile(0.75)

def categorize_hour(cnt):
    if cnt >= percentile_75:
        return 'Jam Sibuk'
    elif cnt <= percentile_25:
        return 'Jam Sepi'
    else:
        return 'Jam Menengah'

avg_cnt_per_hour['kategori_jam'] = avg_cnt_per_hour['cnt'].apply(categorize_hour)

fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='hr', y='cnt', hue='kategori_jam', data=avg_cnt_per_hour, palette='viridis', ax=ax)
ax.set_title('Kategorisasi Jam Berdasarkan Pola Penyewaan Sepeda')
ax.set_xlabel('Jam (hr)')
ax.set_ylabel('Rata-rata Jumlah Sepeda yang Disewa (cnt)')
ax.legend(title='Kategori Jam')
st.pyplot(fig)

# Analisis RFM
st.subheader("Analisis RFM")
data_bicyle['dteday'] = pd.to_datetime(data_bicyle['dteday'])
rfm_df = data_bicyle.groupby('dteday').agg({
    'instant': 'count',  # Frequency (jumlah hari dengan penyewaan)
    'cnt': 'sum'  # Monetary (total penyewaan sepeda)
}).reset_index()

rfm_df['recency'] = (data_bicyle['dteday'].max() - rfm_df['dteday']).dt.days
rfm_df.rename(columns={
    'instant': 'frequency',
    'cnt': 'monetary'
}, inplace=True)

# Hitung RFM Score
rfm_df['RFM_score'] = (0.15 * rfm_df['r_rank_norm'] + 0.28 * rfm_df['f_rank_norm'] + 0.57 * rfm_df['m_rank_norm']) * 0.05
rfm_df = rfm_df.round(2)

# Segmentasi berdasarkan skor RFM
rfm_df['customer_segment'] = np.where(
    rfm_df['RFM_score'] > 4.5, "Top days", np.where(
        rfm_df['RFM_score'] > 4, "High activity days", np.where(
            rfm_df['RFM_score'] > 3, "Medium activity days", np.where(
                rfm_df['RFM_score'] > 1.6, 'Low activity days', 'Inactive days'))))

# Visualisasi segmentasi
segment_counts = rfm_df['customer_segment'].value_counts().reset_index()
segment_counts.columns = ['customer_segment', 'count']

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#72BCD4", "#72BCD4", "#D3D3D3", "#D3D3D3", "#D3D3D3"]

sns.barplot(
    x='count',
    y='customer_segment',
    data=segment_counts.sort_values(by='count', ascending=False),
    palette=colors,
    ax=ax
)
ax.set_title("Number of Days in Each Activity Segment", fontsize=15)
ax.set_ylabel(None)
ax.set_xlabel("Number of Days")
ax.tick_params(axis='y', labelsize=12)
st.pyplot(fig)

# Tampilkan hasil RFM
st.write("Hasil RFM:")
st.write(rfm_df.head(20))

# Conclusion
st.subheader("Kesimpulan")
st.write("""
- **Pertanyaan 1**: Berdasarkan analisis, terdapat hubungan positif antara suhu dan jumlah penyewaan sepeda. Model prediktif dapat dibangun menggunakan variabel cuaca, suhu, dan status hari kerja untuk memprediksi jumlah penyewaan sepeda pada jam-jam tertentu.
- **Pertanyaan 2**: Pola penyewaan sepeda menunjukkan dua periode puncak dalam sehari, yaitu pagi dan sore hari. Suhu yang nyaman juga memengaruhi peningkatan jumlah penyewaan.
""")
