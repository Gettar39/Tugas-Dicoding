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

# Judul Aplikasi
st.title("Analisis RFM untuk Penyewaan Sepeda")

# Load Data
data_bicycle = pd.read_csv("data.csv")  # Pastikan dataset sudah di-load
data_bicycle['dteday'] = pd.to_datetime(data_bicycle['dteday'])

# Hitung RFM
rfm_df = data_bicycle.groupby('dteday').agg({
    'instant': 'count',  # Frequency (jumlah hari dengan penyewaan)
    'cnt': 'sum'  # Monetary (total penyewaan sepeda)
}).reset_index()

rfm_df['recency'] = (data_bicycle['dteday'].max() - rfm_df['dteday']).dt.days
rfm_df.rename(columns={'instant': 'frequency', 'cnt': 'monetary'}, inplace=True)

# Hitung RFM Score
rfm_df['r_rank'] = rfm_df['recency'].rank(ascending=False)
rfm_df['r_rank_norm'] = (rfm_df['r_rank'] / rfm_df['r_rank'].max()) * 100
rfm_df['f_rank'] = rfm_df['frequency'].rank(ascending=True)
rfm_df['f_rank_norm'] = (rfm_df['f_rank'] / rfm_df['f_rank'].max()) * 100
rfm_df['m_rank'] = rfm_df['monetary'].rank(ascending=True)
rfm_df['m_rank_norm'] = (rfm_df['m_rank'] / rfm_df['m_rank'].max()) * 100

rfm_df['RFM_score'] = (0.15 * rfm_df['r_rank_norm'] + 0.28 * rfm_df['f_rank_norm'] + 0.57 * rfm_df['m_rank_norm']) * 0.05
rfm_df = rfm_df.round(2)

# Segmentasi berdasarkan skor RFM
rfm_df['customer_segment'] = np.where(
    rfm_df['RFM_score'] > 4.5, "Top days", np.where(
        rfm_df['RFM_score'] > 4, "High activity days", np.where(
            rfm_df['RFM_score'] > 3, "Medium activity days", np.where(
                rfm_df['RFM_score'] > 1.6, 'Low activity days', 'Inactive days'))))

# Visualisasi Segmentasi
st.subheader("Segmentasi Aktivitas Berdasarkan RFM")
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

# Tampilkan Hasil RFM
st.subheader("Hasil RFM")
st.write(rfm_df.head(20))

# Judul Aplikasi
st.title("Analisis Cluster untuk Penyewaan Sepeda")

# Load Data
data_bicycle = pd.read_csv("data.csv")  # Pastikan dataset sudah di-load

# Binning pada kolom 'temp'
bins_temp = [0, 0.25, 0.5, 0.75, 1.0]
labels_temp = ['Low', 'Medium', 'High', 'Very High']
data_bicycle['temp_bin'] = pd.cut(data_bicycle['temp'], bins=bins_temp, labels=labels_temp)

# Binning pada kolom 'cnt'
bins_cnt = [0, 100, 200, 1000]
labels_cnt = ['Low', 'Medium', 'High']
data_bicycle['cnt_bin'] = pd.cut(data_bicycle['cnt'], bins=bins_cnt, labels=labels_cnt)

# Visualisasi Binning pada 'temp'
st.subheader("Distribusi Binning Suhu")
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x='temp_bin', data=data_bicycle, order=labels_temp, palette='viridis', ax=ax)
ax.set_title('Distribution of Temperature Bins')
ax.set_xlabel('Temperature Bin')
ax.set_ylabel('Count')
st.pyplot(fig)

# Visualisasi Scatter Plot Suhu vs Total Rentals
st.subheader("Scatter Plot Suhu vs Total Rentals")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(x='temp', y='cnt', data=data_bicycle, hue='temp_bin', palette='viridis', ax=ax)
ax.set_title('Scatter Plot of Temperature vs Total Rentals')
ax.set_xlabel('Temperature')
ax.set_ylabel('Total Rentals')
st.pyplot(fig)

# Visualisasi Binning pada 'cnt'
st.subheader("Distribusi Binning Total Rentals")
fig, ax = plt.subplots(figsize=(10, 5))
sns.countplot(x='cnt_bin', data=data_bicycle, order=labels_cnt, palette='magma', ax=ax)
ax.set_title('Distribution of Total Rentals (cnt) Bins')
ax.set_xlabel('Total Rentals Bin')
ax.set_ylabel('Count')
st.pyplot(fig)

# Visualisasi Scatter Plot Total Rentals vs Temperature
st.subheader("Scatter Plot Total Rentals vs Temperature")
fig, ax = plt.subplots(figsize=(10, 5))
sns.scatterplot(x='cnt', y='temp', data=data_bicycle, hue='cnt_bin', palette='magma', ax=ax)
ax.set_title('Scatter Plot of Total Rentals vs Temperature')
ax.set_xlabel('Total Rentals')
ax.set_ylabel('Temperature')
st.pyplot(fig)
