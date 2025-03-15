import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt
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

# Convert 'dteday' to datetime
data_bicyle['dteday'] = pd.to_datetime(data_bicyle['dteday'])

# Sidebar for date range selection
st.sidebar.header('Filter Data Weather')
start_date = st.sidebar.date_input('Pilih Tanggal Awal Cuaca', data_bicyle['dteday'].min())
end_date = st.sidebar.date_input('Pilih Tanggal Akhir Cuaca', data_bicyle['dteday'].max())

# Filter data based on selected date range
filtered_data = data_bicyle[(data_bicyle['dteday'] >= pd.to_datetime(start_date)) & 
                            (data_bicyle['dteday'] <= pd.to_datetime(end_date))]

# Visualisasi Jumlah Sepeda yang Disewa Berdasarkan Kondisi Cuaca
st.write(f"### Jumlah Sepeda yang Disewa Berdasarkan Kondisi Cuaca dari {start_date} hingga {end_date}")
filtered_data['Weather_category'] = pd.cut(filtered_data['weathersit'], bins=4, labels=['Clear','Mist','Light Snow','Heavy Rain'])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='Weather_category', y='cnt', data=filtered_data, errorbar=None, palette='viridis', ax=ax)
ax.set_title(f'Jumlah Sepeda yang Disewa Berdasarkan Kondisi Cuaca dari {start_date} hingga {end_date}')
ax.set_xlabel('Kondisi Cuaca')
ax.set_ylabel('Rata-rata Jumlah Sepeda yang Disewa (cnt)')
st.pyplot(fig)

# Tampilkan data yang difilter
st.write("### Data yang Difilter")
st.dataframe(filtered_data)


# Sidebar for date range selection
st.sidebar.header('Filter Data Suhu')
start_date = st.sidebar.date_input('Pilih Tanggal Awal Suhu', data_bicyle['dteday'].min())
end_date = st.sidebar.date_input('Pilih Tanggal Akhir Suhu', data_bicyle['dteday'].max())

# Filter data based on selected date range
filtered_data = data_bicyle[(data_bicyle['dteday'] >= pd.to_datetime(start_date)) & 
                            (data_bicyle['dteday'] <= pd.to_datetime(end_date))]

# Visualisasi Jumlah Sepeda yang Disewa Berdasarkan Suhu
st.write(f"### Jumlah Sepeda yang Disewa Berdasarkan Suhu dari {start_date} hingga {end_date}")

# Kategorisasi suhu
filtered_data['temp_category'] = pd.cut(filtered_data['temp'], bins=5, labels=['Sangat Dingin', 'Dingin', 'Sedang', 'Hangat', 'Panas'])

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='temp_category', y='cnt', data=filtered_data, errorbar=None, palette='coolwarm', ax=ax)
ax.set_title(f'Jumlah Sepeda yang Disewa Berdasarkan Suhu dari {start_date} hingga {end_date}')
ax.set_xlabel('Kategori Suhu')
ax.set_ylabel('Rata-rata Jumlah Sepeda yang Disewa (cnt)')
st.pyplot(fig)

# Tampilkan data yang difilter
st.write("### Data yang Difilter")
st.dataframe(filtered_data)

#Convert 'dteday' to datetime
data_bicyle['dteday'] = pd.to_datetime(data_bicyle['dteday'])

# Sidebar for date range selection
st.sidebar.header('Filter Data Hari Kerja')
start_date = st.sidebar.date_input('Pilih Tanggal Awal Hari Kerja', data_bicyle['dteday'].min(), key='start_date_workingday')
end_date = st.sidebar.date_input('Pilih Tanggal Akhir Hari Kerja', data_bicyle['dteday'].max(), key='end_date_workingday')

# Filter data based on selected date range
filtered_data = data_bicyle[(data_bicyle['dteday'] >= pd.to_datetime(start_date)) & 
                            (data_bicyle['dteday'] <= pd.to_datetime(end_date))]

# Visualisasi Jumlah Sepeda yang Disewa Berdasarkan Hari Kerja
st.write(f"### Jumlah Sepeda yang Disewa Berdasarkan Hari Kerja dari {start_date} hingga {end_date}")

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='workingday', y='cnt', data=filtered_data, errorbar=None, palette='gist_ncar', ax=ax)
ax.set_title(f'Jumlah Sepeda yang Disewa Berdasarkan Hari Kerja dari {start_date} hingga {end_date}')
ax.set_xlabel('Hari Kerja (0: Bukan Hari Kerja, 1: Hari Kerja)')
ax.set_ylabel('Rata-rata Jumlah Sepeda yang Disewa (cnt)')
st.pyplot(fig)

# Tampilkan data yang difilter
st.write("### Data yang Difilter")
st.dataframe(filtered_data)

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

# Pilihan visualisasi
visualization_option = st.selectbox(
    "Pilih jenis visualisasi:",
    ("Distribusi Binning Suhu", "Scatter Plot Suhu vs Total Rentals")
)

# Visualisasi Binning pada 'temp'
if visualization_option == "Distribusi Binning Suhu":
    st.subheader("Distribusi Binning Suhu")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='temp_bin', data=data_bicycle, order=labels_temp, palette='viridis', ax=ax)
    ax.set_title('Distribution of Temperature Bins')
    ax.set_xlabel('Temperature Bin')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Visualisasi Scatter Plot Suhu vs Total Rentals
elif visualization_option == "Scatter Plot Suhu vs Total Rentals":
    st.subheader("Scatter Plot Suhu vs Total Rentals")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x='temp', y='cnt', data=data_bicycle, hue='temp_bin', palette='viridis', ax=ax)
    ax.set_title('Scatter Plot of Temperature vs Total Rentals')
    ax.set_xlabel('Temperature')
    ax.set_ylabel('Total Rentals')
    st.pyplot(fig)

# Pilihan visualisasi
visualization_option = st.selectbox(
    "Pilih jenis visualisasi:",
    ("Distribusi Binning Total Rentals", "Scatter Plot Total Rentals vs Temperature")
)

# Visualisasi Binning pada 'cnt'
if visualization_option == "Distribusi Binning Total Rentals":
    st.subheader("Distribusi Binning Total Rentals")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(x='cnt_bin', data=data_bicycle, order=labels_cnt, palette='magma', ax=ax)
    ax.set_title('Distribution of Total Rentals (cnt) Bins')
    ax.set_xlabel('Total Rentals Bin')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Visualisasi Scatter Plot Total Rentals vs Temperature
elif visualization_option == "Scatter Plot Total Rentals vs Temperature":
    st.subheader("Scatter Plot Total Rentals vs Temperature")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(x='cnt', y='temp', data=data_bicycle, hue='cnt_bin', palette='magma', ax=ax)
    ax.set_title('Scatter Plot of Total Rentals vs Temperature')
    ax.set_xlabel('Total Rentals')
    ax.set_ylabel('Temperature')
    st.pyplot(fig)
