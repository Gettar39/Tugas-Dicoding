import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Dashboard/hour.csv")


# Konfigurasi halaman
st.set_page_config(page_title="Analisis Data Penyewaan Sepeda", layout="wide")

# Header
st.title("📊 Analisis Data Penyewaan Sepeda")

# Sidebar untuk Upload File
st.sidebar.header("Unggah Dataset")
uploaded_file = st.sidebar.file_uploader("Pilih file CSV", type=["csv"])

if uploaded_file:
    # Membaca dataset
    data_bicyle = pd.read_csv(uploaded_file)
    
    # Menampilkan preview data
    st.subheader("📋 Cuplikan Dataset")
    st.write(data_bicyle.head())

    # Menampilkan informasi dataset
    st.subheader("📌 Informasi Dataset")
    buffer = []
    data_bicyle.info(buf=buffer)
    info_str = "\n".join(buffer)
    st.text(info_str)

    # Menampilkan Statistik Deskriptif
    st.subheader("📊 Statistik Deskriptif")
    st.write(data_bicyle.describe())

    # Visualisasi Data
    st.subheader("📈 Visualisasi Data")

    # Pilihan visualisasi
    vis_option = st.selectbox("Pilih Visualisasi", 
        ["Jumlah Penyewa Berdasarkan Cuaca", 
         "Jumlah Penyewa Berdasarkan Suhu", 
         "Jumlah Penyewa Berdasarkan Jam"])

    if vis_option == "Jumlah Penyewa Berdasarkan Cuaca":
        data_bicyle["Weather_category"] = pd.cut(
            data_bicyle["weathersit"], bins=4, labels=["Clear", "Mist", "Light Snow", "Heavy Rain"]
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Weather_category", y="cnt", data=data_bicyle, palette="viridis", ax=ax)
        ax.set_title("Jumlah Penyewa Sepeda Berdasarkan Kondisi Cuaca")
        ax.set_xlabel("Kondisi Cuaca")
        ax.set_ylabel("Rata-rata Jumlah Penyewa Sepeda")
        st.pyplot(fig)

    elif vis_option == "Jumlah Penyewa Berdasarkan Suhu":
        data_bicyle["temp_category"] = pd.cut(
            data_bicyle["temp"], bins=5, labels=["Sangat Dingin", "Dingin", "Sedang", "Hangat", "Panas"]
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="temp_category", y="cnt", data=data_bicyle, palette="coolwarm", ax=ax)
        ax.set_title("Jumlah Penyewa Sepeda Berdasarkan Suhu")
        ax.set_xlabel("Kategori Suhu")
        ax.set_ylabel("Rata-rata Jumlah Penyewa Sepeda")
        st.pyplot(fig)

    elif vis_option == "Jumlah Penyewa Berdasarkan Jam":
        avg_cnt_per_hour = data_bicyle.groupby("hr")["cnt"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x="hr", y="cnt", data=avg_cnt_per_hour, marker="o", ax=ax)
        ax.set_title("Jumlah Penyewa Sepeda Berdasarkan Jam")
        ax.set_xlabel("Jam")
        ax.set_ylabel("Rata-rata Penyewa")
        st.pyplot(fig)

    # Analisis RFM
    st.subheader("📌 Analisis RFM (Recency, Frequency, Monetary)")
    data_bicyle['dteday'] = pd.to_datetime(data_bicyle['dteday'])

    rfm_df = data_bicyle.groupby('dteday').agg({
        'instant': 'count',  # Frequency (jumlah hari dengan penyewaan)
        'cnt': 'sum'  # Monetary (total penyewaan sepeda)
    }).reset_index()

    rfm_df['recency'] = (data_bicyle['dteday'].max() - rfm_df['dteday']).dt.days
    rfm_df.rename(columns={'instant': 'frequency', 'cnt': 'monetary'}, inplace=True)

    # Skor RFM
    rfm_df['RFM_score'] = (rfm_df['recency'].rank(ascending=False) * 0.15 +
                           rfm_df['frequency'].rank(ascending=True) * 0.28 +
                           rfm_df['monetary'].rank(ascending=True) * 0.57) * 0.05

    rfm_df['customer_segment'] = np.where(
        rfm_df['RFM_score'] > 4.5, "Top days", np.where(
            rfm_df['RFM_score'] > 4, "High activity days", np.where(
                rfm_df['RFM_score'] > 3, "Medium activity days", np.where(
                    rfm_df['RFM_score'] > 1.6, 'Low activity days', 'Inactive days'))))

    # Visualisasi RFM
    segment_counts = rfm_df['customer_segment'].value_counts().reset_index()
    segment_counts.columns = ['customer_segment', 'count']

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x='count', y='customer_segment', data=segment_counts.sort_values(by='count', ascending=False), ax=ax)
    ax.set_title("Distribusi Segmentasi RFM")
    ax.set_xlabel("Jumlah Hari")
    ax.set_ylabel("Kategori Aktivitas")
    st.pyplot(fig)

    # Analisis Cluster
    st.subheader("📌 Analisis Cluster")
    
    bins_temp = [0, 0.25, 0.5, 0.75, 1.0]
    labels_temp = ['Low', 'Medium', 'High', 'Very High']
    data_bicyle['temp_bin'] = pd.cut(data_bicyle['temp'], bins=bins_temp, labels=labels_temp)

    bins_cnt = [0, 100, 200, 1000]
    labels_cnt = ['Low', 'Medium', 'High']
    data_bicyle['cnt_bin'] = pd.cut(data_bicyle['cnt'], bins=bins_cnt, labels=labels_cnt)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.countplot(x='temp_bin', data=data_bicyle, order=labels_temp, palette='viridis', ax=axes[0])
    axes[0].set_title('Distribusi Suhu (Temperature Bins)')

    sns.scatterplot(x='temp', y='cnt', data=data_bicyle, hue='temp_bin', palette='viridis', ax=axes[1])
    axes[1].set_title('Scatter Plot Suhu vs Penyewaan')

    st.pyplot(fig)

else:
    st.warning("Silakan unggah file CSV untuk memulai analisis.")
