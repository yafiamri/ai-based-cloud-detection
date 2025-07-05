# Beranda.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import time
import glob

from utils.layout import apply_global_styles, render_page_header, render_sidebar_footer, section_divider

def cleanup_temp_files(directory="temps", age_hours=1):
    """
    Menghapus file-file di dalam folder 'temps' yang lebih tua dari
    batas usia yang ditentukan (dalam jam).
    """
    try:
        age_seconds = age_hours * 3600
        now = time.time()
        
        files_to_check = glob.glob(os.path.join(directory, "*.pdf")) + glob.glob(os.path.join(directory, "*.zip"))

        for file_path in files_to_check:
            if os.path.isfile(file_path):
                file_age = now - os.path.getmtime(file_path)
                if file_age > age_seconds:
                    os.remove(file_path)
                    print(f"Menghapus file sementara yang sudah lama: {file_path}") # melihat log di terminal
    except Exception as e:
        print(f"Error saat membersihkan file sementara: {e}")

# Panggil fungsi pembersihan saat aplikasi dimulai
cleanup_temp_files()

# --- 1. Konfigurasi Halaman & Gaya ---
st.set_page_config(page_title="Beranda", layout="wide")
apply_global_styles()
render_sidebar_footer()

# --- 2. Render Bagian Atas Halaman (dari Proyek 2) ---
render_page_header("Aplikasi Deteksi Awan Berbasis AI")

st.write("**Selamat datang** di sistem aplikasi pendeteksian awan berbasis *artificial intelligence*!")

section_divider("Menganalisis Langit, Memahami Cuaca", "🌤️")
st.write("Aplikasi ini dirancang untuk membantu Anda menghitung **tutupan awan** dan mengidentifikasi **jenis awan** secara otomatis dari gambar atau video citra langit. Didukung oleh model AI canggih, sistem ini cocok digunakan oleh pengamat cuaca, peneliti, maupun pengguna umum yang ingin memahami kondisi langit secara visual.")

# Kartu Fitur Unggulan (dari Proyek 2)
st.markdown("""
<div style="background-color:var(--card-bg); border-left:5px solid var(--primary); padding:1.2rem; border-radius:8px; margin: 1.5rem 0;">
    <h4>🔧 Fitur Unggulan</h4>
    <ul>
        <li>📌 Segmentasi awan menggunakan model <strong>CloudDeepLabV3+</strong></li>
        <li>🧠 Klasifikasi jenis awan menggunakan model <strong>YOLOv8</strong></li>
        <li>🖼️ Dukungan analisis untuk <strong>gambar statis dan video</strong></li>
        <li>✍️ ROI fleksibel: <strong>Otomatis atau Manual</strong> untuk akurasi maksimal</li>
        <li>📄 Ekspor hasil analisis ke format <strong>PDF, CSV, dan ZIP</strong></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Kartu Navigasi (dari Proyek 2, dengan 4 kartu)
section_divider("Mulai Menjelajah", "🧭")
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    <div class="nav-card">
        <div>
            <h4>☁️ Deteksi Awan</h4>
            <p style='font-size:0.9em; opacity:0.8;'>Unggah gambar atau video untuk menganalisis awan.</p>
        </div>
        <a href="/Deteksi_Awan" target="_self" class="nav-button-link">Mulai Analisis</a>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="nav-card">
        <div>
            <h4>📜 Riwayat Analisis</h4>
            <p style='font-size:0.9em; opacity:0.8;'>Lihat & kelola semua hasil analisis Anda.</p>
        </div>
        <a href="/Riwayat_Analisis" target="_self" class="nav-button-link">Lihat Riwayat</a>
    </div>
    """, unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    st.markdown("""
    <div class="nav-card">
        <div>
            <h4>🧠 Penjelasan Model</h4>
            <p style='font-size:0.9em; opacity:0.8;'>Pelajari arsitektur model AI yang digunakan.</p>
        </div>
        <a href="/Penjelasan_Model" target="_self" class="nav-button-link">Lihat Model</a>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="nav-card">
        <div>
            <h4>🧑‍💻 Tentang Pengembang</h4>
            <p style='font-size:0.9em; opacity:0.8;'>Kenali lebih jauh siapa di balik aplikasi ini.</p>
        </div>
        <a href="/Tentang_Pengembang" target="_self" class="nav-button-link">Lihat Profil</a>
    </div>
    """, unsafe_allow_html=True)

# --- 3. Render Dasbor Statistik (dari Proyek 1) ---
section_divider("Dasbor Statistik Analisis", "📈")
history_path = "temps/history/riwayat.csv"

if os.path.exists(history_path) and os.path.getsize(history_path) > 0:
    try:
        df_history = pd.read_csv(history_path)
        jumlah_data = len(df_history)

        image_files = [f for f in df_history['nama_file'] if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.webp']]
        video_files = [f for f in df_history['nama_file'] if os.path.splitext(f)[1].lower() in ['.avi', '.mov', '.mp4', '.mpeg4']]
        
        jumlah_gambar = len(image_files)
        jumlah_video = len(video_files)

        # Tiga kolom: metrik + dua grafik interaktif
        dash_col1, dash_col2, dash_col3 = st.columns([0.3, 0.35, 0.35])

        with dash_col1:
            st.markdown("#### 🖼️ Total Citra Terproses")
            st.markdown(f"""
            <div style="text-align:center;">
                <p style="font-size:96px; font-weight:normal; color:var(--primary); margin: 0 0 0 0;">
                    {jumlah_data}
                </p>
                <p style="font-size:24px; color:grey; margin-top:10px;">
                    <strong>{jumlah_gambar}</strong> gambar<br><strong>{jumlah_video}</strong> video
                </p>
            </div>
            """, unsafe_allow_html=True)

        with dash_col2:
            st.markdown("#### 📊 Distribusi Tutupan Awan")
            fig1 = px.histogram(df_history, x="coverage", nbins=10, color_discrete_sequence=["skyblue"])
            fig1.update_layout(xaxis_title="Tutupan Awan (%)", yaxis_title="Jumlah Analisis", margin=dict(l=10, r=10, t=30, b=10), height=250)
            st.plotly_chart(fig1, use_container_width=True)

        with dash_col3:
            st.markdown("#### 🌥️ Komposisi Jenis Awan")
            fig2 = px.pie(df_history, names="jenis_awan", hole=0.4, color_discrete_sequence=px.colors.sequential.dense)
            fig2.update_traces(textinfo='percent+label', showlegend=True)
            fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=250)
            st.plotly_chart(fig2, use_container_width=True)

    except pd.errors.EmptyDataError:
        st.info("File riwayat kosong. Dasbor akan muncul di sini setelah analisis pertama Anda.")
else:
    st.info("Belum ada riwayat analisis. Dasbor akan muncul di sini setelah analisis pertama Anda.")