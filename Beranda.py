# Lokasi: /Beranda.py
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# Impor dari modul-modul arsitektur baru kita
from core.io import file_manager
from ui_components import (
    apply_global_styles, 
    render_page_header, 
    render_sidebar_footer, 
    section_divider
)

# --- 1. Konfigurasi Halaman & Gaya ---
st.set_page_config(
    page_title="Beranda • Deteksi Awan AI",
    page_icon="☁️",
    layout="wide"
)
apply_global_styles()
render_sidebar_footer() # Footer di sidebar tetap ada

# --- 2. Fungsi Helper Tanpa Cache ---
def get_dashboard_data():
    """Membaca dan memproses data riwayat untuk ditampilkan di dasbor."""
    history_records = file_manager.get_history()
    if not history_records:
        return None
    return pd.DataFrame(history_records)

# --- 3. Render Halaman ---
render_page_header("Aplikasi Deteksi Awan Berbasis AI", "logo.png")

# Bagian Pengantar
section_divider("Menganalisis Langit, Memahami Cuaca", "🌤️")
st.markdown("""
**Selamat datang** di aplikasi deteksi awan berbasis *artificial intelligence*!

Aplikasi ini dirancang untuk membantu Anda menghitung **tutupan awan** dan mengidentifikasi **jenis awan** secara otomatis dari gambar citra langit. Didukung oleh model AI canggih, sistem ini cocok digunakan oleh pengamat cuaca, peneliti, maupun pengguna umum yang ingin memahami kondisi langit secara visual.

Unggah citra langit Anda, dan dapatkan hasil analisisnya dalam hitungan detik!
""")

# Kartu Fitur Unggulan
st.markdown("""
<div style="background-color:var(--card-bg); border-left:5px solid var(--primary); padding:1.2rem; border-radius:8px; margin: 1rem 0;">
    <h4>🔧 Fitur Unggulan</h4>
    <ul>
        <li>📌 Segmentasi awan presisi menggunakan model <strong>CloudDeepLabV3+</strong></li>
        <li>🧠 Klasifikasi jenis awan cepat menggunakan model <strong>YOLOv8</strong></li>
        <li>✍️ ROI fleksibel: <strong>Otomatis atau Manual</strong> untuk akurasi maksimal</li>
        <li>📄 Ekspor hasil analisis ke format <strong>PDF, CSV, dan ZIP</strong></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Kartu Navigasi
section_divider("Mulai Menjelajah", "🧭")
col1, col2, col3, col4 = st.columns(4)
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
            <p style='font-size:0.9em; opacity:0.8;'>Lihat & kelola semua hasil analisis Anda sebelumnya.</p>
        </div>
        <a href="/Riwayat_Analisis" target="_self" class="nav-button-link">Lihat Riwayat</a>
    </div>
    """, unsafe_allow_html=True)

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

# Dasbor Statistik
section_divider("Dasbor Statistik Global", "📊")
history_df = get_dashboard_data()

if history_df is None or history_df.empty:
    st.info("Belum ada riwayat analisis. Dasbor akan muncul di sini setelah analisis pertama Anda.")
else:
    total_analyses = len(history_df)
    
    dash_col1, dash_col2, dash_col3 = st.columns([0.25, 0.4, 0.35])

    with dash_col1:
        st.markdown("##### Total Analisis")
        st.markdown(f"<h1 style='text-align: center; color:var(--primary);'>{total_analyses}</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; margin-top:-10px;'>File Terproses</p>", unsafe_allow_html=True)

    with dash_col2:
        st.markdown("##### Distribusi Tutupan Awan")
        fig_hist = px.histogram(history_df, x="cloud_coverage_percent", nbins=8, color_discrete_sequence=[st.get_option("theme.primaryColor")])
        fig_hist.update_layout(xaxis_title="Tutupan Awan (%)", yaxis_title="Jumlah", margin=dict(l=10, r=10, t=30, b=10), height=250)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col3:
        st.markdown("##### Komposisi Jenis Awan")
        chart_df = history_df[history_df['dominant_cloud_type'] != 'N/A'].copy()
        if not chart_df.empty:
            fig_pie = px.pie(chart_df, names="dominant_cloud_type", hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label', showlegend=False)
            fig_pie.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=250)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.caption("Belum ada data klasifikasi.")