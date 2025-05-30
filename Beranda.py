# app.py
import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils.layout import apply_global_styles, render_page_header, render_sidebar_footer

st.set_page_config(page_title="Beranda • Deteksi Awan AI", layout="wide")
apply_global_styles()
render_page_header("🏡 Aplikasi Deteksi Awan Berbasis AI")

st.markdown("""**Selamat datang** di aplikasi deteksi awan berbasis *artificial intelligence*!
            
Aplikasi ini dirancang untuk membantu Anda menghitung **tutupan awan** dan mengidentifikasi **jenis awan** secara otomatis dari gambar citra langit.  
Didukung oleh model AI canggih, sistem ini cocok digunakan oleh pengamat cuaca, peneliti, maupun pengguna umum yang ingin memahami kondisi langit secara visual.

Unggah gambar langit Anda, dan dapatkan hasil analisisnya dalam hitungan detik! 🌤️""")
st.markdown("---")

st.markdown("""
<div style="background-color:#f9f9f9; border-left:5px solid #3b77d9; padding:1.2rem; border-radius:10px;">
    <h4>🔧 Fitur Unggulan:</h4>
    <ul>
      <li>📌 Segmentasi awan menggunakan model <strong>CloudDeepLabV3+</strong></li>
      <li>🧠 Klasifikasi awan menggunakan model <strong>YOLOv8</strong></li>
      <li>✍️ ROI fleksibel: <strong>Otomatis atau Manual</strong></li>
      <li>📄 Ekspor hasil ke <strong>PDF & CSV</strong></li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; margin-top:2em;'>
    <a href="/Deteksi_Awan" target="_self">
        <button style='font-size:1.1em; padding:0.6em 1.5em; background-color:#4CAF50; color:white; border:none; border-radius:6px;'>
            🚀 Mulai Deteksi Awan Sekarang
        </button>
    </a>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

history_path = "temps/history/riwayat.csv"
jumlah_data = 0

if os.path.exists(history_path) and os.stat(history_path).st_size > 0:
    df_history = pd.read_csv(history_path)
    jumlah_data = len(df_history)

    # Tiga kolom: metric + dua grafik interaktif
    col1, col2, col3 = st.columns([0.3, 0.35, 0.355])

    with col1:
        st.markdown("#### 🖼️ Gambar Terproses")
        st.markdown(f"""
            <div style="font-size:72px; font-weight:normal; color:#222; text-align:center; margin-top:0.5em;">
                {jumlah_data}
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("#### 📊 Distribusi Tutupan Awan (%)")
        fig1 = px.histogram(
            df_history,
            x="coverage",
            nbins=10,
            title="",
            color_discrete_sequence=["skyblue"]
        )
        fig1.update_layout(
            xaxis_title="Tutupan Awan (%)",
            yaxis_title="Jumlah Gambar",
            margin=dict(l=10, r=10, t=30, b=10),
            height=250
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col3:
        st.markdown("#### 🌥️ Komposisi Jenis Awan")
        fig2 = px.pie(
            df_history,
            names="jenis_awan",
            title="",
            hole=0.4,  # donat
            color_discrete_sequence=px.colors.sequential.dense
        )
        fig2.update_traces(textinfo="percent+label")
        fig2.update_layout(
            showlegend=True,
            margin=dict(l=10, r=10, t=30, b=10),
            height=250
        )
        st.plotly_chart(fig2, use_container_width=True)

else:
    col1, col2 = st.columns(2)
    col1.metric("🖼️ Gambar Terproses", "0")
    col2.metric("🌥️ Statistik Awan", "Belum Ada Data")

render_sidebar_footer()