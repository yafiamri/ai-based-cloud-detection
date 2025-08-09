# utils/layout.py
import streamlit as st
import os
import base64
import mimetypes
from typing import Tuple
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import plotly.express as px

# Impor konfigurasi terpusat yang sudah dimuat
from .config import config

# Ambil seksi konfigurasi yang relevan untuk mempermudah akses
PATHS = config.get('paths', {})
UI_CONFIG = config.get('ui', {})
APP_CONFIG = config.get('app', {})
ANALYSIS_CONFIG = config.get('analysis', {})


@st.cache_data
def _get_image_as_base64(file_path: str) -> Tuple[str, str]:
    """
    Membaca file gambar, mengonversinya ke Base64, dan mendeteksi tipe medianya.
    Mengembalikan tuple (base64_string, mime_type).
    """
    try:
        path = Path(file_path)
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is None:
            # Fallback jika tipe tidak terdeteksi
            mime_type = "image/png"
            
        with path.open("rb") as f:
            data = f.read()
        
        base64_str = base64.b64encode(data).decode()
        return base64_str, mime_type
    except FileNotFoundError:
        st.error(f"File gambar latar tidak ditemukan di: {file_path}")
        return "", ""


def apply_global_styles():
    """
    Menerapkan gaya CSS global ke aplikasi Streamlit.
    Semua nilai diambil dari file config.yaml untuk kemudahan kustomisasi.
    """
    theme = UI_CONFIG.get('theme', {})
    
    # CSS dasar untuk tema
    css = f"""
        /* Variabel Warna Utama dari config.yaml */
        :root {{
            --primary: {theme.get('primary_color', '#1f77b4')};
            --secondary: {theme.get('secondary_color', '#2ca02c')};
            --card-bg: {theme.get('card_bg_light', "#f8f9fa84")};
            --card-border: {theme.get('card_border_light', "#dee2e684")};
        }}
        
        /* Mode Gelap */
        @media (prefers-color-scheme: dark) {{
            :root {{
                --primary: {theme.get('primary_color_dark', '#6fa8dc')};
                --secondary: {theme.get('secondary_color_dark', '#93c47d')};
                --card-bg: {theme.get('card_bg_dark', "#161b2284")};
                --card-border: {theme.get('card_border_dark', "#30363d84")};
            }}
        }}

        /* Gaya Kartu Navigasi */
        .nav-card {{
            display: flex; flex-direction: column; justify-content: space-between;
            height: 100%; padding: 1.5rem 1.2rem; border-radius: 10px;
            background-color: var(--card-bg); border: 1px solid var(--card-border);
            border-top: 4px solid var(--primary); transition: transform 0.2s, box-shadow 0.2s;
            text-align: center;
        }}
        .nav-card:hover {{ transform: translateY(-5px); box-shadow: 0 8px 30px rgba(0,0,0,0.1); }}

        /* Gaya Tombol di dalam Kartu Navigasi */
        .nav-card a.nav-button-link {{
            display: block; text-align: center; font-weight: bold;
            padding: 0.7rem; margin-top: 1rem; background-color: var(--primary);
            color: white !important; border-radius: 5px; text-decoration: none;
            transition: background-color 0.2s;
        }}
        .nav-card a.nav-button-link:hover {{ background-color: var(--secondary); }}

        /* Gaya untuk caption gambar/video yang rata tengah */
        .centered-caption {{
            text-align: center;
            color: grey;
            font-size: 0.9em;
        }}
    """

    # Cek apakah path background ada di config
    background_image_path = PATHS.get('background')
    if background_image_path:
        # Jika ada, coba konversi ke Base64
        base64_img, mime_type = _get_image_as_base64(background_image_path)
        # HANYA JIKA Base64 berhasil dibuat, tambahkan CSS background
        if base64_img:
            # Gunakan mime type untuk memastikan kompatibilitas
            css += f"""
                .stApp {{
                    background-image: url("data:{mime_type};base64,{base64_img}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}
                [data-testid="stHeader"] {{
                    background-color: rgba(0, 0, 0, 0);
                }}
            """

    # Terapkan CSS final ke aplikasi
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_page_header(title: str) -> None:
    """
    Menampilkan header halaman yang konsisten dengan logo dan judul.

    Args:
        title (str): Judul halaman yang akan ditampilkan.
    """
    logo_path = PATHS.get('logo', '')
    try:
        with open(logo_path, "rb") as f:
            b64_logo = base64.b64encode(f.read()).decode()
            logo_src = f"data:image/png;base64,{b64_logo}"
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:1.5rem;">
              <img src="{logo_src}" width="60" style="margin-right:15px;"/>
              <h1 style="margin:0; color:var(--primary); font-size: 3rem;">{title}</h1>
            </div>
            """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"File logo tidak ditemukan di path: '{logo_path}'. Hanya menampilkan judul.")
        st.header(title)


def render_sidebar_footer() -> None:
    """
    Menampilkan footer yang menempel (sticky) dan adaptif di bagian bawah sidebar.
    Menggunakan position:fixed dengan lebar yang dihitung.
    """
    author = APP_CONFIG.get('author', 'Pengguna')
    affiliation = APP_CONFIG.get('affiliation', '')
    
    st.sidebar.markdown(f"""
    <style>
        .sidebar-footer {{
            position: fixed;
            bottom: 0;
            width: inherit;
            max-width: 50vw; /* Lebar maksimal adalah 50% dari lebar layar */
            padding: 1rem;
            border-top: 1px solid var(--card-border);
            text-align: center;
            z-index: 100;
        }}
    </style>
    
    <div class="sidebar-footer">
        <p style="font-size:0.75rem; color:var(--primary);">
            🧑‍💻 Dikembangkan oleh <b>{author}</b><br>
            Mahasiswa {affiliation} 2021
        </p>
    </div>
    """, unsafe_allow_html=True)


def section_divider(text: str, emoji: str = "📌") -> None:
    """
    Membuat pembatas visual antar-seksi yang stylish dengan teks di tengah
    dan garis gradien di kedua sisinya.

    Args:
        text (str): Teks judul untuk seksi tersebut.
        emoji (str, optional): Emoji yang akan ditampilkan di samping judul.
    """
    st.markdown(f"""
    <div style="display: flex; align-items: center; text-align: center; margin-top: 2rem; margin-bottom: 1.5rem;">
      <hr style="flex-grow: 1; height: 2px; border: none; background: linear-gradient(to left, var(--primary), transparent);">
      <h2 style="color:var(--primary); margin: 0 1rem; font-size: 1.75rem; white-space: nowrap;">{emoji} {text}</h2>
      <hr style="flex-grow: 1; height: 2px; border: none; background: linear-gradient(to right, var(--primary), transparent);">
    </div>
    """, unsafe_allow_html=True)


def render_result(result_data: Dict[str, Any]) -> None:
    """
    Menampilkan hasil analisis yang terstruktur untuk satu file.
    Fungsi ini robust terhadap file media yang mungkin hilang.

    Args:
        result_data (Dict[str, Any]): Sebuah dictionary yang berisi satu baris data hasil
                                      analisis dari database.
    """
    placeholder_path = PATHS.get('placeholder', '')
    original_path = result_data.get('original_path', '')
    overlay_path = result_data.get('overlay_path', '')

    # Cek apakah path valid dan filenya ada di disk sebelum digunakan
    display_original = original_path if original_path and os.path.exists(original_path) else placeholder_path
    display_overlay = overlay_path if overlay_path and os.path.exists(overlay_path) else placeholder_path
    
    # Tentukan tipe media berdasarkan path jika ada, jika tidak anggap bukan video    
    video_extensions = tuple(ANALYSIS_CONFIG.get('video_extensions', []))
    is_video = original_path and original_path.lower().endswith(video_extensions)

    st.subheader(f"🖼️ Hasil Analisis: {result_data.get('source_filename', 'N/A')}")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if is_video:
            st.video(display_original)
        else:
            st.image(display_original, use_container_width=True, caption="Citra Asli")
    with col_g2:
        if is_video:
            st.video(display_overlay)
        else:
            st.image(display_overlay, use_container_width=True, caption="*Overlay* Segmentasi")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.subheader("📊 Metrik Hasil Segmentasi")
        st.metric(label="**Tutupan Awan:**", value=f"{result_data.get('cloud_coverage', 0):.2f}%")
        st.metric(label="**Nilai Okta:**", value=f"{result_data.get('okta_value', 0)} / 8")
        st.metric(label="**Kondisi Langit:**", value=f"{result_data.get('sky_condition', 'N/A')}")
    
    with col_t2:
        st.subheader("🌥️ Jenis Awan Terdeteksi")
        st.metric(label="**Jenis Dominan:**", value=result_data.get('dominant_cloud_type', 'Tidak Terdeteksi'))
        st.markdown(f"**Keyakinan Klasifikasi:**")
        details_string = result_data.get('classification_details', 'N/A')
        # Cek jika ada detail untuk ditampilkan
        if details_string and details_string != 'N/A':
            # 1. Pecah string menjadi list berdasarkan '; '
            details_list = details_string.split('; ')
            # 2. Tampilkan semua sebagai bullet, termasuk info jumlah frame
            markdown_list = "\n".join([f"- {item}" for item in details_list])
            st.markdown(markdown_list)
        else:
            st.markdown(details_string)
    st.markdown("---")

    st.markdown("<br>", unsafe_allow_html=True)

# --- FUNGSI UNTUK DASBOR STATISTIK ---
def render_summary_dashboard(data: pd.DataFrame, title: str = "Dasbor Statistik"):
    """
    Merender dasbor statistik universal dengan gaya visual yang kaya.
    Menggabungkan metrik besar dengan fleksibilitas dinamis.

    Args:
        data (pd.DataFrame): DataFrame yang berisi riwayat analisis.
        title (str): Judul yang akan ditampilkan untuk seksi dasbor.
    """
    section_divider(title, "📈")

    # Menggunakan pesan yang lebih spesifik jika data kosong
    if data.empty:
        st.info("Belum ada riwayat analisis. Dasbor akan muncul di sini setelah analisis pertama Anda.")
        return

    # Hitung metrik utama
    jumlah_total = len(data)
    type_counts = data['media_type'].value_counts()
    
    # Siapkan layout kolom
    dash_col1, dash_col2, dash_col3 = st.columns([0.3, 0.35, 0.35])

    with dash_col1:
        st.markdown("#### 🖼️ Total Citra Dianalisis")
        
        # Buat rincian (breakdown) per tipe media secara dinamis
        breakdown_html = ""
        for media_type, count in type_counts.items():
            type_name = media_type.replace('_', ' ').title()
            breakdown_html += f"<strong>{count}</strong> {type_name}{'s' if count > 1 else ''}<br>"

        # Gunakan st.markdown dengan f-string untuk memasukkan metrik ke HTML
        st.markdown(f"""
        <div style="text-align:center;">
            <p style="font-size:96px; font-weight:normal; color:var(--primary); margin: 0;">{jumlah_total}</p>
            <p style="font-size:18px; color:var(--secondary); margin-top:10px;">
                {breakdown_html}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with dash_col2:
        st.markdown("#### 📊 Distribusi Tutupan Awan")
        fig1 = px.histogram(data, x="cloud_coverage", nbins=10, 
                            labels={"cloud_coverage": "Tutupan Awan (%)"},
                            color_discrete_sequence=[UI_CONFIG.get('theme', {}).get('primary_color', '#1f77b4')])
        fig1.update_layout(yaxis_title="Jumlah", margin=dict(l=10, r=10, t=10, b=10), height=250)
        st.plotly_chart(fig1, use_container_width=True)

    with dash_col3:
        st.markdown("#### 🌥️ Komposisi Jenis Awan")
        # Mengatasi error jika 'dominant_cloud_type' kosong
        if 'dominant_cloud_type' in data and not data['dominant_cloud_type'].empty:
            fig2 = px.pie(data, names="dominant_cloud_type", hole=0.4,
                          labels={"dominant_cloud_type": "Jenis Awan Dominan"},
                          color_discrete_sequence=px.colors.qualitative.Set1)
            fig2.update_traces(textinfo='percent+label', showlegend=False)
            fig2.update_layout(margin=dict(l=10, r=10, t=10, b=30), height=250)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Data jenis awan belum cukup untuk ditampilkan.")