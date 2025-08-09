# Beranda.py
import streamlit as st
import plotly.express as px

# Impor semua fondasi dari utils
from utils.config import config
from utils.database import get_history_df
from utils.layout import apply_global_styles, render_page_header, render_sidebar_footer, section_divider, render_summary_dashboard
from utils.segmentation import load_segmentation_model
from utils.classification import load_classification_model
from utils.system import cleanup_temp_files

# --- 1. Konfigurasi Halaman & Inisialisasi ---

# Atur konfigurasi halaman dari config.yaml
st.set_page_config(
    page_title=f"Beranda - {config.get('app', {}).get('title', 'Deteksi Awan AI')}",
    layout="wide"
)

# Jalankan fungsi pembersihan file sementara saat aplikasi dimulai
cleanup_temp_files(age_hours=1)

# Terapkan semua gaya dan komponen layout
apply_global_styles()
render_sidebar_footer()
@st.cache_resource
def get_models():
    """Memuat model AI."""
    return load_segmentation_model(), load_classification_model()
seg_model, cls_model = get_models()

# --- 2. Konten Halaman Utama ---

# Render header dengan judul dari config.yaml
render_page_header(config.get('app', {}).get('title', 'Aplikasi Deteksi Awan Berbasis AI'))

st.write("**Selamat datang** di sistem aplikasi pendeteksian awan berbasis *artificial intelligence*!")

section_divider("Menganalisis Langit, Memahami Cuaca", "🌤️")
st.write(
    "Aplikasi ini dirancang untuk membantu Anda menghitung **tutupan awan** dan "
    "mengidentifikasi **jenis awan** secara otomatis dari gambar atau video citra langit. "
    "Didukung oleh model AI canggih, sistem ini cocok digunakan oleh pengamat cuaca, "
    "peneliti, maupun pengguna umum yang ingin memahami kondisi langit secara visual."
)

# Kartu Fitur Unggulan
st.markdown("""
<div style="background-color:var(--card-bg); border-left:5px solid var(--primary); padding:1.2rem; border-radius:8px; margin: 1.5rem 0;">
    <h4>🔧 Fitur Unggulan</h4>
    <ul>
        <li>📌 Segmentasi awan presisi menggunakan model <strong>CloudDeepLabV3+</strong></li>
        <li>🧠 Klasifikasi jenis awan akurat menggunakan model <strong>YOLOv8</strong></li>
        <li>🖼️ Dukungan analisis untuk <strong>gambar statis dan video</strong></li>
        <li>✍️ ROI fleksibel: <strong>Otomatis atau Manual</strong> untuk akurasi maksimal</li>
        <li>📄 Ekspor hasil analisis ke format <strong>PDF, CSV, dan ZIP</strong></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --- Kartu Navigasi ---
section_divider("Mulai Menjelajah", "🧭")

# 1. Definisikan halaman, deskripsi, dan ikon baru Anda
pages = {
    "Deteksi Awan": ("/Deteksi_Awan", "Unggah gambar atau video untuk analisis mendalam.", "🖼️"),
    "Live Monitoring": ("/Live_Monitoring", "Pantau kondisi awan dari siaran langsung.", "📡"),
    "Riwayat Analisis": ("/Riwayat_Analisis", "Lihat, kelola, dan unduh semua hasil analisis.", "🗂️"),
    "Panduan Pengguna": ("/Panduan_Pengguna", "Pelajari cara kerja dan fitur lengkap aplikasi ini.", "📖")
}

# 2. Logika untuk membuat grid 2x2 (tidak perlu diubah)
nav_items = list(pages.items())
for i in range(0, len(nav_items), 2):
    cols = st.columns(2)
    for j in range(2):
        if i + j < len(nav_items):
            with cols[j]:
                title, (href, desc, emoji) = nav_items[i+j]
                
                # 3. Perbarui HTML untuk menggunakan ikon baru dan warna adaptif
                st.markdown(f"""
                <a href="{href}" target="_self" style="text-decoration: none; color: inherit;">
                    <div class="nav-card">
                        <h4>{emoji} {title}</h4>
                        <p style='font-size:0.9em; opacity:0.8;'>{desc}</p>
                    </div>
                </a>
                """, unsafe_allow_html=True)

# --- 3. Dasbor Statistik dari Database ---
df_history = get_history_df()
render_summary_dashboard(df_history, title="Dasbor Statistik Analisis Keseluruhan")