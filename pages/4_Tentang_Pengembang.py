# pages/4_Tentang_Pengembang.py
import streamlit as st

from utils.layout import apply_global_styles, render_page_header, render_sidebar_footer, section_divider

# --- 1. Konfigurasi Halaman & Gaya ---
st.set_page_config(page_title="Tentang Pengembang", layout="wide")
apply_global_styles()
render_sidebar_footer()

# --- 2. Render Halaman ---
render_page_header("Tentang Pengembang")

section_divider("Profil Singkat", "🧑‍💻")
st.markdown("""
Halo! Saya **Yafi Amri**, mahasiswa Program Studi Sarja Meteorologi Institut Teknologi Bandung (ITB) angkatan 2021.  
Aplikasi ini dikembangkan sebagai bagian dari **Tugas Akhir** dengan tipe Purwarupa.
""")

section_divider("Judul Tugas Akhir", "📌")
st.markdown("""
#### *Pengembangan Sistem Pendeteksian Awan Berbasis Artificial Intelligence*
""")

section_divider("Tujuan & Lingkup Proyek", "🎯")
st.markdown("""
**Tujuan Proyek:** Merancang sistem berbasis AI untuk:
- Melakukan segmentasi tutupan awan (dengan **CloudDeepLabV3+**)
- Melakukan klasifikasi jenis awan (dengan **YOLOv8**)
- Memberikan hasil analisis awan yang objektif, cepat, dan konsisten.

**Lingkup Implementasi:**
- **Input:** Citra langit dalam format gambar dan video.
- **Proses:** Preprocessing → Segmentasi → Klasifikasi.
- **Output:** Visualisasi tutupan awan (% & oktaf), jenis awan, dan ekspor laporan.
""")

section_divider("Hubungi Saya", "🔗")
st.markdown("""
- **GitHub:** [github.com/yafiamri](https://github.com/yafiamri)
- **LinkedIn:** [linkedin.com/in/yafiamri](https://www.linkedin.com/in/yafiamri)

Terima kasih telah menggunakan aplikasi ini! 🌤️
""")