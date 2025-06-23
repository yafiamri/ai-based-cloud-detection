# Lokasi: pages/4_Tentang_Pengembang.py
import streamlit as st
from ui_components import render_page_header, apply_global_styles, render_sidebar_footer

st.set_page_config(page_title="Tentang Pengembang", page_icon="🧑‍💻", layout="wide")
apply_global_styles()
render_page_header("Tentang Pengembang", "🧑‍💻")

st.markdown("""
### Yafi Amri
- **Program Studi**: Meteorologi
- **Institusi**: Institut Teknologi Bandung (ITB)
- **Angkatan**: 2021

Aplikasi ini merupakan bagian dari proyek studi untuk mengembangkan sistem cerdas dalam bidang meteorologi.
""")
# ... (Tambahkan link ke GitHub, LinkedIn, dll.) ...

render_sidebar_footer()