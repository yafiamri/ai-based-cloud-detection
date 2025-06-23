# Lokasi: pages/3_Penjelasan_Model.py
import streamlit as st
from ui_components import render_page_header, apply_global_styles, render_sidebar_footer

st.set_page_config(page_title="Penjelasan Model", page_icon="🧠", layout="wide")
apply_global_styles()
render_page_header("Penjelasan Model AI", "🧠")

st.markdown("Penjelasan detail mengenai arsitektur CloudDeepLabV3+ dan YOLOv8 akan ada di sini...")
# ... (Tambahkan teks, gambar, dan st.expander sesuai kebutuhan) ...

render_sidebar_footer()