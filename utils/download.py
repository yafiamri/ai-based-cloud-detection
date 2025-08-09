# utils/download.py
import streamlit as st
import base64
import time
from datetime import datetime
from typing import List, Dict, Any

# Impor fungsi export dan konfigurasi
from .export import export_csv, export_pdf, export_zip
from .config import config

def _create_download_link(file_path: str, link_text: str, file_name: str) -> str:
    """Membuat link unduhan HTML dari sebuah file lokal."""
    try:
        with open(file_path, "rb") as f:
            bytes_data = f.read()
        b64 = base64.b64encode(bytes_data).decode()
        return f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">{link_text}</a>'
    except FileNotFoundError:
        return f"<p style='color:red;'>File tidak ditemukan: {file_name}</p>"

def download_controller(data: List[Dict[str, Any]], context: str = "detect"):
    """
    Menampilkan komponen UI untuk mengunduh hasil analisis dalam berbagai format.

    Args:
        data (List[Dict[str, Any]]): Daftar dictionary hasil analisis.
        context (str): Konteks pemanggilan untuk membuat key widget yang unik.
    """
    if not data:
        st.warning("Tidak ada data yang valid untuk diunduh.")
        return
    
    col1, col2 = st.columns([0.4, 0.6])
    with col1:
        selected_format = st.selectbox(
            "Pilih Format Unduhan:",
            ["PDF", "CSV", "ZIP"],
            key=f"format_{context}"
        )
    with col2:
        default_user = config.get('app', {}).get('author', 'Pengguna')
        nama_pengguna = st.text_input(
            "Nama untuk Laporan PDF:",
            value=default_user,
            key=f"nama_{context}",
            disabled=(selected_format != "PDF")
        )

    if st.button("🛠️ Buat File & Tampilkan Link Unduh", key=f"btn_{context}", use_container_width=True):
        start_time = time.perf_counter() # Catat waktu mulai
        with st.spinner("⏳ Sedang membuat file, mohon tunggu..."):
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + "_UTC"
                link = ""
                
                if selected_format == "CSV":
                    file_path = export_csv(data)
                    link = _create_download_link(file_path, "💾 Unduh CSV", f"metadata_{context}_{timestamp}.csv")
                elif selected_format == "ZIP":
                    file_path = export_zip(data)
                    link = _create_download_link(file_path, "📦 Unduh ZIP", f"archive_{context}_{timestamp}.zip")
                elif selected_format == "PDF":
                    if not nama_pengguna.strip():
                        st.error("Nama untuk laporan PDF tidak boleh kosong.")
                        return
                    file_path = export_pdf(data, nama_pengguna)
                    link = _create_download_link(file_path, "📑 Unduh PDF", f"report_{context}_{timestamp}.pdf")
                duration = time.perf_counter() - start_time # Hitung durasi
                st.success(f"✅ File siap dalam {duration:.2f} detik! Klik tautan di bawah untuk mengunduh.")
                st.markdown(link, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Gagal membuat file: {e}")
                st.exception(e) # Menampilkan detail error untuk debugging