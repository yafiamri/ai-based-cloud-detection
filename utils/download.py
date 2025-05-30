# utils/download.py
import os
import base64
import logging
import streamlit as st
from typing import List, Dict, Optional
from utils.export import export_csv, export_pdf, export_zip

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_filename(name: str) -> str:
    """Sanitasi nama file untuk keamanan"""
    safe_name = "".join([c for c in name if c.isalpha() or c.isdigit() or c in (' ', '_')]).rstrip()
    return safe_name[:50] or "Laporan"

def show_download_link(href: str, format: str):
    """Menampilkan link download dengan styling"""
    st.markdown(f"""
    <div style="margin: 1rem 0; padding: 1rem; background: #f0f2f6; border-radius: 8px;">
        <div style="display: flex; align-items: center; gap: 1rem;">
            <span style="font-size: 1.2em;">{'📦' if format == 'ZIP' else '📄'}</span>
            <div>{href}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def handle_file_cleanup(file_path: str):
    """Bersihkan file temporary dengan error handling"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"File temporary {file_path} dihapus")
    except Exception as e:
        logger.error(f"Gagal menghapus file temporary: {str(e)}")

def download_controller(data: List[Dict], context: str = "deteksi"):
    """Controller untuk manajemen download dengan error handling lengkap"""
    if not data:
        st.warning("📭 Tidak ada data yang tersedia untuk diunduh")
        return

    try:
        # Layout input
        col1, col2 = st.columns([1, 2])
        with col1:
            format_options = {
                "PDF": ("📄 Laporan PDF", "pdf"),
                "CSV": ("📊 Data CSV", "csv"),
                "ZIP": ("📁 Arsip Gambar", "zip")
            }
            selected_key = st.selectbox(
                "Pilih Format",
                options=list(format_options.keys()),
                format_func=lambda x: format_options[x][0],
                key=f"format_{context}"
            )

        # Input nama untuk PDF
        nama_pengguna = ""
        if selected_key == "PDF":
            with col2:
                nama_pengguna = st.text_input(
                    "Nama Penanggung Jawab",
                    value="Yafi Amri",
                    max_chars=50,
                    key=f"nama_{context}"
                )
                nama_pengguna = validate_filename(nama_pengguna)

        # Tombol generate
        if st.button("🛠️ Generate File", key=f"gen_{context}"):
            with st.spinner(f"Sedang membuat {format_options[selected_key][0]}..."):
                try:
                    if selected_key == "CSV":
                        csv_data = export_csv(data)
                        b64 = base64.b64encode(csv_data).decode()
                        href = f'<a download="hasil_{context}.csv" href="data:file/csv;base64,{b64}">Klik untuk Download CSV</a>'
                        
                    elif selected_key == "ZIP":
                        zip_path = export_zip(data)
                        with open(zip_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                        href = f'<a download="arsip_{context}.zip" href="data:application/zip;base64,{b64}">Klik untuk Download ZIP</a>'
                        handle_file_cleanup(zip_path)
                        
                    elif selected_key == "PDF":
                        pdf_path = export_pdf(data, nama_pengguna)
                        with open(pdf_path, "rb") as f:
                            b64 = base64.b64encode(f.read()).decode()
                        href = f'<a download="laporan_{context}.pdf" href="data:application/pdf;base64,{b64}">Klik untuk Download PDF</a>'
                        handle_file_cleanup(pdf_path)
                    
                    show_download_link(href, selected_key)
                    st.success(f"✅ File {selected_key} siap diunduh!")
                    
                except PermissionError as e:
                    logger.error(f"Error permission: {str(e)}")
                    st.error("🔐 Gagal mengakses file. Pastikan tidak ada program lain yang membuka file tersebut.")
                except FileNotFoundError as e:
                    logger.error(f"File tidak ditemukan: {str(e)}")
                    st.error("📂 File output tidak ditemukan. Silakan coba generate ulang.")
                except Exception as e:
                    logger.error(f"Error tidak terduga: {str(e)}", exc_info=True)
                    st.error(f"⚡ Terjadi kesalahan sistem: {str(e)}")

    except Exception as e:
        logger.error(f"Kesalahan controller: {str(e)}")
        st.error("🔥 Terjadi kesalahan fatal pada sistem download. Silakan hubungi administrator.")