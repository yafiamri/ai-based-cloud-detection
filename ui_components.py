# Lokasi File: /ui_components.py
"""
Berisi SEMUA fungsi pembantu yang dapat digunakan kembali untuk membangun
antarmuka pengguna (UI) dengan Streamlit, mempertahankan semua detail estetika.
"""

from __future__ import annotations
import streamlit as st
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
import io
import base64
import logging

# Impor dari 'core' yang akan digunakan oleh komponen UI ini
from core.utils.image_utils import load_demo_images
from core.io import exporter

log = logging.getLogger(__name__)
ASSETS_DIR = Path("assets")

# =============================================================================
# FUNGSI STYLING DAN LAYOUT
# =============================================================================

def apply_global_styles():
    """Menerapkan gaya global (CSS) komprehensif."""
    css = """
    <style>
    :root {
        --primary: #1f77b4;
        --secondary: #2ca02c;
        /* ... variabel warna lainnya ... */
        --card-bg: #f8f9fa;
        --card-border: #e9ecef;
    }
    @media (prefers-color-scheme: dark) {
        :root {
            --primary: #6fa8dc;
            --secondary: #93c47d;
            /* ... variabel warna gelap lainnya ... */
            --card-bg: #161b22;
            --card-border: #30363d;
        }
    }
    
    /* [PENAMBAHAN BARU] Gaya untuk Kartu Navigasi */
    .nav-card {
        display: flex;
        flex-direction: column;
        justify-content: space-between; /* Mendorong tombol ke bawah */
        height: 100%; /* Kunci untuk tinggi yang seragam */
        padding: 1.5rem 1.2rem;
        border-radius: 10px;
        background-color: var(--card-bg);
        border: 1px solid var(--card-border);
        border-top: 4px solid var(--primary); /* Aksen warna biru di atas */
        transition: transform .2s, box-shadow .2s;
        text-align: center;
    }
    .nav-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
    }
    
    /* [PENAMBAHAN BARU] Gaya untuk Tombol di dalam Kartu Navigasi */
    .nav-card a.nav-button-link {
        display: block;
        text-align: center;
        font-weight: bold;
        padding: 0.6rem;
        background-color: var(--primary);
        color: white !important;
        border-radius: 5px;
        text-decoration: none;
        transition: background-color .2s;
    }
    .nav-card a.nav-button-link:hover {
        background-color: var(--secondary);
        text-decoration: none;
    }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_page_header(title: str, logo_name: str = "logo.png") -> None:
    """
    Tampilkan header di atas tiap halaman, dengan logo dan judul.
    Logo diambil dari `assets/logo.png`.
    """
    logo_path = ASSETS_DIR / logo_name
    logo_src = ""
    try:
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            logo_src = f"data:image/png;base64,{b64}"
    except Exception:
        log.warning(f"Logo tidak ditemukan di {logo_path}")

    st.markdown(f"""
    <div style="display:flex; align-items:center; margin-bottom:1.5rem;">
      <img src="{logo_src}" width="48" style="margin-right:12px;"/>
      <h1 style="margin:0; color:var(--primary);">{title}</h1>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar_footer() -> None:
    """Footer statis di sidebar dengan informasi developer."""
    st.sidebar.markdown("""
    <style>
    #footer {
      position: fixed;
      bottom: 0;
      width: inherit;
      padding: 0.5rem;
      font-size:0.8rem;
      text-align:center;
      border-top:1px solid rgba(0,0,0,0.1);
    }
    </style>
    <div id="footer">
      🧑‍💻 Dikembangkan oleh Yafi Amri<br>
      Mahasiswa Meteorologi ITB 2021
    </div>
    """, unsafe_allow_html=True)

def section_divider(text: Optional[str] = None, emoji: str = "📌") -> None:
    """Pembatas antar‐section dengan judul dan garis ber­gradien."""
    if text:
        st.markdown(f"""
        <hr style="
          border:none;
          height:2px;
          background:linear-gradient(90deg,var(--primary),transparent);
          margin:1.5rem 0;
        ">
        <h3 style="color:var(--primary); margin:0.5rem 0;">
          {emoji} {text}
        </h3>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            '<hr style="border:none; height:2px; background:linear-gradient(90deg,var(--primary),transparent); margin:1.5rem 0;">',
            unsafe_allow_html=True)

def small_caption(text: str, icon: Optional[str] = None) -> None:
    """Teks kecil untuk menandai keterangan bawah gambar atau elemen."""
    prefix = f"{icon} " if icon else ""
    st.markdown(
        f'<div style="font-size:0.85em; opacity:0.7; margin-bottom:0.5rem;">'
        f'{prefix}{text}</div>',
        unsafe_allow_html=True)

def display_image_grid(
    image_pairs: List[Tuple[str, Image.Image]], 
    on_click_func: Callable[[str, bytes], None], 
    columns: int = 4
):
    """
    Menampilkan grid gambar demo yang dapat diklik.
    Setiap tombol akan memanggil on_click_func dengan nama dan konten gambar.
    """
    if not image_pairs:
        st.caption("Tidak ada gambar demo yang tersedia.")
        return

    st.info("Pilih satu atau lebih gambar demo di bawah untuk ditambahkan ke antrian analisis.")
    
    cols = st.columns(columns)
    for i, (name, img) in enumerate(image_pairs):
        with cols[i % 4]:
            # Menggunakan div dengan class CSS dari apply_global_styles
            # Anda bisa menambahkan kembali <div class="image-card"> jika mau
            with st.container(border=True):
                st.image(img, use_container_width=True)
                
                # Saat tombol ini diklik, ia akan memanggil on_click_func
                if st.button(f"Pilih: {name}", key=f"demo_{name}", use_container_width=True):
                    # Ubah gambar PIL menjadi bytes sebelum dikirim ke callback
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    on_click_func(name, buf.getvalue())
                    st.toast(f"✅ '{name}' ditambahkan ke antrian!")

# =============================================================================
# FUNGSI RENDER HASIL ANALISIS (TERPADU DAN KONSISTEN)
# =============================================================================

def _render_segmentation_table_html(metrics: Dict[str, Any], is_video: bool) -> str:
    """Membuat string HTML untuk tabel metrik segmentasi."""
    if is_video:
        rows = [
            ("Rata-rata Tutupan Awan", f"{metrics.get('average_cloud_coverage_percent', 0.0):.2f} %"),
            ("Nilai Okta Representatif", str(metrics.get('representative_sky_oktas', '-'))),
            ("Kondisi Langit Representatif", str(metrics.get('representative_sky_condition', '-')))
        ]
    else:
        rows = [
            ("Tutupan Awan", f"{metrics.get('cloud_coverage_percent', 0.0):.2f} %"),
            ("Nilai Okta", str(metrics.get('sky_oktas', '-'))),
            ("Kondisi Langit", str(metrics.get('sky_condition', '-')))
        ]

    html = '<table class="result-table"><tbody>'
    for label, value in rows:
        html += f"<tr><td><strong>{label}</strong></td><td>{value}</td></tr>"
    html += "</tbody></table>"
    return html

def _render_classification_table_html(metrics: Dict[str, Any], is_video: bool) -> str:
    """Membuat string HTML untuk tabel hasil klasifikasi yang dinamis."""
    
    html = '<table class="result-table">'
    if is_video:
        # Untuk video, kita tetap menampilkan ringkasan sederhana
        value = metrics.get('overall_dominant_cloud_type', "N/A")
        html += f"<tbody><tr><td><strong>Jenis Awan Paling Dominan</strong></td><td>{value}</td></tr></tbody>"
    else:
        # Untuk gambar, tampilkan SEMUA yang lolos threshold
        title = "Hasil Klasifikasi"
        # `top_predictions` dari core sudah berisi semua yang lolos threshold
        top_preds = metrics.get("top_predictions", []) 
        
        html += f'<thead><tr><th colspan="2">{title}</th></tr>'
        html += "<tr><th>Jenis Awan</th><th>Confidence</th></tr></thead><tbody>"
        
        if not top_preds:
            html += '<tr><td colspan="2" style="text-align:center;">Tidak ada awan yang terdeteksi di atas ambang batas.</td></tr>'
        else:
            # Loop melalui semua prediksi yang diberikan, tidak lagi dipotong [:3]
            for label, confidence in top_preds:
                html += f"<tr><td>{label}</td><td>{confidence:.2%}</td></tr>"
        html += "</tbody>"
    html += "</table>"
    
    st.markdown(html, unsafe_allow_html=True)

def render_analysis_card(result_data: Dict[str, Any], original_media_path: Optional[Path] = None):
    """Fungsi render TERPADU untuk menampilkan hasil dengan tata letak 2x2 yang konsisten."""
    if not result_data:
        st.warning("Tidak ada data hasil untuk ditampilkan.")
        return

    is_video = "output_video_path" in result_data

    # --- FASE 1: PERSIAPAN VARIABEL ---
    if is_video:
        file_name = result_data.get("file_name", "N/A")
        header_title = f"🎞️ Hasil untuk Video: {file_name}"
        original_media = str(original_media_path) if original_media_path and original_media_path.is_file() else None
        processed_media = result_data.get("output_video_path")
        footer_text = f"Waktu proses: {result_data.get('duration_seconds', 0.0):.2f} d | Total frame diolah: {result_data.get('metrics', {}).get('processed_frame_count', 0)}"
    else:
        file_name = result_data.get("file_name", "N/A")
        header_title = f"🖼️ Hasil untuk Gambar: {file_name}"
        original_media = result_data.get("images", {}).get("original")
        processed_media = result_data.get("images", {}).get("overlay")
        footer_text = f"Waktu proses: {result_data.get('duration_seconds', 0.0):.2f} d | Hash: `{result_data.get('file_hash', 'N/A')}`"

    # --- FASE 2: RENDER TATA LETAK KONSISTEN ---
    with st.container(border=True):
        st.subheader(header_title)
        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            if original_media:
                st.video(original_media) if is_video else st.image(original_media, use_container_width=True)
                st.caption("Gambar/Video Asli" if not is_video else "Video Asli")
        with col2:
            if processed_media:
                st.video(processed_media) if is_video else st.image(processed_media, use_container_width=True)
                st.caption("Hasil Overlay Segmentasi" if not is_video else "Video dengan Overlay")
        
        st.markdown("---")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**Metrik Segmentasi**")
            seg_metrics_data = result_data.get("metrics") if is_video else result_data
            st.markdown(_render_segmentation_table_html(seg_metrics_data, is_video), unsafe_allow_html=True)
        
        with col4:
            st.markdown("**Hasil Klasifikasi**")
            class_metrics_data = result_data.get("metrics") if is_video else result_data
            st.markdown(_render_classification_table_html(class_metrics_data, is_video), unsafe_allow_html=True)
        
        st.caption(footer_text)

# =============================================================================
# FUNGSI DOWNLOAD CONTROLLER
# =============================================================================

def render_download_controller(results: List[Dict[str, Any]], context: str, author_name: str):
    """
    Menampilkan UI untuk memilih format dan mengunduh hasil.
    Menggabungkan penanganan error yang andal dengan stabilitas UI terbaik.
    """
    if not results:
        st.info("Tidak ada data untuk diunduh.")
        return

    st.markdown("---")
    st.subheader("⬇️ Unduh Hasil Analisis")
    
    format_type = st.radio(
        "Pilih format unduhan:", 
        ["CSV", "ZIP", "PDF"], 
        horizontal=True, 
        key=f"download_format_{context}"
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    file_prefix = f"hasil_{context}_{author_name.replace(' ', '_')}_{timestamp}"

    if format_type == "CSV":
        try:
            csv_bytes = exporter.export_to_csv(results)
            st.download_button(
                label="Unduh CSV",
                data=csv_bytes,
                file_name=f"{file_prefix}.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Gagal membuat file CSV: {e}")

    elif format_type == "ZIP":
        try:
            temp_dir = Path("temps/zip_exports")
            temp_dir.mkdir(exist_ok=True)
            zip_path = temp_dir / f"{file_prefix}.zip"

            exporter.export_to_zip(results, zip_path)
            
            with open(zip_path, "rb") as f:
                st.download_button(
                    label="Unduh ZIP",
                    data=f, # st.download_button bisa menerima file-like object
                    file_name=f"{file_prefix}.zip",
                    mime="application/zip",
                )
        except Exception as e:
            st.error(f"Gagal membuat file ZIP: {e}")

    elif format_type == "PDF":
        try:
            temp_dir = Path("temps/pdf_exports")
            temp_dir.mkdir(exist_ok=True)
            pdf_path = temp_dir / f"{file_prefix}.pdf"

            exporter.export_to_pdf(results, author=author_name, output_path=pdf_path)

            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Unduh PDF",
                    data=f,
                    file_name=f"{file_prefix}.pdf",
                    mime="application/pdf",
                )
        except Exception as e:
            st.error(f"Gagal membuat file PDF: {e}")