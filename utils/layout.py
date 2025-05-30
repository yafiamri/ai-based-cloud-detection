# utils/layout.py
import streamlit as st
import os
import logging
import base64
from typing import List, Optional, Tuple
from PIL import Image, UnidentifiedImageError

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_global_styles():
    """Menerapkan gaya global dengan dukungan dark mode dan optimasi performa"""
    # st.set_page_config(layout="wide", page_icon="🌤️")
    st.markdown("""
    <style>
    /* Variabel CSS untuk tema */
    :root {
        --primary: #1f77b4;
        --secondary: #2ca02c;
        --background: #ffffff;
        --text: #333333;
        --card-bg: #f8f9fa;
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --primary: #6fa8dc;
            --secondary: #93c47d;
            --background: #2d2d2d;
            --text: #ffffff;
            --card-bg: #3a3a3a;
        }
    }

    /* Reset gaya dasar */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', system-ui, sans-serif;
        background-color: var(--background);
        color: var(--text);
        line-height: 1.6;
    }

    /* Kontainer utama */
    .main {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }

    /* Kartu gambar */
    .image-card {
        background: var(--card-bg);
        border-radius: 10px;
        padding: 1rem;
        transition: transform 0.2s, box-shadow 0.2s;
        border: 1px solid rgba(0,0,0,0.1);
    }

    .image-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Header hasil */
    .result-header {
        font-size: 1.4rem;
        color: var(--primary);
        margin: 1.5rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary);
    }

    /* Tabel hasil */
    .result-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
    }
    
    .result-table th, .result-table td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid rgba(0,0,0,0.1);
    }
    
    .result-table th {
        background-color: var(--card-bg);
    }
    </style>
    """, unsafe_allow_html=True)

def render_page_header(title: str, logo_path: str = "assets/logo.png"):
    """Tampilkan header dengan logo dan judul sejajar, font default sistem (lokal-friendly)."""
    try:
        with open(logo_path, "rb") as f:
            logo_encoded = base64.b64encode(f.read()).decode()
            logo_src = f"data:image/png;base64,{logo_encoded}"
    except FileNotFoundError:
        logo_src = ""

    st.markdown(f"""
    <div style='display: flex; align-items: center; margin-bottom: 1.5rem;'>
        <img src="{logo_src}" width="52" style='margin-right: 14px;'/>
        <span style="
            font-size: 2.4rem;
            font-weight: bold;
            font-family: Segoe UI, system-ui, sans-serif;
            color: var(--primary);
            line-height: 1.1;
        ">
            {title}
        </span>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar_footer():
    """Render footer sidebar yang konsisten di semua halaman"""
    st.sidebar.markdown("""
    <style>
    #sidebar-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        width: 15.25rem;
        padding: 1rem;
        font-size: 12px;
        text-align: center;
        background-color: transparent;
        border-top: 1px solid rgba(0,0,0,0.1);
    }
    </style>
    
    <div id="sidebar-footer">
        🧑‍💻 Dikembangkan oleh: <b>Yafi Amri</b><br>
        Mahasiswa Meteorologi ITB 2021
    </div>
    """, unsafe_allow_html=True)

def render_segmentation_metrics(result: dict):
    """Menampilkan metrik segmentasi dalam tabel"""
    st.markdown("#### 📊 Metrik Segmentasi")
    metrics = [
        ("Tutupan Awan (%)", f"{result.get('coverage', 0):.2f}%"),
        ("Nilai Oktaf", str(result.get('oktaf', 0))),
        ("Kondisi Langit", result.get('kondisi_langit', '-'))
    ]

    table_html = """
    <table class="result-table">
        <thead>
            <tr><th>Parameter</th><th>Nilai</th></tr>
        </thead>
        <tbody>
    """
    for label, value in metrics:
        table_html += f"<tr><td><strong>{label}</strong></td><td>{value}</td></tr>"
    table_html += "</tbody></table>"

    st.markdown(table_html, unsafe_allow_html=True)

def render_classification_results(result: dict):
    """Menampilkan hasil klasifikasi dalam tabel"""
    st.markdown("#### 🌥️ Hasil Klasifikasi")
    preds = result.get("top_preds", [])

    if not preds or not isinstance(preds, list):
        st.warning("Data klasifikasi tidak tersedia")
        return

    table_html = """
    <table class="result-table">
        <thead>
            <tr><th>Jenis Awan</th><th>Confidence</th></tr>
        </thead>
        <tbody>
    """
    for label, confidence in preds[:3]:
        table_html += f"<tr><td>{label}</td><td>{confidence:.4f}%</td></tr>"
    table_html += "</tbody></table>"

    st.markdown(table_html, unsafe_allow_html=True)

@st.cache_data(max_entries=100, show_spinner=False)
def load_image_with_cache(img_path: str) -> Image.Image:
    """Memuat gambar dengan cache dan error handling"""
    try:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"File tidak ditemukan: {img_path}")
            
        with open(img_path, "rb") as f:
            return Image.open(f).convert("RGB")
            
    except (FileNotFoundError, UnidentifiedImageError) as e:
        logger.error(f"Gagal memuat gambar: {str(e)}")
        return Image.open("assets/placeholder.png").convert("RGB")
    except Exception as e:
        logger.error(f"Error tidak terduga: {str(e)}")
        return Image.open("assets/placeholder.png").convert("RGB")

def display_image_grid(
    images: List,
    captions: Optional[List[str]] = None,
    columns: int = 4,
    lazy_loading: bool = True
):
    """Menampilkan grid gambar dengan lazy loading dan error handling"""
    cols = st.columns(columns)
    for idx, img_data in enumerate(images):
        with cols[idx % columns]:
            try:
                # Handle tipe input berbeda
                if isinstance(img_data, tuple):
                    img, caption = img_data
                else:
                    img = img_data
                    caption = captions[idx] if captions else None
                    
                # Lazy loading dengan placeholder
                with st.container():
                    st.markdown('<div class="image-card">', unsafe_allow_html=True)
                    
                    if lazy_loading:
                        with st.spinner(""):
                            if isinstance(img, str):
                                img = load_image_with_cache(img)
                                
                            st.image(
                                img,
                                caption=caption,
                                use_column_width=True,
                                output_format="JPEG",
                                clamp=True
                            )
                    else:
                        if isinstance(img, str):
                            img = load_image_with_cache(img)
                            
                        st.image(
                            img,
                            caption=caption,
                            use_column_width=True,
                            output_format="JPEG",
                            clamp=True
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
            except IndexError:
                logger.warning("Jumlah caption tidak sesuai dengan gambar")
            except Exception as e:
                logger.error(f"Error menampilkan gambar: {str(e)}")
                st.error("Terjadi kesalahan saat memuat gambar")

def section_divider(text: Optional[str] = None, emoji: str = "📌"):
    """Membuat pembatas section dengan styling modern"""
    if text:
        st.markdown(
            f"""<hr style="border:none;height:2px;background:linear-gradient(
            90deg, var(--primary) 0%, transparent 100%);margin:1.5rem 0">
            <h3 style="color:var(--primary);margin-bottom:1rem">{emoji} {text}</h3>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """<hr style="border:none;height:2px;background:linear-gradient(
            90deg, var(--primary) 0%, transparent 100%);margin:1.5rem 0">""",
            unsafe_allow_html=True
        )

def small_caption(text: str, icon: Optional[str] = None):
    """Teks keterangan kecil dengan ikon opsional"""
    icon_html = f"{icon} " if icon else ""
    st.markdown(
        f'<div style="font-size:0.85em; color:var(--text); opacity:0.8; margin:0.5rem 0">'
        f'{icon_html}{text}</div>',
        unsafe_allow_html=True
    )

def resolve_path(path):
    if isinstance(path, list):
        path = path[0] if path else ""

    if not path:
        return ""

    path = path.replace("\\", "/")
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)

    return path if os.path.exists(path) else ""

def render_result(result: dict):
    """Menampilkan hasil analisis lengkap untuk gambar atau video."""
    raw_img_path = result.get("original_image_path") or result.get("original_path")
    raw_overlay_path = result.get("overlay_image_path") or result.get("overlay_path")

    img_path = resolve_path(raw_img_path)
    overlay_path = resolve_path(raw_overlay_path)
    video_path = resolve_path(result.get("original_video_path"))
    overlay_video_path = resolve_path(result.get("overlay_video_path"))

    if img_path or overlay_path:
        render_image_result(result)
    elif video_path or overlay_video_path:
        render_video_result(result)
    else:
        st.warning("⚠️ Gambar atau video tidak ditemukan.")

    if result.get("coverage") is not None:
        render_segmentation_metrics(result)

    if result.get("top_preds"):
        render_classification_results(result)

def render_image_result(result: dict):
    """Menampilkan hasil analisis citra dalam dua kolom: original dan overlay."""
    try:
        original_path = resolve_path(result.get("original_image_path") or result.get("original_path"))
        overlay_path = resolve_path(result.get("overlay_image_path") or result.get("overlay_path"))

        col1, col2 = st.columns(2)

        with col1:
            if original_path and os.path.isfile(original_path):
                try:
                    img = Image.open(original_path).convert("RGB")
                    st.image(img, use_column_width=True)
                    small_caption("Citra Asli", icon="🖼️")
                except UnidentifiedImageError:
                    st.warning("⚠️ Format gambar asli tidak dikenali.")
            else:
                st.warning("Gambar asli tidak ditemukan.")

        with col2:
            if overlay_path and os.path.isfile(overlay_path):
                try:
                    img = Image.open(overlay_path).convert("RGB")
                    st.image(img, use_column_width=True)
                    small_caption("Hasil Segmentasi", icon="🧠")
                except UnidentifiedImageError:
                    st.warning("⚠️ Format gambar hasil tidak dikenali.")
            else:
                st.warning("Gambar hasil tidak ditemukan.")

    except Exception as e:
        st.error("Gagal menampilkan hasil gambar.")

def render_video_result(result: dict):
    """Menampilkan hasil analisis video (asli dan hasil overlay) menggunakan st.video() langsung."""
    try:
        video1 = result.get("original_video_path", "")
        video2 = result.get("overlay_video_path", "")

        if not (os.path.exists(video1) and os.path.exists(video2)):
            st.warning("Salah satu atau kedua video tidak tersedia")
            return

        section_divider("Video Asli dan Hasil Segmentasi", emoji="🎞️")

        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                st.video(video1)
                st.caption("Video Asli")

            with col2:
                st.video(video2)
                st.caption("Hasil Analisis")

    except Exception as e:
        logger.error(f"Video display error: {str(e)}")
        st.error("Gagal memuat video")
    """
    with st.container():
        col1, col2 = st.columns(2)
        
        try:
            with col1:
                if os.path.exists(result.get("original_video_path", "")):
                    st.video(result["original_video_path"], start_time=0)
                    st.caption("Video Asli")
                else:
                    st.warning("Video asli tidak tersedia")
                    
            with col2:
                if os.path.exists(result.get("overlay_video_path", "")):
                    st.video(result["overlay_video_path"], start_time=0)
                    st.caption("Hasil Analisis")
                else:
                    st.warning("Video hasil analisis tidak tersedia")
        except Exception as e:
            logger.error(f"Video display error: {str(e)}")
            st.error("Gagal memuat video")
        """