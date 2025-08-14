# pages/2_Live_Monitoring.py
import streamlit as st
import os
import cv2
import time
import tempfile
import yt_dlp
import yt_dlp.utils
import numpy as np
import pandas as pd
import re
import streamlit.components.v1 as components
from PIL import Image
from datetime import datetime, timezone, timedelta
from streamlit_drawable_canvas import st_canvas
from typing import Dict, Any, Optional
from urllib.parse import urlparse

# Impor dari semua utilitas yang relevan
from utils.config import config
from utils.database import add_history_entry, get_pipeline_version_hash
from utils.layout import apply_global_styles, render_page_header, render_sidebar_footer, section_divider, render_result, render_summary_dashboard
from utils.media import fetch_live_stream_source
from utils.processing import get_file_hash, get_analysis_hash, analyze_single_image, create_enhanced_overlay
from utils.segmentation import load_segmentation_model, canvas_to_mask
from utils.classification import load_classification_model
from utils.system import cleanup_temp_files
from utils.download import download_controller

# Fungsi helper untuk memastikan aplikasi berjalan stabil di lingkungan cloud.
def get_frame_from_stream(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """
    Membaca satu frame dari objek VideoCapture yang baru dibuat.
    Karena koneksi selalu baru, frame ini dijamin yang paling mutakhir.
    """
    if not cap or not cap.isOpened():
        return None
    try:
        ret, frame = cap.read()
        return frame if ret else None
    except Exception as e:
        print(f"Error saat membaca frame dari stream: {e}")
        return None

# --- 1. Konfigurasi Halaman & Inisialisasi State ---
st.set_page_config(page_title=f"Live Monitoring - {config['app']['title']}", layout="wide")
# Inisialisasi state yang terpusat dan bersih untuk halaman ini
if "widget_seed" not in st.session_state:
    st.session_state.widget_seed = 0
if "live" not in st.session_state:
    st.session_state.live = {
        "running": False,
        "source_info": None,
        "last_result": None,
        "preview_frame": None,
        "roi_method": "Otomatis",
        "interval": 10,
        "canvas": None,
        "url_input": "",
        "session_results": [],
        "save_to_history": True
    }

# Logika untuk menampilkan toast setelah sesi direset
if "toast_message" not in st.session_state:
    st.session_state.toast_message = None
if st.session_state.toast_message:
    message, icon = st.session_state.toast_message
    st.toast(message, icon=icon)
    st.session_state.toast_message = None # Hapus setelah ditampilkan

# Panggil pembersihan, terapkan layout, muat model
cleanup_temp_files()
apply_global_styles()
render_sidebar_footer()
@st.cache_resource
def get_models():
    """Memuat semua model AI yang diperlukan."""
    return load_segmentation_model(), load_classification_model()
seg_model, cls_model = get_models()

# --- Sidebar ---
with st.sidebar:
    if st.button("🔄️ Mulai Sesi Monitoring Baru", use_container_width=True):
        st.session_state.live = {
            "running": False, "source_info": None, "last_result": None,
            "preview_frame": None, "roi_method": "Otomatis", "interval": 10,
            "canvas": None, "url_input": "", "session_results": [],
            "save_to_history": True
        }
        st.session_state.toast_message = ("Sesi monitoring baru dimulai.", "🔄️")
        st.session_state.widget_seed += 1
        st.rerun()

# --- 2. Render UI & Logika Utama ---
render_page_header("Monitoring Awan Real-Time")
st.write("Analisis tutupan dan jenis awan secara otomatis dari siaran langsung (*live stream*).")

# --- Langkah 1: Input URL & Pratinjau ---
section_divider("Langkah 1: Masukkan URL & Lihat Pratinjau", "📡")
st.markdown("Tempelkan tautan **siaran langsung** dari untuk memulai pemantauan.")

seed = st.session_state.widget_seed

with st.form(key=f"url_form_live_{seed}"):
    url_input = st.text_input("Tempel URL siaran langsung di sini",
                              placeholder="https://www.twitch.tv/USERNAME", 
                              help="Masukkan URL siaran langsung (hanya dari Twitch yang didukung oleh sistem saat ini)",
                              key=f"live_url_input_{seed}"
                              )
    submitted = st.form_submit_button("🔗 Periksa URL", use_container_width=True)

if submitted:
    if url_input:
        st.session_state.live["url_input"] = url_input
        st.session_state.live.update({"last_result": None, "preview_frame": None, "running": False})

        with st.spinner("Memvalidasi URL siaran langsung..."):
            source_info, msg = fetch_live_stream_source(url_input)
        
        if source_info:
            st.toast("URL siaran langsung berhasil divalidasi!", icon="✅")
            st.session_state.live["source_info"] = source_info
            if msg:
                st.info(msg, icon="ℹ️")
        else:
            st.error(msg or "Terjadi kesalahan yang tidak diketahui.", icon="❌")
            st.session_state.live["source_info"] = None
    else:
        st.toast("Input URL kosong. Silakan masukkan URL.", icon="⚠️")

if st.session_state.live.get("source_info"):
    st.subheader("**🎬 Pratinjau Siaran Langsung**")

    source_info = st.session_state.live["source_info"]
    
    if st.session_state.live.get("preview_frame") is None and source_info.get("src"):
        with st.spinner("Mengambil gambar pratinjau dari stream..."):
            # Buka dan tutup koneksi hanya untuk mendapatkan satu frame pratinjau
            cap_preview = cv2.VideoCapture(source_info["src"])
            if cap_preview.isOpened():
                ret, frame = cap_preview.read()
                cap_preview.release()
                if ret:
                    st.session_state.live["preview_frame"] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if st.session_state.live.get("preview_frame"):
        player_max_width = 800
        player_height = int(player_max_width * (9 / 16))
        display_url = source_info["display_url"]
        stream_url = source_info.get("src")
        
        player_html = ""
        
        if "youtube.com" in display_url or "youtu.be" in display_url:
            match = re.search(r"(?:v=|\/|live\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", display_url)
            if match:
                video_id = match.group(1)
                player_html = f'<iframe src="https://www.youtube.com/embed/{video_id}?autoplay=0&mute=1" width="100%" height="{player_height}px" style="border:none;" allow="fullscreen"></iframe>'
        
        elif "twitch.tv" in display_url:
            match = re.search(r"twitch\.tv/([a-zA-Z0-9_]+)", display_url)
            if match:
                channel_name = match.group(1)
                hostname = urlparse(st.get_option("server.baseUrlPath")).hostname or "localhost"
                player_html = f"""
                    <div id="twitch-embed"></div>
                    <script src="https://embed.twitch.tv/embed/v1.js"></script>
                    <script type="text/javascript">
                      new Twitch.Embed("twitch-embed", {{
                        width: "100%",
                        height: {player_height},
                        channel: "{channel_name}",
                        layout: "video",
                        muted: true,
                        parent: ["{hostname}"]
                      }});
                    </script>
                """

        if not player_html and source_info.get("type") != "rtsp":
            player_html = f"""
                <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
                <video id="live-video" controls muted width="100%" height="{player_height}px" style="background-color:black;"></video>
                <script>
                  var video = document.getElementById('live-video');
                  if(Hls.isSupported()) {{
                    var hls = new Hls();
                    hls.loadSource("{stream_url}");
                    hls.attachMedia(video);
                  }}
                </script>
            """
        
        if player_html:
            components.html(
                f'<div style="max-width:{player_max_width}px; margin: auto;">{player_html}</div>',
                height=player_height + 40
            )
        else:
             _, col_img, _ = st.columns([1, 2, 1])
             with col_img:
                st.image(st.session_state.live["preview_frame"], caption="Pratinjau Statis dari Stream", use_container_width=True)

    else:
        st.warning("Tidak dapat memuat pratinjau stream untuk ditampilkan.")

# --- Langkah 2: Konfigurasi Analisis ---
section_divider("Langkah 2: Konfigurasi Analisis", "⚙️")

def update_live_config(key_to_update, widget_key):
    """Callback untuk memperbarui state di dalam st.session_state.live."""
    st.session_state.live[key_to_update] = st.session_state[widget_key]

is_config_disabled = not st.session_state.live.get("source_info")

with st.container(border=True):
    col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
    with col1:
        interval_key = "live_interval_widget"
        st.slider(
            "Interval antar-*frame* (detik)", 10, 600, 
            value=st.session_state.live.get("interval", 10), 
            step=5,
            key=interval_key,
            on_change=update_live_config,
            args=("interval", interval_key),
            disabled=is_config_disabled, 
            help="Seberapa sering analisis akan dijalankan."
        )
    with col2:
        roi_options = ["Otomatis", "Manual (Kotak)", "Manual (Poligon)", "Manual (Diameter Lingkaran)"]
        try:
            current_roi_index = roi_options.index(st.session_state.live.get("roi_method", "Otomatis"))
        except ValueError:
            current_roi_index = 0
        
        roi_key = "live_roi_widget"
        st.selectbox(
            "Metode (*Region of Interest*) ROI:", roi_options,
            index=current_roi_index,
            key=roi_key,
            on_change=update_live_config,
            args=("roi_method", roi_key),
            disabled=is_config_disabled
        )
    with col3:
        save_key = "live_save_widget"
        st.toggle(
            "Simpan hasil ke riwayat", 
            value=st.session_state.live.get("save_to_history", True),
            key=save_key,
            on_change=update_live_config,
            args=("save_to_history", save_key),
            disabled=is_config_disabled, 
            help="Jika aktif, setiap frame yang dianalisis akan disimpan ke database riwayat."
        )

user_roi_mask = None
if "Manual" in st.session_state.live.get("roi_method", "Otomatis") and not is_config_disabled:
    with st.expander("✏️ Gambar ROI Manual", expanded=True):
        if st.session_state.live.get("preview_frame"):
            st.info("Gambar bentuk dengan melakukan *drag and drop* pada kanvas di bawah untuk menandai area langit yang ingin dianalisis.")
            bg_image = st.session_state.live["preview_frame"]

            w, h = bg_image.size
            ratio = h / w if w > 0 else 1
            canvas_w = 512
            canvas_h = int(canvas_w * ratio)
            MAX_CANVAS_HEIGHT = 600
            canvas_h = min(canvas_h, MAX_CANVAS_HEIGHT)
            if canvas_h == MAX_CANVAS_HEIGHT and ratio > 0:
                canvas_w = int(canvas_h / ratio)
            
            roi_method_selected = st.session_state.live["roi_method"]
            drawing_mode = "rect" if "Kotak" in roi_method_selected else "polygon" if "Poligon" in roi_method_selected else "line"

            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)", stroke_width=2,
                background_image=bg_image.resize((canvas_w, canvas_h)),
                height=canvas_h, width=canvas_w, 
                drawing_mode=drawing_mode,
                key="live_canvas"
            )
            st.session_state.live['canvas'] = canvas_result

            if canvas_result and canvas_result.json_data and canvas_result.json_data.get("objects"):
                user_roi_mask = canvas_to_mask(canvas_result, h, w)
        else:
            st.warning("Tidak bisa menampilkan pratinjau untuk kanvas ROI. Berkas mungkin korup atau formatnya tidak didukung.")

# --- Langkah 3: Jalankan Monitoring & Tampilkan Hasil ---
section_divider("Langkah 3: Jalankan Monitoring", "🚀")

button_text = "⏹️ Hentikan Monitoring" if st.session_state.live.get("running") else "▶️ Mulai Monitoring"
if st.button(button_text, type="primary", use_container_width=True, disabled=is_config_disabled):
    is_running = st.session_state.live["running"]
    st.session_state.live["running"] = not is_running
    if not is_running:
        st.session_state.live["session_results"] = []
        st.session_state.live["last_result"] = None
    else:
        st.session_state.toast_message = ("Monitoring dihentikan.", "🛑")
    st.rerun()

# --- BLOK TAMPILAN DAN PEMROSESAN UTAMA ---
info_placeholder = st.empty()
result_placeholder = st.empty()
        
if not st.session_state.live.get("running") and st.session_state.live.get("last_result"):
    with result_placeholder.container():
        render_result(st.session_state.live.get("last_result"))

if st.session_state.live.get("running"):
    info_placeholder.info(f"Monitoring sedang berlangsung. Menunggu interval {st.session_state.live.get('interval', 10)} detik...", icon="🛰️")
    st.toast("Monitoring dimulai!", icon="👀")

    source_info = st.session_state.live["source_info"]
    stream_url = source_info.get("src")

    if not stream_url:
        st.error("URL stream tidak valid. Monitoring dihentikan.")
        st.session_state.live["running"] = False
        st.rerun()

    pipeline_hash = get_pipeline_version_hash()
    consecutive_failures = 0
    MAX_FAILURES = 3

    while st.session_state.live.get("running"):
        loop_start_time = time.time()
        
        info_placeholder.info(f"Membuka koneksi baru ke stream untuk mendapatkan frame termutakhir...", icon="🔗")
        
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            consecutive_failures += 1
            info_placeholder.warning(f"Gagal membuka koneksi stream (percobaan {consecutive_failures}/{MAX_FAILURES}).")
            if consecutive_failures >= MAX_FAILURES:
                st.error("Gagal membuka koneksi beberapa kali. Monitoring dihentikan.")
                st.session_state.live["running"] = False
                st.rerun()
            time.sleep(5) # Tunggu lebih lama jika koneksi gagal
            continue

        frame = get_frame_from_stream(cap)
        cap.release() # Langsung tutup koneksi setelah frame didapat
        
        if frame is None:
            consecutive_failures += 1
            info_placeholder.warning(f"Gagal mengambil frame dari koneksi baru (percobaan {consecutive_failures}/{MAX_FAILURES}).")
            if consecutive_failures >= MAX_FAILURES:
                st.error("Gagal mengambil frame beberapa kali. Stream mungkin tidak stabil. Monitoring dihentikan.")
                st.session_state.live["running"] = False
                st.rerun()
            time.sleep(2)
            continue

        consecutive_failures = 0
        info_placeholder.info(f"Frame berhasil diambil. Memulai analisis...", icon="🔬")
        
        # --- MULAI BLOK ANALISIS ---
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        live_config = {
            "roi_method": st.session_state.live.get("roi_method", "Otomatis"),
            "canvas": st.session_state.live.get("canvas"),
            "interval": st.session_state.live.get("interval", 10)
        }

        user_roi_mask = None
        if "Manual" in live_config["roi_method"]:
            canvas_data = live_config["canvas"]
            if canvas_data and canvas_data.json_data and canvas_data.json_data.get("objects"):
                user_roi_mask = canvas_to_mask(canvas_data, pil_frame.height, pil_frame.width)
            else:
                user_roi_mask = np.zeros((pil_frame.height, pil_frame.width), dtype=np.uint8)
        
        analysis_data = analyze_single_image(pil_frame, seg_model, cls_model, user_roi_mask)

        with tempfile.NamedTemporaryFile(delete=True, suffix=".png") as tmp:
            pil_frame.save(tmp, format="PNG")
            tmp.seek(0)
            file_hash = get_file_hash(tmp)
            analysis_hash = get_analysis_hash(file_hash, pipeline_hash, live_config)
            file_size = tmp.tell()

        is_saving_permanently = st.session_state.live["save_to_history"]
        sufix = "live_monitoring" if is_saving_permanently else "unsaved_live"
        timestamp_name = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}UTC_{sufix}"

        if is_saving_permanently:
            base_original_dir, base_mask_dir, base_overlay_dir = (config['paths']['original_archive'], config['paths']['mask_archive'], config['paths']['overlay_archive'])
        else:
            temp_session_dir = os.path.join(config['paths']['temp_dir'], "live_session_artefacts")
            base_original_dir, base_mask_dir, base_overlay_dir = (os.path.join(temp_session_dir, "original"), os.path.join(temp_session_dir, "masks"), os.path.join(temp_session_dir, "overlays"))

        original_path = os.path.join(base_original_dir, timestamp_name, f"{timestamp_name}_original.png")
        mask_path = os.path.join(base_mask_dir, timestamp_name, f"{timestamp_name}_mask.png")
        overlay_path = os.path.join(base_overlay_dir, timestamp_name, f"{timestamp_name}_overlay.png")

        for p in [original_path, mask_path, overlay_path]:
            os.makedirs(os.path.dirname(p), exist_ok=True)
        pil_frame.save(original_path, "PNG")
        Image.fromarray(analysis_data['segmentation_mask'] * 255).save(mask_path)
        overlay_img = create_enhanced_overlay(pil_frame, analysis_data['segmentation_mask'], analysis_data['roi_mask'])
        overlay_img.save(overlay_path)

        relative_original = os.path.relpath(original_path).replace("\\", "/")
        relative_mask = os.path.relpath(mask_path).replace("\\", "/")
        relative_overlay = os.path.relpath(overlay_path).replace("\\", "/")

        analysis_duration = time.time() - loop_start_time
        db_entry = {
            **analysis_data,
            "pipeline_version_hash": get_pipeline_version_hash(),
            "file_hash": file_hash,
            "analysis_hash": analysis_hash,
            "source_filename": f"Live Frame ({datetime.now():%Y-%m-%d %H:%M:%S} UTC){' (Unsaved)' if not is_saving_permanently else ''}",
            "media_type": "live_frame",
            "file_size_bytes": file_size,
            "analyzed_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
            "analysis_duration_sec": analysis_duration,
            "original_path": relative_original,
            "mask_path": relative_mask,
            "overlay_path": relative_overlay,
        }
        
        if is_saving_permanently:
            add_history_entry(db_entry)
        
        st.session_state.live["session_results"].append(db_entry)
        st.session_state.live["last_result"] = db_entry

        with result_placeholder.container():
            render_result(db_entry)

        interval = st.session_state.live.get("interval", 10)
        sleep_duration = interval - analysis_duration

        if sleep_duration > 0:
            info_placeholder.info(f"Analisis selesai dalam {analysis_duration:.2f} detik. Menunggu {sleep_duration:.2f} detik untuk siklus berikutnya...", icon="⏱️")
            time.sleep(sleep_duration)

    if not st.session_state.live.get("running"):
        st.rerun()

# --- Langkah 4: Rangkuman Sesi Monitoring ---
if st.session_state.live.get("session_results") and not st.session_state.live.get("running"):
    df_session_results = pd.DataFrame(st.session_state.live["session_results"])
    
    render_summary_dashboard(df_session_results, title="Rangkuman Sesi Monitoring")
    
    section_divider("Unduh Hasil Sesi Ini", "📥")
    download_controller(st.session_state.live["session_results"], context="live")
