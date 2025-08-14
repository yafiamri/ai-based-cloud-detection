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


def get_frame_from_stream(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """
    Membaca satu frame dari objek VideoCapture yang sudah diinisialisasi.
    """
    if not cap or not cap.isOpened():
        return None
    try:
        cap.grab()
        ret, frame = cap.retrieve()
        return frame if ret else None
    except Exception as e:
        print(f"Error saat membaca frame dari stream: {e}")
        return None

# --- Konfigurasi Halaman & Inisialisasi State ---
st.set_page_config(page_title=f"Live Monitoring - {config['app']['title']}", layout="wide")
if "widget_seed" not in st.session_state:
    st.session_state.widget_seed = 0
if "live" not in st.session_state:
    st.session_state.live = {
        "running": False, "source_info": None, "last_result": None,
        "preview_frame": None, "roi_method": "Otomatis", "interval": 10,
        "canvas": None, "url_input": "", "session_results": [],
        "save_to_history": True
    }
if "toast_message" not in st.session_state:
    st.session_state.toast_message = None
if st.session_state.toast_message:
    st.toast(st.session_state.toast_message[0], icon=st.session_state.toast_message[1])
    st.session_state.toast_message = None

cleanup_temp_files()
apply_global_styles()
render_sidebar_footer()
@st.cache_resource
def get_models():
    return load_segmentation_model(), load_classification_model()
seg_model, cls_model = get_models()

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

render_page_header("Monitoring Awan Real-Time")
st.write("Analisis tutupan dan jenis awan secara otomatis dari siaran langsung (*live stream*).")

section_divider("Langkah 1: Masukkan URL & Lihat Pratinjau", "📡")
st.markdown("Tempelkan tautan **siaran langsung** dari platform seperti YouTube atau Twitch untuk memulai pemantauan.")

seed = st.session_state.widget_seed
with st.form(key=f"url_form_live_{seed}"):
    url_input = st.text_input("Tempel URL siaran langsung di sini",
                              placeholder="https://www.youtube.com/live/VIDEO_ID",
                              key=f"live_url_input_{seed}")
    submitted = st.form_submit_button("🔗 Periksa URL", use_container_width=True)

if submitted and url_input:
    st.session_state.live.update({"url_input": url_input, "last_result": None, "preview_frame": None, "running": False})
    with st.spinner("Memvalidasi URL siaran langsung..."):
        source_info, msg = fetch_live_stream_source(url_input)
    if source_info:
        st.toast("URL siaran langsung berhasil divalidasi!", icon="✅")
        st.session_state.live["source_info"] = source_info
        if msg: st.info(msg, icon="ℹ️")
    else:
        st.error(msg or "Terjadi kesalahan yang tidak diketahui.", icon="❌")
        st.session_state.live["source_info"] = None
elif submitted:
    st.toast("Input URL kosong. Silakan masukkan URL.", icon="⚠️")

if st.session_state.live.get("source_info"):
    st.subheader("**🎬 Pratinjau Siaran Langsung**")
    source_info = st.session_state.live["source_info"]
    stream_url = source_info.get("src")

    if st.session_state.live.get("preview_frame") is None and stream_url:
        with st.spinner("Mengambil gambar pratinjau dari stream..."):
            cap = cv2.VideoCapture(stream_url)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                if ret:
                    st.session_state.live["preview_frame"] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if st.session_state.live.get("preview_frame"):
        # --- PERUBAHAN DIMULAI DI SINI: Logika Pratinjau yang Disederhanakan ---
        
        # Tetapkan ukuran dasar pemutar. Lebar 800px adalah nilai maksimum yang baik.
        player_max_width = 800
        # Hitung tinggi berdasarkan rasio aspek standar (16:9) agar konsisten.
        player_height = int(player_max_width * (9 / 16))

        display_url = source_info["display_url"]
        stream_url = source_info.get("src")
        
        player_html = ""
        
        # 1. Logika untuk YouTube
        if "youtube.com" in display_url or "youtu.be" in display_url:
            match = re.search(r"(?:v=|\/|live\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", display_url)
            if match:
                video_id = match.group(1)
                # Gunakan iframe dengan ukuran yang sudah dihitung
                player_html = f'<iframe src="https://www.youtube.com/embed/{video_id}?autoplay=0&mute=1" width="100%" height="{player_height}px" style="border:none;" allow="autoplay; fullscreen"></iframe>'
        
        # 2. Logika untuk Twitch
        elif "twitch.tv" in display_url:
            match = re.search(r"twitch\.tv/([a-zA-Z0-9_]+)", display_url)
            if match:
                channel_name = match.group(1)
                hostname = urlparse(st.get_option("server.baseUrlPath")).hostname or "localhost"
                # Gunakan embed resmi dengan ukuran yang sudah dihitung
                player_html = f"""
                    <div id="twitch-embed"></div>
                    <script src="https://embed.twitch.tv/embed/v1.js"></script>
                    <script type="text/javascript">
                      new Twitch.Embed("twitch-embed", {{
                        width: "100%",
                        height: {player_height},
                        channel: "{channel_name}",
                        layout: "video",
                        parent: ["{hostname}"]
                      }});
                    </script>
                """

        # 3. Logika Fallback untuk platform lain
        if not player_html and source_info.get("type") != "rtsp":
            # Gunakan tag <video> standar dengan ukuran yang dihitung
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
        
        # Render pemutar yang terpilih di dalam wadah yang terpusat
        if player_html:
            components.html(
                f'<div style="max-width:{player_max_width}px; margin: auto;">{player_html}</div>',
                # Berikan sedikit ruang ekstra (misal, 40px) untuk memastikan tidak ada yang terpotong
                height=player_height + 40 
            )
        else: # Fallback untuk RTSP
             _, col_img, _ = st.columns([1, 2, 1])
             with col_img:
                st.image(st.session_state.live["preview_frame"], caption="Pratinjau Statis dari Stream", use_container_width=True)
        # --- AKHIR DARI PERUBAHAN ---

    else:
        st.warning("Tidak dapat memuat pratinjau stream untuk ditampilkan.")

# (Sisa skrip dari sini ke bawah tetap sama seperti versi sebelumnya dan sudah benar)
# --- Langkah 2: Konfigurasi Analisis ---
section_divider("Langkah 2: Konfigurasi Analisis", "⚙️")
def update_live_config(key_to_update, widget_key):
    st.session_state.live[key_to_update] = st.session_state[widget_key]
is_config_disabled = not st.session_state.live.get("source_info")
with st.container(border=True):
    col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
    with col1:
        st.slider("Interval antar-*frame* (detik)", 10, 600, 
                  value=st.session_state.live.get("interval", 10), 
                  step=5, key="live_interval_widget", on_change=update_live_config,
                  args=("interval", "live_interval_widget"), disabled=is_config_disabled)
    with col2:
        st.selectbox("Metode ROI:", ["Otomatis", "Manual (Kotak)", "Manual (Poligon)"], 
                     index=["Otomatis", "Manual (Kotak)", "Manual (Poligon)"].index(st.session_state.live.get("roi_method", "Otomatis")),
                     key="live_roi_widget", on_change=update_live_config, args=("roi_method", "live_roi_widget"),
                     disabled=is_config_disabled)
    with col3:
        st.toggle("Simpan hasil ke riwayat", 
                  value=st.session_state.live.get("save_to_history", True),
                  key="live_save_widget", on_change=update_live_config, args=("save_to_history", "live_save_widget"),
                  disabled=is_config_disabled)

if "Manual" in st.session_state.live.get("roi_method", "Otomatis") and not is_config_disabled:
    with st.expander("✏️ Gambar ROI Manual", expanded=True):
        if st.session_state.live.get("preview_frame"):
            st.info("Gambar bentuk pada kanvas di bawah untuk menandai area langit yang ingin dianalisis.")
            bg_image = st.session_state.live["preview_frame"]
            w, h = bg_image.size
            ratio = h / w if w > 0 else 1
            canvas_w, canvas_h = 512, int(512 * ratio)
            if canvas_h > 600:
                canvas_h = 600
                canvas_w = int(canvas_h / ratio) if ratio > 0 else 512
            drawing_mode = "rect" if "Kotak" in st.session_state.live["roi_method"] else "polygon"
            canvas_result = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)", stroke_width=2,
                background_image=bg_image.resize((canvas_w, canvas_h)),
                height=canvas_h, width=canvas_w, 
                drawing_mode=drawing_mode, key="live_canvas"
            )
            st.session_state.live['canvas'] = canvas_result
        else:
            st.warning("Tidak bisa menampilkan pratinjau untuk kanvas ROI.")

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

info_placeholder = st.empty()
result_placeholder = st.empty()

if not st.session_state.live.get("running") and st.session_state.live.get("last_result"):
    with result_placeholder.container():
        render_result(st.session_state.live.get("last_result"))

if st.session_state.live.get("running"):
    info_placeholder.info(f"Monitoring... Menganalisis setiap {st.session_state.live.get('interval', 10)} detik...", icon="🛰️")
    st.toast("Monitoring dimulai!", icon="👀")
    source_info = st.session_state.live["source_info"]
    stream_url = source_info.get("src")
    if not stream_url:
        st.error("URL stream tidak valid. Monitoring dihentikan.")
        st.session_state.live["running"] = False
        st.rerun()
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        st.error(f"Gagal membuka stream dari URL. Monitoring dihentikan.")
        st.session_state.live["running"] = False
        st.rerun()
    pipeline_hash = get_pipeline_version_hash()
    last_analysis_time = 0
    consecutive_failures = 0
    MAX_FAILURES = 5
    while st.session_state.live.get("running"):
        current_time_for_loop = time.time()
        if current_time_for_loop - last_analysis_time < st.session_state.live.get("interval", 10):
            time.sleep(1)
            continue
        analysis_start_time = time.time()
        last_analysis_time = analysis_start_time
        frame = get_frame_from_stream(cap)
        if frame is None:
            consecutive_failures += 1
            info_placeholder.warning(f"Gagal mengambil frame (percobaan {consecutive_failures}/{MAX_FAILURES})...")
            if consecutive_failures >= MAX_FAILURES:
                st.error("Gagal mengambil frame beberapa kali. Stream berakhir. Monitoring dihentikan.")
                st.session_state.live["running"] = False
            continue
        consecutive_failures = 0
        info_placeholder.info(f"Monitoring... Menganalisis setiap {st.session_state.live.get('interval', 10)} detik...", icon="🛰️")
        
        # --- BLOK ANALISIS (TIDAK BERUBAH) ---
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        live_config = {"roi_method": st.session_state.live.get("roi_method", "Otomatis"), "canvas": st.session_state.live.get("canvas"), "interval": st.session_state.live.get("interval", 10)}
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
        base_original_dir, base_mask_dir, base_overlay_dir = (config['paths']['original_archive'], config['paths']['mask_archive'], config['paths']['overlay_archive']) if is_saving_permanently else (os.path.join(config['paths']['temp_dir'], "live_session_artefacts", "original"), os.path.join(config['paths']['temp_dir'], "live_session_artefacts", "masks"), os.path.join(config['paths']['temp_dir'], "live_session_artefacts", "overlays"))
        original_path, mask_path, overlay_path = (os.path.join(base_original_dir, timestamp_name, f"{timestamp_name}_original.png"), os.path.join(base_mask_dir, timestamp_name, f"{timestamp_name}_mask.png"), os.path.join(base_overlay_dir, timestamp_name, f"{timestamp_name}_overlay.png"))
        for p in [original_path, mask_path, overlay_path]: os.makedirs(os.path.dirname(p), exist_ok=True)
        pil_frame.save(original_path, "PNG")
        Image.fromarray(analysis_data['segmentation_mask'] * 255).save(mask_path)
        create_enhanced_overlay(pil_frame, analysis_data['segmentation_mask'], analysis_data['roi_mask']).save(overlay_path)
        db_entry = {**analysis_data, "pipeline_version_hash": get_pipeline_version_hash(), "file_hash": file_hash, "analysis_hash": analysis_hash, "source_filename": f"Live Frame ({datetime.now():%Y-%m-%d %H:%M:%S} UTC){' (Unsaved)' if not is_saving_permanently else ''}", "media_type": "live_frame", "file_size_bytes": file_size, "analyzed_at": datetime.now(timezone(timedelta(hours=7))).isoformat(), "analysis_duration_sec": time.time() - analysis_start_time, "original_path": os.path.relpath(original_path).replace("\\", "/"), "mask_path": os.path.relpath(mask_path).replace("\\", "/"), "overlay_path": os.path.relpath(overlay_path).replace("\\", "/")}
        if is_saving_permanently: add_history_entry(db_entry)
        st.session_state.live["session_results"].append(db_entry)
        st.session_state.live["last_result"] = db_entry
        with result_placeholder.container(): render_result(db_entry)

    if cap: cap.release()
    if not st.session_state.live.get("running"): st.rerun()

if st.session_state.live.get("session_results") and not st.session_state.live.get("running"):
    df_session_results = pd.DataFrame(st.session_state.live["session_results"])
    render_summary_dashboard(df_session_results, title="Rangkuman Sesi Monitoring")
    section_divider("Unduh Hasil Sesi Ini", "📥")
    download_controller(st.session_state.live["session_results"], context="live")
