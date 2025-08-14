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


# --- PERUBAHAN 1: KEMBALIKAN FUNGSI get_frame_via_download UNTUK YOUTUBE ---
def get_frame_via_download(stream_url: str) -> Optional[np.ndarray]:
    """
    Mengunduh segmen pendek dari stream menggunakan yt-dlp.
    Ini adalah metode fallback yang andal khusus untuk YouTube.
    """
    temp_clip_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_clip:
            temp_clip_path = tmp_clip.name

        ydl_opts = {
            'outtmpl': temp_clip_path,
            'format': 'best[height<=720][ext=mp4]/best[ext=mp4]/best', # Batasi resolusi untuk kecepatan
            'quiet': True,
            'download_ranges': yt_dlp.utils.download_range_func(None, [(0, 3)]), # Unduh 3 detik
            'postprocessors': [{'key': 'FFmpegVideoRemuxer', 'preferedformat': 'mp4'}],
            'overwrites': True,
            # Tambahkan user-agent agar terlihat seperti browser biasa
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([stream_url])
        
        cap = cv2.VideoCapture(temp_clip_path)
        if not cap.isOpened():
            return None
        
        ret, frame = cap.read()
        cap.release()
        
        return frame if ret else None
    
    except Exception as e:
        print(f"Error dalam fungsi get_frame_via_download: {e}")
        return None
    finally:
        if temp_clip_path and os.path.exists(temp_clip_path):
            os.remove(temp_clip_path)

def get_frame_from_stream(cap: cv2.VideoCapture) -> Optional[np.ndarray]:
    """
    Membaca satu frame dari objek VideoCapture (untuk Twitch dan lainnya).
    """
    if not cap or not cap.isOpened():
        return None
    try:
        ret, frame = cap.read()
        return frame if ret else None
    except Exception as e:
        print(f"Error saat membaca frame dari stream: {e}")
        return None
# --- AKHIR PERUBAHAN 1 ---


# --- Konfigurasi Halaman & Inisialisasi State (Tidak ada perubahan di sini) ---
st.set_page_config(page_title=f"Live Monitoring - {config['app']['title']}", layout="wide")
if "widget_seed" not in st.session_state: st.session_state.widget_seed = 0
if "live" not in st.session_state:
    st.session_state.live = {
        "running": False, "source_info": None, "last_result": None,
        "preview_frame": None, "roi_method": "Otomatis", "interval": 10,
        "canvas": None, "url_input": "", "session_results": [],
        "save_to_history": True
    }
if "toast_message" in st.session_state and st.session_state.toast_message:
    message, icon = st.session_state.toast_message
    st.toast(message, icon=icon)
    st.session_state.toast_message = None
cleanup_temp_files(); apply_global_styles(); render_sidebar_footer()
@st.cache_resource
def get_models(): return load_segmentation_model(), load_classification_model()
seg_model, cls_model = get_models()
with st.sidebar:
    if st.button("🔄️ Mulai Sesi Monitoring Baru", use_container_width=True):
        st.session_state.live = {
            "running": False, "source_info": None, "last_result": None, "preview_frame": None, 
            "roi_method": "Otomatis", "interval": 10, "canvas": None, "url_input": "", 
            "session_results": [], "save_to_history": True
        }
        st.session_state.toast_message = ("Sesi monitoring baru dimulai.", "🔄️")
        st.session_state.widget_seed += 1
        st.rerun()

# --- Render UI & Logika Utama ---
render_page_header("Monitoring Awan Real-Time")
st.write("Analisis tutupan dan jenis awan secara otomatis dari siaran langsung (*live stream*).")
section_divider("Langkah 1: Masukkan URL & Lihat Pratinjau", "📡")
st.markdown("Tempelkan tautan **siaran langsung** dari platform seperti YouTube atau Twitch untuk memulai.")
seed = st.session_state.widget_seed
with st.form(key=f"url_form_live_{seed}"):
    url_input = st.text_input("Tempel URL siaran langsung di sini",
                              placeholder="https://www.youtube.com/live/VIDEO_ID",
                              key=f"live_url_input_{seed}")
    submitted = st.form_submit_button("🔗 Periksa URL", use_container_width=True)

if submitted and url_input:
    st.session_state.live["url_input"] = url_input
    st.session_state.live.update({"last_result": None, "preview_frame": None, "running": False})
    with st.spinner("Memvalidasi URL siaran langsung..."):
        source_info, msg = fetch_live_stream_source(url_input)
    if source_info:
        st.toast("URL siaran langsung berhasil divalidasi!", icon="✅")
        st.session_state.live["source_info"] = source_info
        if msg: st.info(msg, icon="ℹ️")
    else:
        st.error(msg or "Terjadi kesalahan.", icon="❌")
        st.session_state.live["source_info"] = None

if st.session_state.live.get("source_info"):
    st.subheader("**🎬 Pratinjau Siaran Langsung**")
    source_info = st.session_state.live["source_info"]
    stream_url = source_info.get("src")
    display_url = source_info.get("display_url")

    # Ambil frame pratinjau jika belum ada
    if st.session_state.live.get("preview_frame") is None and stream_url:
        with st.spinner("Mengambil gambar pratinjau..."):
            frame = None
            if 'youtube.com' in display_url or 'youtu.be' in display_url:
                frame = get_frame_via_download(display_url)
            else:
                cap = cv2.VideoCapture(stream_url)
                if cap.isOpened():
                    ret, frame_cap = cap.read()
                    if ret: frame = frame_cap
                    cap.release()
            if frame is not None:
                st.session_state.live["preview_frame"] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if st.session_state.live.get("preview_frame"):
        preview_img = st.session_state.live["preview_frame"]
        w, h = preview_img.size
        aspect_ratio_padding = (h / w * 100) if w > 0 else 75.0
        player_html = ""

        # --- PERUBAHAN 2: TAMBAHKAN LOGIKA PLAYER KHUSUS UNTUK TWITCH ---
        if "youtube.com" in display_url or "youtu.be" in display_url:
            match = re.search(r"(?:v=|\/|live\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", display_url)
            if match:
                video_id = match.group(1)
                player_html = f'<iframe src="https://www.youtube.com/embed/{video_id}?autoplay=0" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allowfullscreen></iframe>'
        
        elif "twitch.tv" in display_url:
            match = re.search(r"twitch\.tv/([a-zA-Z0-9_]+)", display_url)
            if match:
                channel = match.group(1)
                # Anda mungkin perlu mengganti "your-streamlit-app-domain.com" dengan domain asli aplikasi Anda
                # Namun, seringkali cukup dengan domain parent saja.
                parent_domain = st.secrets.get("DOMAIN", "localhost") 
                player_html = f'<iframe src="https://player.twitch.tv/?channel={channel}&parent={parent_domain}&muted=true" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;" frameborder="0" allow="autoplay; fullscreen" scrolling="no"></iframe>'

        # Fallback untuk platform lain atau jika embed gagal
        if not player_html and source_info.get("type") != "rtsp":
            player_html = f"""
                <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
                <video id="live-video" controls muted style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></video>
                <script>
                  var video = document.getElementById('live-video');
                  if(Hls.isSupported()) {{
                    var hls = new Hls();
                    hls.loadSource("{stream_url}");
                    hls.attachMedia(video);
                  }}
                </script>
            """
        # --- AKHIR PERUBAHAN 2 ---
        
        if player_html:
            height_val = (h / w * 800) if w > 0 and w < 2000 else 600
            components.html(
                f'<div style="max-width: 800px; margin: auto;"><div style="position: relative; width: 100%; padding-bottom: {aspect_ratio_padding}%; height: 0;">{player_html}</div></div>',
                height=height_val + 20
            )
        else: # Fallback untuk RTSP atau jika player gagal dibuat
             st.image(st.session_state.live["preview_frame"], caption="Pratinjau Statis dari Stream", use_container_width=True)
    else:
        st.warning("Tidak dapat memuat pratinjau stream.")


# --- Konfigurasi Analisis (Tidak ada perubahan di sini) ---
section_divider("Langkah 2: Konfigurasi Analisis", "⚙️")
def update_live_config(key_to_update, widget_key): st.session_state.live[key_to_update] = st.session_state[widget_key]
is_config_disabled = not st.session_state.live.get("source_info")
with st.container(border=True):
    # ... (UI Konfigurasi tidak berubah)
    col1, col2, col3 = st.columns([0.5, 0.25, 0.25])
    with col1:
        interval_key = "live_interval_widget"
        st.slider("Interval antar-*frame* (detik)", 10, 600, 
                  value=st.session_state.live.get("interval", 10), 
                  step=5, key=interval_key, on_change=update_live_config,
                  args=("interval", interval_key), disabled=is_config_disabled)
    with col2:
        roi_options = ["Otomatis", "Manual (Kotak)", "Manual (Poligon)"]
        try:
            current_roi_index = roi_options.index(st.session_state.live.get("roi_method", "Otomatis"))
        except ValueError: current_roi_index = 0
        roi_key = "live_roi_widget"
        st.selectbox("Metode ROI:", roi_options, index=current_roi_index, key=roi_key,
                     on_change=update_live_config, args=("roi_method", roi_key),
                     disabled=is_config_disabled)
    with col3:
        save_key = "live_save_widget"
        st.toggle("Simpan hasil ke riwayat", 
                  value=st.session_state.live.get("save_to_history", True),
                  key=save_key, on_change=update_live_config, args=("save_to_history", save_key),
                  disabled=is_config_disabled)
if "Manual" in st.session_state.live.get("roi_method", "Otomatis") and not is_config_disabled:
    with st.expander("✏️ Gambar ROI Manual", expanded=True):
        if st.session_state.live.get("preview_frame"):
            # ... (UI Canvas tidak berubah)
            bg_image = st.session_state.live["preview_frame"]
            w, h = bg_image.size
            ratio = h / w if w > 0 else 1
            canvas_w, canvas_h = 512, int(512 * ratio)
            if canvas_h > 600: canvas_h = 600; canvas_w = int(canvas_h / ratio) if ratio > 0 else 512
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

# --- Jalankan Monitoring & Tampilkan Hasil ---
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
    with result_placeholder.container(): render_result(st.session_state.live.get("last_result"))


# --- PERUBAHAN 3: LOGIKA LOOP UTAMA DENGAN STRATEGI HIBRIDA ---
if st.session_state.live.get("running"):
    info_placeholder.info(f"Monitoring sedang berlangsung...", icon="🛰️")
    st.toast("Monitoring dimulai!", icon="👀")

    source_info = st.session_state.live["source_info"]
    is_youtube = 'youtube.com' in source_info['display_url'] or 'youtu.be' in source_info['display_url']
    
    cap = None
    if not is_youtube:
        # Untuk Twitch dan lainnya, buka stream dengan OpenCV
        cap = cv2.VideoCapture(source_info.get("src"))
        if not cap.isOpened():
            st.error("Gagal membuka stream. Monitoring dihentikan.")
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
        
        frame = None
        if is_youtube:
            # Gunakan metode download untuk YouTube
            frame = get_frame_via_download(source_info['display_url'])
        else:
            # Gunakan metode baca stream untuk Twitch/lainnya
            frame = get_frame_from_stream(cap)
        
        if frame is None:
            consecutive_failures += 1
            info_placeholder.warning(f"Gagal mengambil frame (percobaan {consecutive_failures}/{MAX_FAILURES}).")
            if consecutive_failures >= MAX_FAILURES:
                st.error("Gagal mengambil frame beberapa kali. Stream mungkin berakhir. Monitoring dihentikan.")
                st.session_state.live["running"] = False
            continue

        consecutive_failures = 0
        info_placeholder.info(f"Menganalisis frame... (Interval: {st.session_state.live.get('interval', 10)} detik)", icon="🛰️")

        # --- BLOK ANALISIS (Tidak ada perubahan) ---
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # ... (Semua logika analisis Anda dari sini tetap sama persis)
        live_config = {
            "roi_method": st.session_state.live.get("roi_method", "Otomatis"),
            "canvas": st.session_state.live.get("canvas"),
            "interval": st.session_state.live.get("interval", 10)
        }
        user_roi_mask = None
        if "Manual" in live_config["roi_method"] and live_config["canvas"]:
            if live_config["canvas"].json_data and live_config["canvas"].json_data.get("objects"):
                user_roi_mask = canvas_to_mask(live_config["canvas"], pil_frame.height, pil_frame.width)
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
        for p in [original_path, mask_path, overlay_path]: os.makedirs(os.path.dirname(p), exist_ok=True)
        pil_frame.save(original_path, "PNG")
        Image.fromarray(analysis_data['segmentation_mask'] * 255).save(mask_path)
        overlay_img = create_enhanced_overlay(pil_frame, analysis_data['segmentation_mask'], analysis_data.get('roi_mask'))
        overlay_img.save(overlay_path)
        db_entry = {
            **analysis_data, "pipeline_version_hash": get_pipeline_version_hash(),
            "file_hash": file_hash, "analysis_hash": analysis_hash,
            "source_filename": f"Live Frame ({datetime.now():%Y-%m-%d %H:%M:%S} UTC)",
            "media_type": "live_frame", "file_size_bytes": file_size,
            "analyzed_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
            "analysis_duration_sec": time.time() - analysis_start_time,
            "original_path": os.path.relpath(original_path).replace("\\", "/"),
            "mask_path": os.path.relpath(mask_path).replace("\\", "/"),
            "overlay_path": os.path.relpath(overlay_path).replace("\\", "/"),
        }
        if is_saving_permanently: add_history_entry(db_entry)
        st.session_state.live["session_results"].append(db_entry)
        st.session_state.live["last_result"] = db_entry
        with result_placeholder.container(): render_result(db_entry)
        # --- Akhir Blok Analisis ---

    if cap: cap.release()
    if not st.session_state.live.get("running"): st.rerun()
# --- AKHIR PERUBAHAN 3 ---


# --- Rangkuman Sesi (Tidak ada perubahan di sini) ---
if st.session_state.live.get("session_results") and not st.session_state.live.get("running"):
    df_session_results = pd.DataFrame(st.session_state.live["session_results"])
    render_summary_dashboard(df_session_results, title="Rangkuman Sesi Monitoring")
    section_divider("Unduh Hasil Sesi Ini", "📥")
    download_controller(st.session_state.live["session_results"], context="live")
