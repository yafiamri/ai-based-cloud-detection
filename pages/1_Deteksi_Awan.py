# pages/1_Deteksi_Awan.py
import streamlit as st
import os
import io
import cv2
import shutil
import tempfile
import subprocess
import time
import pandas as pd
from PIL import Image
from datetime import datetime, timezone, timedelta
from streamlit_drawable_canvas import st_canvas
from typing import List, Dict, Any, IO, Tuple

# Impor semua fondasi dari utils
from utils.config import config
from utils.database import add_history_entry, find_history, get_pipeline_version_hash
from utils.layout import apply_global_styles, render_page_header, render_sidebar_footer, section_divider, render_result, render_summary_dashboard
from utils.media import extract_media_from_zip, load_demo_files, fetch_media_from_url, get_preview_as_pil, get_video_metadata 
from utils.processing import get_file_hash, get_analysis_hash, analyze_single_image, create_enhanced_overlay
from utils.segmentation import load_segmentation_model, canvas_to_mask
from utils.classification import load_classification_model
from utils.download import download_controller
from utils.system import cleanup_temp_files

# --- Fungsi Helper Spesifik Halaman ---
def _find_executable(name: str) -> str:
    """Mencari path executable. Memprioritaskan folder 'bin/' lokal."""
    local_bin_path = os.path.join(os.getcwd(), 'bin', f"{name}.exe" if os.name == 'nt' else name)
    if os.path.exists(local_bin_path): return local_bin_path
    if path := shutil.which(name): return path
    st.error(f"Dependensi '{name}' tidak ditemukan. Pastikan ia ada di PATH sistem atau di dalam folder 'bin/' proyek.")
    st.stop()

FFMPEG_PATH = _find_executable("ffmpeg")
FFPROBE_PATH = _find_executable("ffprobe")

# --- 1. Konfigurasi Halaman & Inisialisasi State ---
st.set_page_config(page_title=f"Deteksi Awan - {config['app']['title']}", layout="wide")
   
# Inisialisasi session state untuk manajemen sesi yang persisten
if "widget_seed" not in st.session_state:
    st.session_state.widget_seed = 0
if "files_to_process" not in st.session_state:
    st.session_state.files_to_process = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []
if "configurations" not in st.session_state:
    st.session_state.configurations = {}
if "config_mode" not in st.session_state:
    st.session_state.config_mode = "Otomatis untuk Semua"
if "video_previews" not in st.session_state:
    st.session_state.video_previews = {}
if "video_durations" not in st.session_state:
    st.session_state.video_durations = {}
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = set()
if "staged_url_items" not in st.session_state:
    st.session_state.staged_url_items = []
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
    """Memuat model AI."""
    return load_segmentation_model(), load_classification_model()
seg_model, cls_model = get_models()

# Tombol untuk membersihkan sesi di sidebar
with st.sidebar:
    if st.button("🔄️ Mulai Sesi Analisis Baru"):
        # Hapus semua state sesi yang relevan
        keys_to_clear = [
            "files_to_process", "analysis_results", "configurations", 
            "config_mode", "video_previews", "processed_urls", "staged_url_items"
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        st.session_state.widget_seed += 1
        st.session_state.toast_message = ("Sesi analisis baru dimulai.", "🔄️")
        st.rerun()

# --- 2. Render UI & Logika Utama ---
render_page_header("Deteksi Tutupan dan Jenis Awan")
st.write(
    "Unggah gambar atau video, konfigurasikan analisis, dan dapatkan hasil deteksi awan secara otomatis, "
    "lengkap dengan *caching* untuk file yang pernah dianalisis."
)

# --- Langkah 1: Input Media & Manajemen Antrean ---
section_divider("Langkah 1: Unggah Citra Langit", "📤")

seed = st.session_state.widget_seed

def add_files_to_queue(files_to_add: List[IO[bytes]]):
    """Callback terpusat untuk menambahkan file ke antrean dengan aman dan anti-duplikat."""
    current_hashes = {get_file_hash(f) for f in st.session_state.files_to_process}
    added_count = 0
    duplicate_count = 0 # Tambahkan penghitung duplikat

    if not files_to_add:
        return

    with st.spinner("Memeriksa dan menyiapkan file..."):
        for file in files_to_add:
            file_hash = get_file_hash(file)
            if file_hash not in current_hashes:
                MAX_FILE_SIZE_MB = 200
                if file.getbuffer().nbytes > MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.warning(f"Peringatan: Ukuran file '{file.name}' ({file.getbuffer().nbytes / (1024*1024):.1f} MB) melebihi {MAX_FILE_SIZE_MB} MB. Proses mungkin lambat atau gagal karena batasan memori.")
                
                is_video = file.name.lower().endswith(tuple(config['analysis']['video_extensions']))
                if is_video:
                    # Panggil fungsi utilitas SEKALI untuk mendapatkan SEMUANYA
                    preview, duration = get_video_metadata(file)
                    st.session_state.video_previews[file_hash] = preview
                    st.session_state.video_durations[file_hash] = duration
                
                st.session_state.files_to_process.append(file)
                current_hashes.add(file_hash)
                added_count += 1
            else:
                duplicate_count += 1
    # Logika notifikasi toast yang baru dan lebih informatif
    if added_count > 0 and duplicate_count > 0:
        st.session_state.toast_message = (f"{added_count} berkas ditambahkan; {duplicate_count} duplikat diabaikan.", "👍")
    elif added_count > 0:
        st.session_state.toast_message = (f"{added_count} berkas baru berhasil ditambahkan.", "✅")
    elif duplicate_count > 0:
        st.session_state.toast_message = (f"{duplicate_count} berkas duplikat diabaikan.", "⚠️")
        
def remove_file_from_queue(index_to_remove):
    """
    Menghapus file dari antrean tanpa mereset widget input untuk menjaga konsistensi UI.
    """
    # Ambil file yang akan dihapus dari antrean utama
    file_to_remove = st.session_state.files_to_process.pop(index_to_remove)
    file_hash_to_del = get_file_hash(file_to_remove)
    
    # Hapus data terkait (konfigurasi dan pratinjau video)
    st.session_state.configurations.pop(file_to_remove.name, None)
    st.session_state.video_previews.pop(file_hash_to_del, None)
    
    # Beri notifikasi bahwa file telah dihapus dari antrean
    st.toast(f"'{file_to_remove.name}' dihapus dari antrean.", icon="🗑️")

def clear_queue():
    """Menghapus semua file dari antrean dan state terkait."""
    st.session_state.files_to_process.clear()
    st.session_state.configurations.clear()
    st.session_state.video_previews.clear()
    st.session_state.processed_urls.clear()
    st.session_state.staged_url_items.clear()
    st.toast("Semua antrean telah dibersihkan.", icon="✨")

def remove_staged_url_item(url_to_remove: str):
    """Menghapus item dari area persiapan URL berdasarkan URL-nya."""
    st.session_state.staged_url_items = [
        item for item in st.session_state.staged_url_items if item['url'] != url_to_remove
    ]
    st.toast(f"URL dihapus dari area persiapan.", icon="🗑️")

# UI Input Media (Demo, Unggah, URL)
st.markdown("Pilih atau unggah semua berkas yang Anda inginkan, lalu klik tombol 'Tambahkan' di bawah.")

# Opsi 1: Media Demo (Gambar & Video)
demo_media_data = load_demo_files()
demo_names = [d[0] for d in demo_media_data]
st.multiselect(
    "Pilih berkas demo untuk uji coba analisis:", 
    demo_names, 
    key=f"demo_selector_{seed}"
)

# Opsi 2: Unggah File Lokal
supported_types = config['analysis']['image_extensions'] + config['analysis']['video_extensions'] + [config['analysis']['zip_extension'].strip('.')]
st.file_uploader(
    "Atau unggah gambar, video, dan berkas arsip dari lokal:", 
    type=supported_types, 
    accept_multiple_files=True, 
    key=f"uploader_{seed}"
    # on_change dihapus dari sini
)

# Opsi 3: URL
with st.form(key=f"url_form_{seed}", border=False):
    col1, col2 = st.columns([0.85, 0.15])
    with col1:
        url_input = st.text_input("Atau tambahkan media dari URL:",
            placeholder="https://drive.google.com/uc?id=FILE_ID",
            help="Masukkan URL folder/berkas dari Google Drive, video non-streaming YouTube, atau tautan langsung ke media.",
            key=f"url_input_{seed}"
        )
    with col2:
        # Beri sedikit jarak dari atas agar sejajar dengan kotak input
        st.markdown("<br>", unsafe_allow_html=True)
        submit_url = st.form_submit_button("🔗 Periksa URL", use_container_width=True) 
    if submit_url:
        # Cek duplikat di antrean utama DAN di area persiapan
        staged_urls = [item['url'] for item in st.session_state.staged_url_items]
        if not url_input:
            st.toast("Input URL kosong.", icon="⚠️")
        elif url_input in st.session_state.processed_urls or url_input in staged_urls:
            st.toast("Media dari URL ini sudah ada di antrean atau area persiapan.", icon="🤷")
        else:
            with st.spinner("Memeriksa dan mengambil media dari URL..."):
                files_from_url, error_msg = fetch_media_from_url(url_input)
                if error_msg:
                    st.error(f"{error_msg}", icon="❌")
                elif files_from_url:
                    # Tambahkan item baru ke list 'staged_url_items'
                    new_item = {
                        "url": url_input,
                        "files": files_from_url,
                        "message": f"{len(files_from_url)} berkas ditemukan"
                    }
                    st.session_state.staged_url_items.append(new_item)
                    st.toast("URL berhasil disiapkan!", icon="✅")

# --- Tampilkan daftar URL di area persiapan ---
if st.session_state.staged_url_items:
    st.caption("⛓️ URL berikut sudah siap untuk ditambahkan ke antrean utama. Anda bisa membatalkannya jika tidak jadi.")
    for item in st.session_state.staged_url_items:
        col_info, col_btn = st.columns([0.85, 0.15])
        with col_info:
            # Tampilkan URL dan jumlah file yang ditemukan
            st.info(f"**URL ({item['message']})**: [{item['url']}]({item['url']})")
        with col_btn:
            # Tombol hapus untuk setiap item
            st.button(
                "❌ Batalkan", 
                key=f"delete_staged_{item['url']}", 
                on_click=remove_staged_url_item, 
                args=(item['url'],),
                use_container_width=True,
            )
st.markdown("---")

# --- Tombol terpusat untuk memproses semua input ---
if st.button("➕ Tambahkan Semua ke Antrean", use_container_width=True, type="primary"):
    all_files_to_add = []
    
    # 1. Kumpulkan file dari Demo
    selected_demo_names = st.session_state.get(f'demo_selector_{seed}', [])
    if selected_demo_names:
        for name, img, media_type in demo_media_data:
            if name in selected_demo_names:
                if media_type == "image" and img is not None:
                    buf = io.BytesIO(); img.save(buf, "PNG"); buf.name = name
                    all_files_to_add.append(buf)
                elif media_type == "video":
                    demo_dir = config['paths'].get('demo', 'assets/demo')
                    video_path = os.path.join(demo_dir, name)
                    if os.path.exists(video_path):
                        with open(video_path, "rb") as f:
                            buf = io.BytesIO(f.read()); buf.name = name
                            all_files_to_add.append(buf)

    # 2. Kumpulkan file dari Uploader
    uploaded_files = st.session_state.get(f'uploader_{seed}', [])
    if uploaded_files:
        with st.spinner("Mengekstrak file ZIP jika ada..."):
            for file in uploaded_files:
                if file.name.lower().endswith(config['analysis']['zip_extension']):
                    all_files_to_add.extend(extract_media_from_zip(file))
                else:
                    all_files_to_add.append(file)

    # 3. Ambil file dari "area persiapan" URL
    if st.session_state.staged_url_items:
        for item in st.session_state.staged_url_items:
            all_files_to_add.extend(item['files'])
            # Ingat URL yang sudah diproses
            st.session_state.processed_urls.add(item['url'])
    
    # 4. Tambahkan semua file yang terkumpul ke antrean
    if all_files_to_add:
        add_files_to_queue(all_files_to_add)
        st.rerun()
    else:
        st.toast("Tidak ada berkas baru untuk ditambahkan ke antrean.", icon="⚠️")

# --- Tampilan Antrean ---
# Hitung jumlah gambar dan video dalam antrean
video_ext = tuple(config['analysis']['video_extensions'])
image_ext = tuple(config['analysis']['image_extensions'])
num_images = len([f for f in st.session_state.files_to_process if f.name.lower().endswith(image_ext)])
num_videos = len([f for f in st.session_state.files_to_process if f.name.lower().endswith(video_ext)])
total_files = num_images + num_videos
# Buat string ringkasan yang detail
if total_files > 0:
    summary_parts = []
    if num_images > 0: summary_parts.append(f"{num_images} gambar")
    if num_videos > 0: summary_parts.append(f"{num_videos} video")
    count_str = " dan ".join(summary_parts)
    
    # Layout untuk ringkasan dan tombol hapus semua ---
    col_summary, col_clear_btn = st.columns([0.85, 0.15])
    with col_summary:
        st.success(f"🖼️ **Total**: {total_files} berkas siap diproses ({count_str})")
    with col_clear_btn:
        st.button(
            "🗑️ Bersihkan Antrean", 
            on_click=clear_queue, 
            use_container_width=True,
            help="Hapus semua berkas dari daftar antrean di bawah.",
            type="secondary"
        )

if not st.session_state.files_to_process:
    st.warning("Antrean analisis kosong. Silakan unggah atau pilih berkas untuk memulai.")
else:
    # 2. Buat label yang dinamis untuk expander
    expander_label = f"Lihat pratinjau atau pilih berkas untuk dihapus dari antrean:"

    # 3. Masukkan seluruh galeri ke dalam st.expander
    with st.expander(expander_label, expanded=True): # expanded=True membuat terbuka secara default

        # Logika render galeri per baris
        files_in_queue = st.session_state.files_to_process
        num_cols = 4
        rows_of_files = [files_in_queue[i:i + num_cols] for i in range(0, len(files_in_queue), num_cols)]

        for row_index, files_in_row in enumerate(rows_of_files):
            cols = st.columns(num_cols)
            
            for col_index, file in enumerate(files_in_row):
                with cols[col_index]:
                    i = (row_index * num_cols) + col_index
                    
                    try:
                        is_video = file.name.lower().endswith(video_ext)
                        if is_video:
                            st.video(file)
                        else:
                            st.image(file, use_container_width=True)
                    except Exception as e:
                        st.warning(f"⚠️ Gagal menampilkan pratinjau.")
                    
                    col_info, col_action = st.columns([4, 1])
                    with col_info:
                        st.caption(file.name)
                    with col_action:
                        st.button(
                            "🗑️",
                            key=f"del_{get_file_hash(file)}",
                            on_click=remove_file_from_queue,
                            args=(i,),
                            use_container_width=True,
                            help=f"Hapus {file.name}",
                            type="tertiary"
                        )

# --- Langkah 2: Konfigurasi Analisis ---
section_divider("Langkah 2: Konfigurasi Analisis", "⚙️")

# Fungsi callback untuk menangani perubahan pada widget konfigurasi
def update_file_config(file_name, key, widget_key):
    """Callback untuk memperbarui state konfigurasi file individual."""
    if file_name in st.session_state.configurations:
        st.session_state.configurations[file_name][key] = st.session_state[widget_key]

# Definisikan opsi di luar agar konsisten
config_options = ["Otomatis untuk Semua", "Manual per Berkas"]
roi_options = ["Otomatis", "Manual (Kotak)", "Manual (Poligon)", "Manual (Diameter Lingkaran)"]

# Tentukan indeks saat ini dari state
try:
    current_config_index = config_options.index(st.session_state.get("config_mode", config_options[0]))
except ValueError:
    current_config_index = 0

# Widget radio utama
st.radio(
    "Mode Konfigurasi:",
    config_options,
    horizontal=True,
    index=current_config_index,
    key="config_mode_widget", # Gunakan key baru untuk widget
    on_change=lambda: st.session_state.update(config_mode=st.session_state.config_mode_widget), # Callback untuk update state utama
    disabled=not st.session_state.files_to_process
)

# Logika Tampilan
if st.session_state.files_to_process:
    config_mode = st.session_state.get("config_mode", config_options[0])

    if config_mode == "Otomatis untuk Semua":
        st.info("Mode otomatis akan mendeteksi ROI lingkaran dan menggunakan interval 5 detik untuk video.")
        for file in st.session_state.files_to_process:
            st.session_state.configurations[file.name] = {'roi_method': 'Otomatis', 'interval': 5, 'canvas': None}
                       
    else: # Mode Manual
        st.info("Atur konfigurasi untuk setiap berkas di bawah ini. Konfigurasi Anda akan tersimpan selama sesi ini.")

        # Loop melalui setiap file untuk menampilkan opsi konfigurasinya
        for file in st.session_state.files_to_process:
            is_video = file.name.lower().endswith(tuple(config['analysis']['video_extensions']))
            
            # Inisialisasi konfigurasi untuk file ini jika belum ada
            if file.name not in st.session_state.configurations:
                st.session_state.configurations[file.name] = {'roi_method': 'Otomatis', 'interval': 5, 'canvas': None}
            
            file_config = st.session_state.configurations[file.name]

            with st.expander(f"Atur untuk: **{file.name}**", expanded=True):
                # Tentukan indeks saat ini untuk selectbox ROI
                try:
                    current_roi_index = roi_options.index(file_config.get('roi_method', 'Otomatis'))
                except ValueError:
                    current_roi_index = 0
                                    
                # Widget Selectbox ROI dengan callback
                roi_key = f"roi_{file.name}"
                st.selectbox(
                    "Metode ROI", roi_options,
                    index=current_roi_index,
                    key=roi_key,
                    on_change=update_file_config,
                    args=(file.name, 'roi_method', roi_key)
                )
                
                if is_video:
                    file_hash = get_file_hash(file)
                    
                    # Ambil durasi dari state, beri nilai default 60 jika tidak ada
                    max_duration = st.session_state.video_durations.get(file_hash, 60.0)
                    # Pastikan nilai maksimal minimal 1 detik
                    max_slider_value = max(1, int(max_duration))
                    
                    # Ambil nilai interval saat ini dari konfigurasi
                    current_interval = file_config.get('interval', 5)
                    # Pastikan nilai awal tidak melebihi nilai maksimal slider
                    safe_value = min(current_interval, max_slider_value)

                    interval_key = f"interval_{file.name}"
                    st.slider(
                        "Interval antar-frame (detik)", 
                        min_value=1, 
                        max_value=max_slider_value,  # Gunakan durasi dinamis
                        value=safe_value,            # Gunakan nilai yang aman
                        key=interval_key, 
                        on_change=update_file_config,
                        args=(file.name, 'interval', interval_key),
                        help="Seberapa sering analisis akan dijalankan pada video."
                    )
                
                # Logika kanvas ROI
                if "Manual" in file_config.get('roi_method', 'Otomatis'):
                    st.info("Gambar bentuk dengan melakukan *drag and drop* pada kanvas pratinjau di bawah untuk menandai area langit yang ingin dianalisis.")
                    
                    # Dapatkan background image langsung dari session_state atau file
                    is_video = file.name.lower().endswith(tuple(config['analysis']['video_extensions']))
                    if is_video:
                        bg_image = st.session_state.video_previews.get(get_file_hash(file))
                    else:
                        file.seek(0)
                        bg_image = Image.open(file)
                    
                    if bg_image:
                        drawing_mode = "rect" if "Kotak" in file_config.get('roi_method') else "polygon" if "Poligon" in file_config.get('roi_method') else "line"
                        # _, col_canvas, _ = st.columns([1, 2.5, 1])
                        # with col_canvas:
                        w, h = bg_image.size
                        ratio = h / w if w > 0 else 1
                        # 1. Tetapkan lebar dasar yang 'aman' (seperti yang sudah Anda lakukan)
                        canvas_w = 512
                        # 2. Hitung tinggi berdasarkan rasio
                        canvas_h = int(canvas_w * ratio)
                        # 3. Batasi tinggi maksimum untuk mencegah scrolling berlebih
                        MAX_CANVAS_HEIGHT = 600 # Anda bisa sesuaikan nilai ini
                        canvas_h = min(canvas_h, MAX_CANVAS_HEIGHT)
                        # 4. itung ulang lebar jika tingginya dipotong,
                        if canvas_h == MAX_CANVAS_HEIGHT and ratio > 0:
                            canvas_w = int(canvas_h / ratio)

                        canvas_result = st_canvas(
                            fill_color="rgba(255, 0, 0, 0.3)",
                            stroke_width=2,
                            background_image=bg_image.resize((canvas_w, canvas_h)),
                            height=canvas_h,
                            width=canvas_w,
                            drawing_mode=drawing_mode,
                            key=f"canvas_{file.name}"
                        )
            
                        # Simpan/update kanvas hasil ke konfigurasi
                        if canvas_result:
                            st.session_state.configurations[file.name]['canvas'] = canvas_result
                    else:
                        st.warning("Tidak bisa menampilkan pratinjau untuk kanvas ROI. Berkas mungkin korup atau formatnya tidak didukung.")

# --- Langkah 3: Orkestrasi Analisis ---
section_divider("Langkah 3: Jalankan Analisis", "🚀")

if st.button("🪄 Proses Semua Berkas dalam Antrean", type="primary", use_container_width=True, disabled=not st.session_state.files_to_process):
    # Hapus semua hasil sebelumnya saat analisis baru dimulai
    st.session_state.analysis_results = []
    analysis_start_time = time.time()
    files_to_run = st.session_state.files_to_process.copy()

    # Pra-Kalkulasi Total Langkah Analisis
    total_steps = 0
    with st.spinner("Menghitung total langkah analisis..."):
        for file in files_to_run:
            if file.name.lower().endswith(tuple(config['analysis']['video_extensions'])):
                try:
                    # Perlu membaca metadata video untuk menghitung frame
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                        tmp.write(file.getvalue())
                        temp_video_path = tmp.name
                    
                    cap = cv2.VideoCapture(temp_video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    os.remove(temp_video_path)

                    file_config = st.session_state.configurations.get(file.name, {'interval': 5})
                    interval_seconds = file_config.get('interval', 5)
                    frame_interval = max(1, int(fps * interval_seconds))
                    num_frames = len(range(0, frame_count, frame_interval))
                    total_steps += num_frames
                except Exception:
                    total_steps += 1 # Fallback jika gagal baca metadata
            else:
                total_steps += 1 # Gambar dihitung sebagai 1 langkah

    # Eksekusi Analisis dengan Progress Bar yang Akurat
    progress_bar = st.progress(0, "⏳ Memulai analisis...")
    step_counter = 0
    pipeline_hash = get_pipeline_version_hash()
    newly_analyzed_results = []
    
    for i, file in enumerate(files_to_run):
        # (Logika hash dan cek cache tetap sama)
        file_config = st.session_state.configurations.get(file.name, {'roi_method': 'Otomatis', 'interval': 5, 'canvas': None})
        file_hash = get_file_hash(file)
        analysis_hash = get_analysis_hash(file_hash, pipeline_hash, file_config)

        cached_result = find_history(analysis_hash)
        if cached_result:
            # Jika dari cache, perbarui progress sesuai jumlah langkahnya
            if cached_result['media_type'] == 'video':
                # Perkiraan kasar jika data lama tidak punya info frame
                step_counter += max(1, total_steps // len(files_to_run)) 
            else:
                step_counter += 1
            progress_bar.progress(min(1.0, step_counter / total_steps), f"Mengambil dari cache: {file.name}")
            st.toast(f"Hasil '{file.name}' ditemukan di cache.", icon="🗄️")
            newly_analyzed_results.append(dict(cached_result))
            continue
        
        # 3. Jika Tidak Ada di Cache, Lanjutkan Analisis
        st.toast(f"Mulai memproses '{file.name}'...", icon="🧠")
        is_video = file.name.lower().endswith(tuple(config['analysis']['video_extensions']))

        user_roi_mask = None
        if file_config['roi_method'] != 'Otomatis' and file_config.get('canvas'):
            try:
                if is_video:
                    preview_img = st.session_state.video_previews.get(file_hash)
                    if preview_img: user_roi_mask = canvas_to_mask(file_config['canvas'], preview_img.height, preview_img.width)
                else:
                    img_temp = Image.open(file); user_roi_mask = canvas_to_mask(file_config['canvas'], img_temp.height, img_temp.width)
            except Exception as e:
                st.warning(f"Gagal membuat ROI manual untuk '{file.name}': {e}. Menggunakan ROI otomatis.")

        # Siapkan entri untuk database
        db_entry = {
            "analysis_hash": analysis_hash,
            "file_hash": file_hash,
            "pipeline_version_hash": pipeline_hash,
            "source_filename": file.name,
            "media_type": "video" if is_video else "image",
            "file_size_bytes": file.getbuffer().nbytes,
            "analyzed_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        }

        try:
            if not is_video: # --- A. PROSES GAMBAR STATIS ---
                step_counter += 1
                progress_text = f"⏳ Menganalisis gambar: {file.name}"
                progress_bar.progress(step_counter / total_steps, text=progress_text)
                
                # Pastikan file berada di awal
                file.seek(0)

                # Baca gambar dari file, konversi ke RGB, dan analisis
                img = Image.open(file).convert("RGB")
                analysis_data = analyze_single_image(img, seg_model, cls_model, user_roi_mask)

                # `timestamp_name` sekarang menjadi nama FOLDER unik untuk analisis ini
                timestamp_name = f"{datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d%H%M%S%f')}UTC_{os.path.splitext(file.name)[0]}"

                # Buat path lengkap ke dalam subfolder unik           
                original_path = os.path.join(config['paths']['original_archive'], timestamp_name, f"{timestamp_name}_original.png")
                mask_path = os.path.join(config['paths']['mask_archive'], timestamp_name, f"{timestamp_name}_mask.png")
                overlay_path = os.path.join(config['paths']['overlay_archive'], timestamp_name, f"{timestamp_name}_overlay.png")

                # Buat semua direktori yang diperlukan (termasuk subfolder timestamp_name)
                for p in [original_path, mask_path, overlay_path]:
                    os.makedirs(os.path.dirname(p), exist_ok=True)

                # Simpan gambar asli, mask, dan overlay ke disk
                overlay_img = create_enhanced_overlay(img, analysis_data['segmentation_mask'], analysis_data['roi_mask'])
                img.save(original_path)
                Image.fromarray(analysis_data['segmentation_mask'] * 255).save(mask_path)
                overlay_img.save(overlay_path)

                # Ubah path absolut menjadi path relatif dari direktori kerja utama.
                # Ganti semua separator `\` menjadi `/` untuk konsistensi lintas platform.
                relative_original = os.path.relpath(original_path).replace("\\", "/")
                relative_mask = os.path.relpath(mask_path).replace("\\", "/")
                relative_overlay = os.path.relpath(overlay_path).replace("\\", "/")
                
                # Simpan path yang sudah bersih dan relatif ke database
                db_entry.update({
                    **analysis_data, 
                    "original_path": relative_original, 
                    "mask_path": relative_mask, 
                    "overlay_path": relative_overlay
                })

            else: # --- B. PROSES VIDEO LENGKAP ---
                temp_dir = tempfile.mkdtemp()
                try:
                    # Simpan file video dari memori ke disk sementara untuk diproses
                    temp_video_path = os.path.join(temp_dir, file.name)
                    with open(temp_video_path, "wb") as f:
                        f.write(file.getvalue())

                    # Dapatkan metadata video (FPS, jumlah frame)
                    cap = cv2.VideoCapture(temp_video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    if frame_count == 0:
                        raise ValueError("Video tidak memiliki frame atau formatnya tidak dapat dibaca.")

                    # Tentukan frame mana saja yang akan dianalisis berdasarkan interval
                    interval_seconds = file_config.get('interval', 5)
                    frame_interval = max(1, int(fps * interval_seconds))
                    frame_indices = list(range(0, frame_count, frame_interval))

                    # Direktori untuk menyimpan frame overlay dan mask dibuat di awal
                    temp_overlay_dir = os.path.join(temp_dir, "overlay_frames")
                    os.makedirs(temp_overlay_dir, exist_ok=True)

                    results_per_frame = [] # List untuk mengumpulkan data analisis (JSON, bukan gambar)

                    # Loop utama untuk menganalisis setiap frame yang dipilih
                    for idx, frame_num in enumerate(frame_indices):
                        step_counter += 1
                        progress_text = f"⏳ Menganalisis video: '{file.name}' (frame {idx + 1}/{len(frame_indices)})"
                        progress_bar.progress(step_counter / total_steps, text=progress_text)

                        cap = cv2.VideoCapture(temp_video_path)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        ret, frame = cap.read()
                        cap.release()
                        if not ret:
                            continue

                        # Analisis frame tunggal
                        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        analysis_data = analyze_single_image(img, seg_model, cls_model, user_roi_mask)
                        results_per_frame.append(analysis_data)

                        # Buat gambar overlay dan langsung simpan ke disk
                        sticker_img = create_enhanced_overlay(
                            original_image=img, 
                            segmentation_mask=analysis_data['segmentation_mask'], 
                            roi_mask=analysis_data['roi_mask'],
                            as_sticker=True
                        )
                        sticker_img.save(os.path.join(temp_overlay_dir, f"sticker_{idx:06d}.png"), "PNG")

                    if not results_per_frame:
                        raise ValueError("Tidak ada frame yang dapat diproses dari video.")
                    
                    progress_text = f"⏳ Mengagregasi video: {file.name}"
                    progress_bar.progress(step_counter / total_steps, text=progress_text)

                    # Agregasi hasil dari semua frame yang dianalisis
                    df_results = pd.DataFrame(results_per_frame)
                    avg_coverage = df_results['cloud_coverage'].mean()
                    final_okta = int(round((avg_coverage / 100) * 8))
                    final_sky_condition_index = min(final_okta // 2, len(config['analysis']['sky_conditions']) - 1)
                    final_sky_condition = config['analysis']['sky_conditions'][final_sky_condition_index]
                    final_dominant_cloud = df_results['dominant_cloud_type'].mode()[0] if not df_results['dominant_cloud_type'].empty else "Tidak terdeteksi"

                    # Tambahan: Agregasi confidence semua jenis awan
                    cloud_types = []
                    cloud_confidences = {}
                    for result in results_per_frame:
                        for cloud, conf in result.get('cloud_type_confidences', {}).items():
                            cloud_types.append(cloud)
                            cloud_confidences.setdefault(cloud, []).append(conf)
                    # Hitung rata-rata confidence tiap jenis awan
                    cloud_confidence_summary = {cloud: sum(confs)/len(confs)*100 for cloud, confs in cloud_confidences.items()}
                    # Urutkan berdasarkan confidence tertinggi
                    sorted_clouds = sorted(cloud_confidence_summary.items(), key=lambda x: x[1], reverse=True)
                    details_list = [f"Rata-rata dari {len(results_per_frame)} frame"]
                    for cloud, conf in sorted_clouds:
                        details_list.append(f"{cloud} ({conf:.2f}%)")
                    # Simpan ke database sebagai string agar tidak error binding
                    db_entry.update({
                        "cloud_coverage": avg_coverage, "okta_value": final_okta,
                        "sky_condition": final_sky_condition, "dominant_cloud_type": final_dominant_cloud,
                        "classification_details": "; ".join(details_list)
                    })

                    # `timestamp_name` sekarang menjadi nama FOLDER unik untuk analisis ini
                    timestamp_name = f"{datetime.now(timezone(timedelta(hours=7))).strftime('%Y%m%d%H%M%S%f')}UTC_{os.path.splitext(file.name)[0]}"

                    # Buat path lengkap ke dalam subfolder unik untuk setiap jenis artefak
                    original_path = os.path.join(config['paths']['original_archive'], timestamp_name, f"{timestamp_name}_original.mp4")
                    overlay_path = os.path.join(config['paths']['overlay_archive'], timestamp_name, f"{timestamp_name}_overlay.mp4")
                    archive_mask_dir = os.path.join(config['paths']['mask_archive'], timestamp_name)

                    # Buat direktori untuk file video original dan overlay
                    os.makedirs(os.path.dirname(original_path), exist_ok=True)
                    os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
                    # Buat direktori untuk mask video secara eksplisit
                    os.makedirs(archive_mask_dir, exist_ok=True)

                    # Simpan video asli ke arsip
                    shutil.copy(temp_video_path, original_path)

                    # Hitung framerate untuk stream overlay
                    video_duration = frame_count / fps if fps > 0 else 0
                    overlay_framerate = len(frame_indices) / video_duration if video_duration > 0 else 1.0

                    # Jahit frame overlay menjadi video menggunakan FFmpeg
                    ffmpeg_cmd = [
                        FFMPEG_PATH, '-y',
                        '-i', original_path, # Input 0: Video asli
                        '-framerate', str(overlay_framerate),
                        '-i', os.path.join(temp_overlay_dir, 'sticker_%06d.png'), # Input 1: Gambar overlay
                        '-filter_complex', "[0:v][1:v]overlay=shortest=1:format=auto",
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-preset', 'veryfast',
                        overlay_path
                    ]

                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
                    if result.returncode != 0:
                        st.error(f"Gagal membuat video overlay untuk '{file.name}'.")
                        st.code(result.stderr)
                        overlay_path = None

                    # Simpan frame-frame mask ke arsip
                    for idx, result_data in enumerate(results_per_frame):
                        Image.fromarray(result_data['segmentation_mask'] * 255).save(os.path.join(archive_mask_dir, f"mask_{idx:06d}.png"))

                    # Update entri database dengan path yang sudah bersih dan relatif
                    db_entry.update({
                        "original_path": os.path.relpath(original_path).replace("\\", "/"), 
                        "mask_path": os.path.relpath(archive_mask_dir).replace("\\", "/"), 
                        "overlay_path": os.path.relpath(overlay_path).replace("\\", "/") if overlay_path else None
                    })
                
                finally:
                    # Pastikan direktori sementara selalu dibersihkan, bahkan jika terjadi error
                    shutil.rmtree(temp_dir)
                            
            db_entry["analysis_duration_sec"] = time.time() - analysis_start_time
            add_history_entry(db_entry)
            newly_analyzed_results.append(db_entry)

        except Exception as e:
            st.error(f"Gagal memproses file '{file.name}': {e}")
    
    progress_bar.empty()
    st.toast("Semua berkas telah selesai dianalisis!", icon="🎉")
    st.success(f"Analisis total selesai dalam {time.time() - analysis_start_time:.2f} detik!")

    # Update hasil sesi
    st.session_state.analysis_results = newly_analyzed_results

# --- Langkah 4: Tampilkan Hasil ---
if st.session_state.analysis_results:
    # Menampilkan semua kartu hasil analisis
    for result in st.session_state.analysis_results:
        render_result(result)
    
    # Ubah hasil sesi menjadi DataFrame
    df_session_results = pd.DataFrame(st.session_state.analysis_results)
    # Panggil fungsi dasbor universal untuk merangkum sesi ini
    render_summary_dashboard(df_session_results, title="Rangkuman Sesi Analisis")

    # Buat seksi terpisah khusus untuk tombol unduh
    section_divider("Langkah 4: Unduh Laporan & Data", "📥")
    # Panggil download_controller HANYA jika ada hasil
    download_controller(st.session_state.analysis_results, context="detection")

else:
    # Jika tidak ada hasil analisis sama sekali, tampilkan placeholder
    section_divider("Langkah 4: Unduh Hasil Analisis", "📥")
    # Tampilkan tombol nonaktif sesuai keinginan Anda
    st.button("🛠️ Buat File & Unduh", disabled=True, use_container_width=True)