# Lokasi: pages/1_Deteksi_Awan.py
import streamlit as st
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from streamlit_drawable_canvas import st_canvas
import io, zipfile, time, logging, base64, cv2, numpy as np

# Impor semua "perkakas" dari arsitektur baru kita
from core.processing import analyze_image, analyze_video
from core.models import get_segmentation_model, get_classification_model
from core.io import file_manager
from ui_components import (
    render_page_header, render_sidebar_footer, render_analysis_card,
    render_download_controller, apply_global_styles, section_divider, load_demo_images
)

log = logging.getLogger(__name__)

# --- 1. KONFIGURASI HALAMAN & STATE ---
st.set_page_config(page_title="Deteksi Awan", page_icon="☁️", layout="wide")

def initialize_state():
    """Inisialisasi session state di awal untuk mencegah error."""
    defaults = {
        "analysis_queue": {}, "analysis_results": [],
        "last_uploaded_keys": [], "demo_selection": []
    }
    for key, value in defaults.items():
        if key not in st.session_state: st.session_state[key] = value
initialize_state()
apply_global_styles()

# --- 2. PEMUATAN MODEL & FUNGSI LOGIKA ---
@st.cache_resource
def load_models_once():
    """Memuat model AI sekali dan menyimpannya di cache Streamlit."""
    with st.spinner("⏳ Memuat model AI untuk sesi ini..."):
        seg_model, seg_device = get_segmentation_model()
        cls_model = get_classification_model()
        if seg_model is None or cls_model is None:
            st.error("Gagal memuat model AI. Periksa konsol untuk detail error.")
            st.stop()
    return seg_model, seg_device, cls_model

seg_model, seg_device, cls_model = load_models_once()

def add_to_queue(name, content_bytes):
    """Menambahkan file unik ke antrian di session state."""
    if name in st.session_state.analysis_queue: return
    try:
        file_type = "video" if name.lower().endswith(".mp4") else "image"
        preview_img = None
        if file_type == "image":
            preview_img = Image.open(io.BytesIO(content_bytes)).convert("RGB")
        else: # Untuk video, buat preview dari frame pertama
            temp_path = Path("temps") / f"preview_{name}"
            temp_path.parent.mkdir(exist_ok=True, parents=True)
            temp_path.write_bytes(content_bytes)
            cap = cv2.VideoCapture(str(temp_path))
            if cap.isOpened():
                ret, frame = cap.read()
                if ret: preview_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                cap.release()
            if temp_path.exists(): temp_path.unlink()

        st.session_state.analysis_queue[name] = {
            "content": content_bytes, "type": file_type, "preview": preview_img,
            "settings": {"roi_mode": "Otomatis", "canvas_data": None, "frame_interval_seconds": 5}
        }
    except Exception as e:
        st.warning(f"Gagal memuat pratinjau untuk '{name}': {e}")
        log.error(f"Error di add_to_queue untuk {name}", exc_info=True)

def run_batch_analysis(global_params):
    """Menjalankan analisis untuk semua file di antrian."""
    st.session_state.analysis_results = []
    results = []
    progress_bar = st.progress(0, "Memulai analisis batch...")
    queue_items = list(st.session_state.analysis_queue.items())

    for i, (name, item) in enumerate(queue_items):
        progress_bar.progress((i) / len(queue_items), f"Menganalisis: {name}")
        
        try:
            result = None
            if item.get('type') == 'image':
                # Siapkan parameter spesifik untuk gambar
                image_params = {**global_params, **item["settings"]}
                image_params.pop('frame_interval_seconds', None)
                
                image = Image.open(io.BytesIO(item["content"])).convert("RGB")
                result = analyze_image(image=image, file_name=name, **image_params)

            elif item.get('type') == 'video':
                video_params = {**global_params, **item["settings"]}
                temp_vid_path = Path("temps") / name
                temp_vid_path.parent.mkdir(exist_ok=True, parents=True)
                temp_vid_path.write_bytes(item["content"])
                result = analyze_video(video_path=temp_vid_path, file_name=name, **video_params)
                if temp_vid_path.exists(): temp_vid_path.unlink()

            if result:
                existing = file_manager.check_if_hash_exists(result.get("file_hash", ""))
                
                final_result_for_display = None
                if not existing:
                    # Ini hasil baru, simpan ke riwayat
                    st.toast(f"Analisis baru untuk '{name}' selesai.", icon="✨")
                    if item.get('type') == 'image': 
                        result.update(file_manager.save_analysis_artifacts(result))
                    file_manager.add_record_to_history(result)
                    final_result_for_display = result # Hasil ini sudah berisi objek gambar PIL
                else:
                    # Ini hasil lama dari riwayat.
                    st.toast(f"Hasil identik untuk '{name}' ditemukan di riwayat.", icon="ℹ️")
                    
                    # "Perkaya" data riwayat dengan memuat ulang gambar dari path-nya
                    final_result_for_display = existing
                    images_from_history = {}
                    try:
                        # Muat gambar original dan overlay dari path yang tersimpan di CSV
                        if final_result_for_display.get("original_path"):
                            images_from_history["original"] = Image.open(final_result_for_display["original_path"])
                        if final_result_for_display.get("overlay_path"):
                            images_from_history["overlay"] = Image.open(final_result_for_display["overlay_path"])
                    except Exception as e:
                        log.error(f"Gagal memuat ulang gambar dari riwayat untuk {name}: {e}")
                    
                    # Tambahkan sub-dictionary 'images' yang dibutuhkan oleh komponen render
                    final_result_for_display["images"] = images_from_history
                    
                results.append(final_result_for_display)
        except Exception as e:
            st.error(f"Gagal total menganalisis '{name}': {e}", icon="🚨")
            log.error(f"Error menganalisis {name}", exc_info=True)

    st.session_state.analysis_results = results
    progress_bar.progress(1.0, "Analisis batch selesai!")
    time.sleep(1); st.balloons(); progress_bar.empty()

# --- 3. TATA LETAK & ALUR KERJA UI ---
with st.sidebar:
    st.header("⚙️ Pengaturan Global")
    with st.expander("Ambang Batas (Threshold)", expanded=True):
        st.info("Parameter ini berlaku untuk semua file dalam sekali analisis.")
        seg_thresh = st.slider("Threshold Segmentasi", 0.0, 1.0, 0.5, 0.05)
        cls_thresh = st.slider("Threshold Klasifikasi", 0.0, 1.0, 0.5, 0.05)
    
    st.markdown("---")
    if st.button("🗑️ Kosongkan Antrian & Hasil", use_container_width=True):
        st.session_state.analysis_queue, st.session_state.analysis_results = {}, []
        st.session_state.demo_selection, st.session_state.last_uploaded_keys = [], []
        st.rerun()
    render_sidebar_footer()

render_page_header("Deteksi Tutupan dan Jenis Awan", "logo.png")

# LANGKAH 1: INPUT DATA
section_divider("Langkah 1: Pilih Input Data", "📥")
demo_images = load_demo_images()
if demo_images:
    selected_demos = st.multiselect("Pilih Gambar Demo:", [n for n, _ in demo_images], st.session_state.demo_selection)
    if selected_demos != st.session_state.demo_selection:
        st.session_state.demo_selection = selected_demos
        st.rerun()

# Proses penambahan/pengurangan demo dari antrian
queue_demo_names = {name for name, item in st.session_state.analysis_queue.items() if any(name == demo_name for demo_name, _ in demo_images)}
demos_to_add = set(st.session_state.demo_selection) - queue_demo_names
demos_to_remove = queue_demo_names - set(st.session_state.demo_selection)
for name, img in demo_images:
    if name in demos_to_add:
        buf = io.BytesIO(); img.save(buf, 'PNG'); add_to_queue(name, buf.getvalue())
for name in demos_to_remove:
    if name in st.session_state.analysis_queue: del st.session_state.analysis_queue[name]

uploaded_files = st.file_uploader("Atau Unggah File Anda (bisa lebih dari satu)", type=["jpg","png","mp4","zip"], accept_multiple_files=True)
if uploaded_files:
    current_keys = [f.name + str(f.size) for f in uploaded_files]
    if current_keys != st.session_state.last_uploaded_keys:
        st.session_state.last_uploaded_keys = current_keys
        for file in uploaded_files:
            if file.name.lower().endswith('.zip'):
                with zipfile.ZipFile(file, 'r') as zf:
                    for member in zf.infolist():
                        if not member.is_dir() and not member.filename.startswith('__MACOSX'):
                            add_to_queue(Path(member.filename).name, zf.read(member.filename))
            else:
                add_to_queue(file.name, file.getvalue())
        st.rerun()

# LANGKAH 2: PREVIEW & KONFIGURASI
section_divider("Langkah 2: Konfigurasi Analisis", "⚙️")
if not st.session_state.analysis_queue:
    st.info("Unggah file atau pilih gambar demo untuk memulai.")
else:
    st.success(f"**{len(st.session_state.analysis_queue)} file** dalam antrian. Atur di bawah ini atau langsung proses.")
    
    # Galeri Preview
    preview_items = [(name, item.get("preview")) for name, item in st.session_state.analysis_queue.items() if item.get("preview")]
    if preview_items:
        st.subheader("Galeri Preview Antrian")
        cols = st.columns(4)
        for i, (name, img) in enumerate(preview_items):
            cols[i % 4].image(img, caption=name, use_container_width=True)
        st.markdown("---")

    st.subheader("Konfigurasi per File (Opsional)")
    config_mode = st.radio("Mode Konfigurasi:", ["Otomatis untuk Semua", "Manual per File"], horizontal=True, index=0)

    for name, item in st.session_state.analysis_queue.items():
        if config_mode == "Manual per File":
            with st.expander(f"Atur untuk: {name}"):
                # [PERBAIKAN] Kembalikan semua opsi ROI manual
                roi_options = ["Otomatis", "Manual (Kotak)", "Manual (Poligon)", "Manual (Garis)"]
                item["settings"]["roi_mode"] = st.selectbox("Metode ROI:", roi_options, key=f"roi_{name}")
                
                if item.get('type') == 'video':
                    item["settings"]["frame_interval_seconds"] = st.slider("Interval Video (detik):", 1, 60, item['settings']['frame_interval_seconds'], key=f"interval_{name}")
                
                if "Manual" in item["settings"]["roi_mode"]:
                    img_for_canvas = item.get("preview") or Image.open(io.BytesIO(item["content"])).convert("RGB")
                    st.info("Gambar bentuk pada kanvas di bawah untuk menandai area langit.")
                    
                    w, h = img_for_canvas.size
                    new_w = 700
                    new_h = int(h * new_w / w)
                    resized_img = img_for_canvas.resize((new_w, new_h))
                
                    drawing_mode_map = {"Kotak": "rect", "Poligon": "polygon", "Garis": "line"}
                    shape_name = item["settings"]["roi_mode"].split(" (")[1][:-1]  # contoh: "Manual (Kotak)" -> "Kotak"
                    drawing_mode = drawing_mode_map.get(shape_name, "freedraw")
                
                    canvas_result = st_canvas(
                        fill_color="rgba(255,0,0,0.2)",
                        stroke_width=2,
                        background_image=resized_img,  # ✅ langsung pakai PIL.Image
                        height=new_h,
                        width=new_w,
                        drawing_mode=drawing_mode,
                        key=f"canvas_{name}"
                    )
                
                    if canvas_result.json_data:
                        item["settings"]["canvas_data"] = {
                            "json_data": canvas_result.json_data,
                            "canvas_size": (new_w, new_h),
                            "original_size": img_for_canvas.size
                        }
        else: # Mode Otomatis
            item["settings"]["roi_mode"] = "otomatis"
            item["settings"]["canvas_data"] = None
            if item.get('type') == 'video': item["settings"]["frame_interval_seconds"] = 5
                
# LANGKAH 3: JALANKAN ANALISIS
section_divider("Langkah 3: Jalankan Analisis", "🚀")
if st.button("Proses Semua File", use_container_width=True, type="primary", disabled=not st.session_state.analysis_queue):
    global_params = {"seg_model": seg_model, "seg_device": seg_device, "cls_model": cls_model, "seg_threshold": seg_thresh, "cls_threshold": cls_thresh}
    run_batch_analysis(global_params)

# LANGKAH 4: HASIL ANALISIS
if st.session_state.analysis_results:
    section_divider("Hasil Analisis", "📊")
    for res in st.session_state.analysis_results:
        original_path = res.get("original_path") or res.get("original_video_path")
        render_analysis_card(res, original_media_path=Path(original_path) if original_path else None)
    
    st.markdown("---")
    st.subheader("⬇️ Unduh Laporan")
    author = st.text_input("Masukkan Nama Anda untuk Laporan PDF:", "Pengguna Aplikasi")
    if author:
        render_download_controller(st.session_state.analysis_results, "deteksi", author)