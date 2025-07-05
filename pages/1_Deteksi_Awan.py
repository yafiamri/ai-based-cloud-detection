# pages/1_Deteksi_Awan.py
import streamlit as st
import os, io, cv2, shutil
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

from utils.layout import apply_global_styles, render_page_header, render_sidebar_footer, section_divider, render_result
from utils.segmentation import load_segmentation_model, prepare_input_tensor, predict_segmentation, detect_circle_roi, canvas_to_mask
from utils.classification import load_classification_model, predict_classification
from utils.image import load_demo_images, load_uploaded_images, create_image_overlay, extract_media_from_zip
from utils.video import get_video_info, analyze_video
from utils.download import download_controller

# --- 1. KONFIGURASI HALAMAN & PEMUATAN MODEL ---
st.set_page_config(page_title="Deteksi Awan", layout="wide")
apply_global_styles()

# State ini akan kita ubah untuk memaksa widget dirender ulang dari nol.
if "widget_seed" not in st.session_state:
    st.session_state.widget_seed = 0
if "report_data" not in st.session_state:
    st.session_state.report_data = []

with st.sidebar:
    if st.button("🔄️ Bersihkan Halaman", use_container_width=True):
        st.session_state.report_data = []
        st.session_state.widget_seed += 1
        st.rerun()
render_sidebar_footer()

@st.cache_resource
def get_models():
    """Memuat model AI."""
    return load_segmentation_model(), load_classification_model()
seg_model, cls_model = get_models()

# --- 2. UI UTAMA ---
render_page_header("Deteksi Tutupan dan Jenis Awan")
st.write("Pilih gambar atau video citra langit, sesuaikan area analisis, lalu biarkan sistem menghitung **tutupan awan** dan mengenali **jenis awan** secara otomatis.")

# --- LANGKAH 1: UNGGAH CITRA (LOGIKA STATELESS) ---
section_divider("Langkah 1: Unggah Citra Langit", "📤")

seed = st.session_state.widget_seed
uploader_key = f"file_uploader_{seed}"
multiselect_key = f"multiselect_{seed}"

demo_images = load_demo_images()
demo_names = [d[0] for d in demo_images]
selected_demos = st.multiselect(
    "Pilih gambar demo untuk uji coba analisis:",
    demo_names,
    default=[],
    key=multiselect_key
)
uploaded_files_list = st.file_uploader(
    "Atau unggah gambar, video, dan berkas arsip:",
    type=["jpeg", "jpg", "png", "webp", "avi", "mov", "mp4", "mpeg4", "zip"],
    accept_multiple_files=True,
    key=uploader_key
)

# Membuat daftar berkas yang akan diproses dari awal setiap kali skrip berjalan
final_files_to_process = []

# Proses demo images
for fname, img in demo_images:
    if fname in selected_demos:
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.name = fname
        final_files_to_process.append(buf)

# Proses berkas yang diunggah
if uploaded_files_list:
    for file in uploaded_files_list:
        if file.name.lower().endswith(".zip"):
            with st.spinner(f"Mengekstrak berkas dari {file.name}"):
                extracted_files = extract_media_from_zip(file)
                final_files_to_process.extend(extracted_files)
        else:
            final_files_to_process.append(file)

# --- TAMPILKAN PRATINJAU ---
if final_files_to_process:
    image_files_only = [f for f in final_files_to_process if not f.name.lower().endswith(('.avi', '.mov', '.mp4', '.mpeg4'))]
    processed_images = {name: img for name, img in load_uploaded_images(image_files_only)}
    
    st.subheader("🖼️ Pratinjau Antrian Berkas untuk Diproses")
    num_images = len(image_files_only)
    num_videos = len(final_files_to_process) - num_images

    parts = []
    if num_images > 0: parts.append(f"{num_images} gambar")
    if num_videos > 0: parts.append(f"{num_videos} video")
    count_str = " dan ".join(parts)

    st.success(f"**{len(final_files_to_process)} berkas akan diproses**: : {count_str}")

    cols = st.columns(4)
    for i, file in enumerate(final_files_to_process):
        with cols[i % 4]:
            file.seek(0)
            ext = os.path.splitext(file.name)[1].lower()
            if ext in ['.avi', '.mov', '.mp4', '.mpeg4']:
                st.video(file)
            elif file.name in processed_images:
                st.image(processed_images[file.name], use_container_width=True)
            st.markdown(f"<p style='text-align: center;'>{file.name}</p>", unsafe_allow_html=True)

# --- LANGKAH 2: KONFIGURASI ANALISIS ---
section_divider("Langkah 2: Konfigurasi Analisis", "⚙️")
configs = {}
if not final_files_to_process:
    st.info("Pilih gambar demo atau unggah berkas untuk memulai konfigurasi.")
else:
    st.write("Pilih cara menentukan *Region of Interest* (ROI) area pengamatan dan interval antar-*frame* untuk video.")
    config_mode = st.radio("Pilih mode:", ["Otomatis untuk Semua Berkas", "Manual untuk Setiap Berkas"], horizontal=True)

    if config_mode == "Manual untuk Setiap Berkas":
        for file in final_files_to_process:
            name, ext = file.name, os.path.splitext(file.name)[1].lower()
            with st.expander(f"Atur untuk: **{name}**"):
                roi_method = st.selectbox("Metode ROI", ["Otomatis", "Manual (Kotak)", "Manual (Poligon)", "Manual (Lingkaran dari Garis)"], key=f"roi_method_{name}")
                configs[name] = {'roi_method': roi_method, 'canvas': None, 'interval': 5}
                
                is_video = ext in ['.avi', '.mov', '.mp4', '.mpeg4']
                if is_video:
                    file.seek(0); duration, _ = get_video_info(file)
                    if duration > 0:
                        configs[name]['interval'] = st.slider(f"Interval antar-*frame* (detik)", 1, max(1, int(duration)), min(5, max(1, int(duration))), 1, key=f"interval_{name}")
                
                if "Manual" in roi_method:
                    st.info("Gambar bentuk dengan *drag and drop* pada kanvas di bawah ini untuk menandai area langit.")
                    drawing_mode = "rect" if "Kotak" in roi_method else "polygon" if "Poligon" in roi_method else "line"
                    background_image = None
                    file.seek(0)
                    if is_video: _, background_image = get_video_info(file)
                    elif name in processed_images: background_image = processed_images.get(name)
                    
                    if background_image:
                        w, h = background_image.size; new_w, new_h = (640, int(640 * h / w))
                        canvas = st_canvas(fill_color="rgba(255, 0, 0, 0.3)", stroke_width=2, background_image=background_image.resize((new_w, new_h)), height=new_h, width=new_w, drawing_mode=drawing_mode, key=f"canvas_{name}")
                        configs[name]['canvas'] = canvas
    else:
        st.info("Mode otomatis akan mendeteksi ROI dari area langit secara otomatis dari citra yang diberikan dan interval antar-*frame* diatur ke 5 detik untuk berkas video.")
        for file in final_files_to_process:
            ext = os.path.splitext(file.name)[1].lower()
            configs[file.name] = {'roi_method': 'Otomatis', 'canvas': None, 'interval': 5 if ext in ['.avi', '.mov', '.mp4', '.mpeg4'] else None}

# --- LANGKAH 3: JALANKAN ANALISIS ---
section_divider("Langkah 3: Jalankan Analisis", "🚀")
if st.button("Proses Semua Berkas", type="primary", use_container_width=True, disabled=not final_files_to_process):
    start_time = datetime.now()
    total_steps = 0
    for file in final_files_to_process:
        ext = os.path.splitext(file.name)[1].lower()
        if ext in ['.avi', '.mov', '.mp4', '.mpeg4']:
            file.seek(0); duration, _ = get_video_info(file)
            interval = configs.get(file.name, {}).get('interval', 5)
            total_steps += len(range(0, int(duration), interval)) if duration > 0 and interval > 0 else 0
        else:
            total_steps += 1
    
    with st.spinner("⏳ Menganalisis semua citra. Harap tunggu..."):
        progress_bar = st.progress(0, text="Memulai analisis")
        step_counter = 0
        final_results = []
        image_files_only = [f for f in final_files_to_process if not f.name.lower().endswith(('.avi', '.mov', '.mp4', '.mpeg4'))]
        processed_images_for_analysis = {name: img for name, img in load_uploaded_images(image_files_only)}

        for file in final_files_to_process:
            file.seek(0)
            name, config, ext = file.name, configs.get(file.name), os.path.splitext(file.name)[1].lower()
            if not config: continue
            
            mask_user = None
            if config['roi_method'] != 'Otomatis' and (canvas := config.get('canvas')):
                if canvas.json_data and canvas.json_data.get("objects"):
                    original_h, original_w = 0, 0
                    file.seek(0)
                    if ext in ['.avi', '.mov', '.mp4', '.mpeg4']:
                        _, frame = get_video_info(file)
                        if frame: original_h, original_w = frame.height, frame.width
                    elif name in processed_images_for_analysis:
                        original_h, original_w = processed_images_for_analysis[name].height, processed_images_for_analysis[name].width
                    if original_h > 0: mask_user = canvas_to_mask(canvas, original_h, original_w)
            
            if ext in ['.avi', '.mov', '.mp4', '.mpeg4']:
                analysis_package = analyze_video(file, mask_user, config['interval'], seg_model, cls_model, progress_bar, step_counter, total_steps)
                if analysis_package:
                    final_results.append({"name": name, **analysis_package["result"]})
                    step_counter = analysis_package["step_counter"]
            elif name in processed_images_for_analysis:
                step_counter += 1
                progress_bar.progress(step_counter / total_steps, text=f"Menganalisis berkas: {name}")
                img = processed_images_for_analysis[name]
                np_img = np.array(img) / 255.0
                mask_roi = detect_circle_roi(np_img) if mask_user is None else mask_user
                tensor = prepare_input_tensor(np_img)
                pred = predict_segmentation(seg_model, tensor)
                pred = cv2.resize(pred.astype(np.uint8), (img.width, img.height), interpolation=cv2.INTER_NEAREST)
                final_mask = pred * mask_roi
                awan, total = final_mask.sum(), mask_roi.sum()
                coverage = (100 * awan / total) if total > 0 else 0
                oktaf = int(round((coverage / 100) * 8))
                kondisi = ["Cerah", "Sebagian Cerah", "Sebagian Berawan", "Berawan", "Hampir Tertutup", "Tertutup"][min(oktaf // 2, 5)]
                
                os.makedirs("temps/history/images", exist_ok=True); os.makedirs("temps/history/masks", exist_ok=True); os.makedirs("temps/history/overlays", exist_ok=True)
                timestamp, base_name = datetime.now().strftime("%Y%m%d_%H%M%S"), os.path.splitext(name)[0].replace(" ", "_")
                img_path = f"temps/history/images/{timestamp}_{base_name}.png"
                mask_path = f"temps/history/masks/{timestamp}_{base_name}.png"
                overlay_path = f"temps/history/overlays/{timestamp}_{base_name}.png"
                img.save(img_path)
                Image.fromarray(final_mask * 255).save(mask_path)
                create_image_overlay(img, final_mask).save(overlay_path)
                
                preds = predict_classification(cls_model, img_path)
                jenis = preds[0][0] if preds else "Tidak Terdeteksi"
                final_results.append({"name": name, "original_path": img_path, "mask_path": mask_path, "overlay_path": overlay_path, "coverage": coverage, "oktaf": oktaf, "kondisi_langit": kondisi, "jenis_awan": jenis, "top_preds": preds})
    
    progress_bar.empty()
    st.session_state["report_data"] = final_results
    st.session_state['analysis_duration'] = (datetime.now() - start_time).total_seconds()
    
    if final_results:
        riwayat_path = "temps/history/riwayat.csv"
        try: existing_df = pd.read_csv(riwayat_path) if os.path.exists(riwayat_path) and os.path.getsize(riwayat_path) > 0 else pd.DataFrame()
        except pd.errors.EmptyDataError: existing_df = pd.DataFrame()
        new_entries = []
        for r in final_results:
            top_preds_str = "; ".join([f"{label} ({round(prob*100, 1)}%)" for label, prob in r.get("top_preds", [])]) if r.get("top_preds") else ""
            new_entries.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"nama_file": r["name"],"original_path": r["original_path"],"mask_path": r["mask_path"],"overlay_path": r["overlay_path"],"coverage": round(r["coverage"], 2),"oktaf": r["oktaf"],"kondisi_langit": r["kondisi_langit"],"jenis_awan": r["jenis_awan"],"top_preds": top_preds_str})
        if new_entries:
            df_new = pd.DataFrame(new_entries)
            df_combined = pd.concat([existing_df, df_new], ignore_index=True)
            df_combined.to_csv(riwayat_path, index=False)
    st.rerun()

# --- LANGKAH 4: HASIL ANALISIS & UNDUH ---
report_data = st.session_state.get("report_data", [])
if report_data:
    if 'analysis_duration' in st.session_state:
        duration = st.session_state.pop('analysis_duration')
        st.success(f"✅ Analisis selesai dalam {duration:.2f} detik!")
    else:
        st.success("✅ Analisis selesai!")
    for r in report_data:
        render_result(r)
        
section_divider("Langkah 4: Unduh Hasil Analisis", "📥")
if not report_data:
    st.info("Hasil analisis dapat diunduh di sini setelah proses analisis selesai dilakukan.")
else:
    download_controller(report_data, context="deteksi")