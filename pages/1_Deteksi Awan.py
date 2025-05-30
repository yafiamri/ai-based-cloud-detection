# pages/detect.py
import streamlit as st
import os, io, cv2, base64
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
from streamlit_drawable_canvas import st_canvas

from utils.segmentation import load_segmentation_model, prepare_input_tensor, predict_segmentation
from utils.classification import load_classification_model, predict_classification
from utils.layout import apply_global_styles, render_page_header, render_sidebar_footer, section_divider, small_caption, display_image_grid, render_result, render_video_result
from utils.image import clear_temps, load_demo_images, load_uploaded_files
from utils.processing import analyze_image, analyze_video
from utils.roi import generate_roi_mask
from utils.download import download_controller

st.set_page_config(page_title="Deteksi Awan", layout="wide")
apply_global_styles()
render_page_header("☁️ Deteksi Tutupan dan Jenis Awan")

st.write("Pilih gambar citra langit, sesuaikan area pengamatan, lalu biarkan sistem menghitung **tutupan awan** dan mengenali **jenis awan** secara otomatis.")

clear_temps()

@st.cache_resource
def get_models():
    return load_segmentation_model(), load_classification_model()

seg_model, cls_model = get_models()

if "report_data" not in st.session_state:
    st.session_state["report_data"] = []

# Load demo dan upload
demo_images = load_demo_images()
demo_names = [d[0] for d in demo_images]
selected_demos = st.multiselect("Pilih file demo:", demo_names, default=[])

uploaded_files = st.file_uploader(
    "Atau unggah citra langit:",
    type=["jpg", "jpeg", "png", "zip", "mp4"],
    accept_multiple_files=True
)

# Gabungkan demo dan upload
demo_buffers = []
for fname, img in demo_images:
    if fname in selected_demos:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.name = fname
        demo_buffers.append(buf)

uploaded_files = demo_buffers + (uploaded_files if uploaded_files else [])

if uploaded_files:
    image_files = [f for f in uploaded_files if not f.name.endswith(".mp4")]
    video_files = [f for f in uploaded_files if f.name.endswith(".mp4")]

    results = []
    video_settings = {}
    video_previews = {}

    # Proses video preview
    for vid in video_files:
        bytes_io = io.BytesIO(vid.read())
        bytes_io.seek(0)
        temp_path = f"temps/{vid.name}"
        with open(temp_path, "wb") as f:
            f.write(bytes_io.read())

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            st.error(f"⚠️ Gagal memproses video {vid.name}")
            continue

        ret, frame = cap.read()
        cap.release()
        if not ret:
            st.error(f"⚠️ Frame video {vid.name} tidak terbaca")
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preview_pil = Image.fromarray(frame_rgb)
        video_previews[vid.name] = preview_pil

    # Load gambar
    images, _ = load_uploaded_files(image_files)

    preview_images = images + [(name, img) for name, img in video_previews.items()]
    display_image_grid([img for _, img in preview_images], [name for name, _ in preview_images])

    section_divider("Region of Interest (ROI)", emoji="🎯")
    roi_selections, canvas_results = {}, {}
    mode = st.radio("Pilih cara menentukan ROI:", ["Otomatis untuk Semua", "Manual untuk Setiap File"])

    if mode == "Otomatis untuk Semua":
        for name, _ in preview_images:
            roi_selections[name] = "Otomatis"
            canvas_results[name] = None
            if name.endswith(".mp4"):
                st.session_state.setdefault(f"interval_{name}", 1)
                video_settings[name] = {
                    "temp_path": f"temps/{name}",
                    "interval": 1,
                    "roi_mode": "auto",
                    "roi_mask": None
                }
    else:
        for name, img in preview_images:
            with st.expander(f"ROI untuk {name}"):
                roi_type = st.selectbox(
                    "Metode ROI", ["Otomatis", "Manual (Kotak)", "Manual (Poligon)", "Manual (Lingkaran)"],
                    key=f"roi_{name}"
                )
                roi_selections[name] = roi_type

                if name.endswith(".mp4"):
                    st.session_state.setdefault(f"interval_{name}", 1)
                    interval = st.selectbox(
                        "🎬 Ambil frame setiap ... detik",
                        [1, 2, 3, 5, 10, 15, 30],
                        index=0,
                        key=f"interval_{name}"
                    )

                if roi_type != "Otomatis":
                    w, h = img.size
                    ratio = h / w
                    new_w = 640
                    new_h = int(new_w * ratio)
                    resized = img.resize((new_w, new_h))

                    canvas = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.3)",
                        stroke_width=2,
                        background_image=resized,
                        update_streamlit=True,
                        height=new_h,
                        width=new_w,
                        drawing_mode="rect" if "Kotak" in roi_type else "polygon" if "Poligon" in roi_type else "line",
                        key=f"canvas_{name}"
                    )
                    small_caption(f"Canvas: {new_w}×{new_h}")
                    canvas_results[name] = canvas

                    if name.endswith(".mp4"):
                        video_settings[name] = {
                            "temp_path": f"temps/{name}",
                            "interval": interval,
                            "roi_mode": "manual",
                            "roi_mask": generate_roi_mask(
                                np.array(img) / 255.0,
                                mode="manual",
                                canvas_data=canvas,
                                target_size=(img.height, img.width)
                            )
                        }
                else:
                    canvas_results[name] = None
                    if name.endswith(".mp4"):
                        video_settings[name] = {
                            "temp_path": f"temps/{name}",
                            "interval": interval,
                            "roi_mode": "auto",
                            "roi_mask": None
                        }
    
    # Tombol Analisis
    section_divider("Jalankan Analisis", emoji="🧠")
    if st.button("🚀 Proses Semua Data"):
        with st.spinner("⏳ Sedang memproses..."):
            for name, img in images:
                roi_type = roi_selections.get(name, "Otomatis")
                canvas = canvas_results.get(name)
                result, warning = analyze_image(name, img, roi_type, canvas, seg_model, cls_model)
                if warning:
                    st.warning(warning)
                elif result:
                    results.append(result)

            for vid in video_files:
                setting = video_settings.get(vid.name, {})
                if not setting:
                    continue
                result_list, warning = analyze_video(
                    video_path=setting["temp_path"],
                    seg_model=seg_model,
                    cls_model=cls_model,
                    interval_detik=setting["interval"],
                    roi_mask=setting.get("roi_mask"),
                    roi_mode=setting.get("roi_mode", "auto")
                )
                if warning:
                    st.warning(warning)
                elif result_list:
                    results.extend(result_list)

            st.session_state["report_data"] = results

# Tampilkan Hasil
report_data = st.session_state.get("report_data", [])
if report_data:
    section_divider("Hasil Analisis", emoji="📊")
    for r in report_data:
        if "video_id" in r:
            render_video_result(r)
        else:
            render_result(r)
    st.success("✅ Analisis selesai.")
    st.markdown("### 📥 Unduh Hasil Analisis")
    download_controller(report_data, context="deteksi")

render_sidebar_footer()