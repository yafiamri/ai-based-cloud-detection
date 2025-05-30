# pages/history.py
import streamlit as st
import pandas as pd
import os
from components.download import download_controller
from utils.layout import apply_global_styles

apply_global_styles()

def main():
    st.image("assets/logo_transparent.png", width=100)
    st.title("📖 Riwayat Analisis Awan")
    
    # Baca data
    csv_path = "temps/history/riwayat.csv"
    if not os.path.exists(csv_path):
        st.error("Belum ada riwayat analisis")
        return
    
    df = pd.read_csv(csv_path)
    if df.empty:
        st.warning("Riwayat kosong")
        return
    
    # Filter
    with st.expander("🔍 Filter"):
        jenis_filter = st.multiselect("Jenis Awan:", df["jenis_awan"].unique())
        kondisi_filter = st.multiselect("Kondisi Langit:", df["kondisi_langit"].unique())
    
    filtered_df = df.copy()
    if jenis_filter:
        filtered_df = filtered_df[filtered_df["jenis_awan"].isin(jenis_filter)]
    if kondisi_filter:
        filtered_df = filtered_df[filtered_df["kondisi_langit"].isin(kondisi_filter)]
    
    # Tampilkan data
    st.dataframe(
        filtered_df[["nama_gambar", "coverage", "oktaf", "jenis_awan", "kondisi_langit"]],
        use_container_width=True
    )
    
    # Pilihan
    selected_indices = st.multiselect(
        "Pilih analisis untuk diunduh/dihapus:",
        filtered_df.index.tolist(),
        format_func=lambda x: filtered_df.loc[x, "nama_gambar"]
    )
    
    # Tombol Aksi
    col1, col2 = st.columns(2)
    with col1:
        if selected_indices:
            download_controller(
                filtered_df.loc[selected_indices].to_dict("records"),
                context="history"
            )
    
    with col2:
        if st.button("🗑️ Hapus yang Dipilih"):
            # Hapus file gambar
            for idx in selected_indices:
                row = df.loc[idx]
                try: os.remove(row["original_path"])
                except: pass
                try: os.remove(row["overlay_path"])
                except: pass
            
            # Update CSV
            new_df = df.drop(index=selected_indices)
            new_df.to_csv(csv_path, index=False)
            st.success("Data terpilih dihapus!")
            st.experimental_rerun()

if __name__ == "__main__":
    main()



# pages/detect.py
import streamlit as st
import os
import numpy as np
from PIL import Image
from datetime import datetime

from utils.segmentation import load_segmentation_model, prepare_input_tensor, predict_segmentation
from utils.classification import load_classification_model, predict_classification
from utils.layout import apply_global_styles
from components.download import download_controller

apply_global_styles()

st.image("assets/logo_transparent.png", width=100)
st.title("☁️ Deteksi Tutupan dan Jenis Awan")
st.write("Unggah gambar langit atau ZIP, lalu tentukan ROI dan jalankan analisis.")

@st.cache_resource
def get_models():
    return load_segmentation_model(), load_classification_model()

seg_model, cls_model = get_models()

def save_to_history(name, coverage, oktaf, jenis_awan, kondisi_langit, img, overlay_img):
    """Simpan hasil ke direktori history"""
    os.makedirs("temps/history/images", exist_ok=True)
    os.makedirs("temps/history/overlays", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join([c if c.isalnum() else "_" for c in name])
    
    original_path = f"temps/history/images/{timestamp}_{safe_name}.jpg"
    overlay_path = f"temps/history/overlays/{timestamp}_{safe_name}.jpg"
    
    img.save(original_path)
    overlay_img.save(overlay_path)
    
    return {
        "name": name,
        "timestamp": timestamp,
        "coverage": coverage,
        "oktaf": oktaf,
        "jenis_awan": jenis_awan,
        "kondisi_langit": kondisi_langit,
        "original_path": original_path,
        "overlay_path": overlay_path,
        "overlay_img": overlay_img
    }

def draw_roi_interface(img, key):
    st.markdown("#### Pilih ROI untuk: " + key)
    roi_type = st.radio(
        f"Metode ROI ({key})",
        ["Otomatis (Lingkaran)", "Manual (Kotak)", "Manual (Poligon)", "Manual (Lingkaran - Klik 2 Titik)"],
        key=key + "_roi"
    )
    if roi_type.startswith("Manual"):
        mode = "rect" if "Kotak" in roi_type else "polygon" if "Poligon" in roi_type else "line"
        fixed_width = 640
        w, h = img.size
        aspect_ratio = h / w
        new_height = int(fixed_width * aspect_ratio)
        img_resized = img.resize((fixed_width, new_height))
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            background_image=img_resized,
            update_streamlit=True,
            height=new_height,
            width=fixed_width,
            drawing_mode=mode,
            key="canvas_" + key
        )
        small_caption(f"Canvas: {fixed_width}×{new_height} (rasio asli dipertahankan)")
        return roi_type, canvas_result
    else:
        small_caption("Mask ROI akan dideteksi otomatis berdasarkan area terang berbentuk lingkaran.")
        return roi_type, None

demo_images = load_demo_images()
demo_options = ["(Unggah gambar sendiri)"] + [d[0] for d in demo_images]
selected_demo = st.sidebar.selectbox("🖼️ Pilih Gambar", demo_options)

if selected_demo != "(Unggah gambar sendiri)":
    uploaded_files = []
    for fname, img in demo_images:
        if fname == selected_demo:
            file_like = io.BytesIO()
            img.save(file_like, format='JPEG')
            file_like.name = fname
            uploaded_files = [file_like]
            break
else:
    uploaded_files = st.file_uploader("Unggah Gambar atau ZIP", type=["jpg", "jpeg", "png", "zip"], accept_multiple_files=True)

if uploaded_files:
    uploaded_names = [getattr(f, "name", str(i)) + str(getattr(f, "size", 0)) for i, f in enumerate(uploaded_files)]
    if st.session_state.get("last_uploaded") != uploaded_names:
        st.session_state["uploaded_images_changed"] = True
        st.session_state["last_uploaded"] = uploaded_names
    else:
        st.session_state["uploaded_images_changed"] = False

    images = load_uploaded_images(uploaded_files)
    st.markdown(f"### Pratinjau Gambar ({len(images)} ditemukan)")
    display_image_grid([img for _, img in images], [name for name, _ in images])

    section_divider()
    st.header("🎯 Pengaturan Region of Interest (ROI)")
    
    roi_mode = st.radio("Bagaimana kamu ingin menetapkan ROI?", ["Gunakan ROI Otomatis untuk Semua Gambar", "Tentukan ROI untuk Setiap Gambar"])
    roi_selections, canvas_results = {}, {}
    roi_options = ["Otomatis (Lingkaran)", "Manual (Kotak)", "Manual (Poligon)", "Manual (Lingkaran - Klik Garis Diameter)"]
    
    if roi_mode == "Gunakan ROI Otomatis untuk Semua Gambar":
        roi_global = "Otomatis (Lingkaran)"
        st.info(f"ROI otomatis akan digunakan untuk semua gambar: **{roi_global}**")
        for name, img in images:
            roi_selections[name] = roi_global
            canvas_results[name] = None  # tidak pakai canvas
    else:
        st.markdown("Pilih metode ROI untuk setiap gambar:")
        for name, img in images:
            with st.expander(f"🖼️ ROI: {name}", expanded=False):
                roi_type = st.selectbox(f"Metode ROI ({name})", roi_options, key=f"roi_{name}")
                roi_selections[name] = roi_type
                if roi_type.startswith("Manual"):
                    fixed_width = 640
                    w, h = img.size
                    aspect_ratio = h / w
                    new_height = int(fixed_width * aspect_ratio)
                    img_resized = img.resize((fixed_width, new_height))
    
                    canvas = st_canvas(
                        fill_color="rgba(255, 0, 0, 0.3)",
                        stroke_width=2,
                        background_image=img_resized,
                        update_streamlit=True,
                        height=new_height,
                        width=fixed_width,
                        drawing_mode="rect" if "Kotak" in roi_type else "polygon" if "Poligon" in roi_type else "line",
                        key="canvas_" + name
                    )
                    small_caption(f"Canvas: {fixed_width}×{new_height} (rasio asli dipertahankan)")
                    canvas_results[name] = canvas
                else:
                    small_caption("Mask ROI akan dideteksi otomatis berdasarkan area terang berbentuk lingkaran.")
                    canvas_results[name] = None

    section_divider()
    st.header("🧠 Analisis Gambar")

    if st.button("🚀 Jalankan Analisis pada Semua Gambar") or (
        "report_data" not in st.session_state and st.session_state.get("uploaded_images_changed")
    ):
        st.session_state.pop("report_data", None)
        with st.spinner("⏳ Analisis sedang diproses..."):
            summary_data, report_data = [], []
    
            for name, img in images:
                st.subheader(f"📊 Hasil: {name}")
            
                image_np = np.array(img) / 255.0
                roi_type = roi_selections[name]
                canvas = canvas_results[name]
            
                mask_roi = detect_circle_roi(image_np) if roi_type.startswith("Otomatis") else \
                            canvas_to_mask(canvas, img.height, img.width)
                mask_roi = cv2.resize(mask_roi.astype(np.uint8), (img.width, img.height), interpolation=cv2.INTER_NEAREST)
            
                input_tensor = prepare_input_tensor(image_np)
                mask_pred = predict_segmentation(seg_model, input_tensor)
                mask_pred_resized = cv2.resize(mask_pred.astype(np.uint8), (img.width, img.height), interpolation=cv2.INTER_NEAREST)
            
                awan = (mask_pred_resized * mask_roi).sum()
                total = mask_roi.sum()
                coverage = 100 * awan / total if total > 0 else 0
                oktaf = int(round((coverage / 100) * 8))
            
                # Klasifikasi kondisi langit
                if oktaf == 0:
                    kondisi_langit = "Cerah"
                elif oktaf <= 2:
                    kondisi_langit = "Sebagian Cerah"
                elif oktaf <= 4:
                    kondisi_langit = "Sebagian Berawan"
                elif oktaf <= 6:
                    kondisi_langit = "Berawan"
                elif oktaf == 7:
                    kondisi_langit = "Hampir Tertutup"
                else:
                    kondisi_langit = "Tertutup"
            
                temp_path = f"temps/temp_{name.replace('.', '_')}.jpg"
                os.makedirs("temps", exist_ok=True)
                img.save(temp_path)
                top_preds = predict_classification(cls_model, temp_path)
                jenis_awan = top_preds[0][0]
            
                overlay_np = np.where(
                    (mask_pred_resized * mask_roi)[..., None] == 1,
                    (1 - 0.4) * image_np + 0.4 * np.array([1.0, 0.0, 0.0]),
                    image_np
                )
                overlay_img = Image.fromarray((overlay_np * 255).astype(np.uint8))
            
                # BARIS ATAS — GAMBAR            
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.image(img, caption="Gambar Asli")
                with col_g2:
                    st.image(overlay_img, caption="Overlay Segmentasi Awan")
                
                # BARIS BAWAH — RINGKASAN TABEL           
                col_t1, col_t2 = st.columns([1, 1])
                
                # with col_t1:
                #     st.markdown("#### 🖼️ Hasil Segmentasi")
                #     st.markdown(f"""
                #     | **Metode Analisis**      | **Nilai**            |
                #     |--------------------------|----------------------|
                #     | Tutupan Awan (%)         | {coverage:.2f}%      |
                #     | Tutupan Awan (oktaf)     | {oktaf} oktaf        |
                #     | Kondisi Langit           | {kondisi_langit}     |
                #     """)
                
                # with col_t2:
                #     st.markdown("#### 🌥️ Jenis Awan Terdeteksi:")
                #     table_md = "| Jenis Awan | Confidence |\n|------------|------------|\n"
                #     for i, (label, score) in enumerate(top_preds):
                #         confidence = f"{score * 100:.1f}%"
                #         if i == 0:
                #             table_md += f"| **{label}** | **{confidence}** |\n"
                #         else:
                #             table_md += f"| {label} | {confidence} |\n"
                #     st.markdown(table_md)
    
                with st.container():
                    col_t1, col_t2 = st.columns([1, 1])
                    with col_t1:
                        st.markdown("#### 🖼️ Hasil Segmentasi")
                        st.markdown(f"""
                        <style>
                        table {{
                            width: 100%;
                        }}
                        </style>
                        <table>
                            <thead>
                                <tr><th>Metode Analisis</th><th>Nilai</th></tr>
                            </thead>
                            <tbody>
                                <tr><td>Tutupan Awan (%)</td><td>{coverage:.2f}%</td></tr>
                                <tr><td>Tutupan Awan (oktaf)</td><td>{oktaf} oktaf</td></tr>
                                <tr><td>Kondisi Langit</td><td>{kondisi_langit}</td></tr>
                            </tbody>
                        </table>
                        """, unsafe_allow_html=True)
                
                    with col_t2:
                        st.markdown("#### 🌥️ Jenis Awan Terdeteksi:")
                        table_md = f"""
                        <style>
                        table {{
                            width: 100%;
                        }}
                        </style>
                        <table>
                            <thead>
                                <tr><th>Jenis Awan</th><th>Confidence</th></tr>
                            </thead>
                            <tbody>
                        """
                        for i, (label, score) in enumerate(top_preds):
                            confidence = f"{score * 100:.1f}%"
                            if i == 0:
                                table_md += f"<tr><td><b>{label}</b> (Dominan)</td><td><b>{confidence}</b></td></tr>"
                            else:
                                table_md += f"<tr><td>{label}</td><td>{confidence}</td></tr>"
                        table_md += "</tbody></table>"
                        st.markdown(table_md, unsafe_allow_html=True)
            
                # Simpan hasil
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = f"temps/history/images/{timestamp}_{name}"
                overlay_path = f"temps/history/overlays/{timestamp}_{name}"
                img.save(img_path)
                overlay_img.save(overlay_path)
            
                save_to_history(
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    name,
                    coverage,
                    oktaf,
                    jenis_awan,
                    kondisi_langit,
                    img_path,
                    overlay_path
                )
            
                summary_data.append({
                    "Nama Gambar": name,
                    "Tutupan Awan (%)": round(coverage, 2),
                    "Oktaf": oktaf,
                    "Jenis Awan Dominan": jenis_awan,
                    "Confidence Dominan (%)": round(top_preds[0][1]*100, 1)
                })
            
                report_data.append({
                    "name": name,
                    "timestamp": timestamp,
                    "coverage": coverage,
                    "oktaf": oktaf,
                    "kondisi_langit": kondisi_langit,
                    "top_preds": top_preds,  # list of (label, confidence)
                    "overlay_img": overlay_img,  # PIL Image
                    "image_original": img      # PIL Image
                })
        st.session_state["report_data"] = report_data
        st.success("✅ Analisis selesai.")
        
if "report_data" in st.session_state:
    report_data = st.session_state["report_data"]
    st.markdown("<div id='unduh'></div>", unsafe_allow_html=True)
    st.divider()
    st.markdown("### 📅 Unduh Hasil Analisis")

    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        opsi_unduh = st.selectbox("Pilih format unduhan:", ["CSV", "PDF", "ZIP Gambar"], key="unduh_format")
        nama_pengguna = st.text_input("Nama untuk laporan PDF:", value="Yafi Amri")

    with col2:
        if opsi_unduh == "CSV":
            csv_data = export_csv(report_data)
            b64 = base64.b64encode(csv_data).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="hasil_analisis.csv">📅 Klik untuk mengunduh CSV</a>'
            st.markdown(href, unsafe_allow_html=True)

        elif opsi_unduh == "PDF":
            pdf_path = export_pdf(report_data, nama_pengguna)
            with open(pdf_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="hasil_analisis.pdf">📄 Klik untuk mengunduh PDF</a>'
                st.markdown(href, unsafe_allow_html=True)

        elif opsi_unduh == "ZIP Gambar":
            zip_path = export_zip(report_data)
            with open(zip_path, "rb") as z:
                b64 = base64.b64encode(z.read()).decode()
                href = f'<a href="data:application/zip;base64,{b64}" download="gambar_segmentasi.zip">📦 Klik untuk mengunduh ZIP</a>'
                st.markdown(href, unsafe_allow_html=True)

    # scroll otomatis ke bagian unduh
    st.markdown("""
    <script>
        const el = document.getElementById("unduh");
        if (el) {
            setTimeout(() => {
                el.scrollIntoView({behavior: 'smooth'});
            }, 300);
        }
    </script>
    """, unsafe_allow_html=True)



# utils/export.py
import os
import pandas as pd
import zipfile
from io import BytesIO
from fpdf import FPDF
from datetime import datetime

def safe_text(text):
    return text.encode("latin-1", "replace").decode("latin-1")
    
def export_csv(data):
    df_data = []
    for item in data:
        df_data.append({
            "timestamp": item.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            "nama_gambar": item["name"],
            "original_path": item["original_path"],
            "overlay_path": item["overlay_path"],
            "coverage": round(item["coverage"], 2),
            "oktaf": item["oktaf"],
            "jenis_awan": item.get("jenis_awan", ""),
            "kondisi_langit": item.get("kondisi_langit", "")
        })
    df = pd.DataFrame(df_data)
    return df.to_csv(index=False).encode("utf-8")

def export_pdf(data, nama_pengguna):
    """Buat PDF report dengan template konsisten"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    logo_path = "assets/logo_transparent.png"

    # Cover Page
    pdf.add_page()
    logo_width = 100
    x_logo = (pdf.w - logo_width) / 2
    pdf.image(logo_path, x=x_logo, y=30, w=logo_width)
    pdf.set_y(150)
    pdf.set_font("Arial", 'B', 18)
    pdf.cell(0, 10, safe_text("Laporan Hasil Deteksi Awan Berbasis AI"), ln=True, align="C")
    pdf.set_font("Arial", '', 12)
    pdf.ln(10)
    pdf.cell(0, 10, safe_text(f"Disusun Oleh: {nama_pengguna}"), ln=True, align="C")
    pdf.cell(0, 10, safe_text(f"Tanggal Cetak: {datetime.now().strftime('%Y-%m-%d %H:%M WIB')}"), ln=True, align="C")

    # Konten
    for item in data:
        pdf.add_page()
        
        # Header
        pdf.image(logo_path, x=10, y=10, w=10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, safe_text(f"Analisis: {item['name']}"), ln=True, align="C")
        pdf.set_draw_color(180, 180, 180)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(5)

        # Body
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, safe_text(f"Waktu    : {item.get('timestamp', '')}"), ln=True)
        pdf.cell(0, 8, safe_text(f"Tutupan  : {item['coverage']:.2f}% ({item['oktaf']} oktaf)"), ln=True)
        pdf.cell(0, 8, safe_text(f"Jenis    : {item.get('jenis_awan', '')}"), ln=True)
        pdf.cell(0, 8, safe_text(f"Kondisi  : {item.get('kondisi_langit', '')}"), ln=True)
        pdf.ln(5)

        # Gambar
        try:
            if os.path.exists(item["overlay_path"]):
                pdf.image(item["overlay_path"], w=180)
            else:
                raise FileNotFoundError
        except:
            try:
                buf = BytesIO()
                item["overlay_img"].save(buf, format='JPEG')
                pdf.image(buf, w=180)
            except:
                pdf.cell(0, 10, safe_text("[Gagal menampilkan gambar overlay]"), ln=True)

        # Footer
        pdf.set_y(-30)
        pdf.set_font("Arial", 'I', 8)
        pdf.set_text_color(100)
        pdf.cell(100, 10, safe_text("© (2025) Yafi Amri - Meteorologi ITB"), align="L")
        pdf.cell(0, 10, safe_text(f"Halaman {pdf.page_no()}"), align="R")

    # Simpan ke temporary file
    os.makedirs("temps", exist_ok=True)
    output_path = f"temps/report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    pdf.output(output_path)
    return output_path

def export_zip(data):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        for item in data:
            # Original images
            src = item["original_path"]
            if os.path.exists(src):
                zipf.write(src, arcname=f"original/{os.path.basename(src)}")
            
            # Overlay images
            src = item["overlay_path"]
            if os.path.exists(src):
                zipf.write(src, arcname=f"overlay/{os.path.basename(src)}")
    
    # Simpan ke file
    os.makedirs("temps", exist_ok=True)
    output_path = f"temps/export_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
    with open(output_path, "wb") as f:
        zip_buffer.seek(0)
        f.write(zip_buffer.read())
    
    return output_path




# components/download.py
import streamlit as st
import base64
from utils.export import export_csv, export_pdf, export_zip

def download_controller(data, context="detect"):
    if not data:
        st.warning("Tidak ada data untuk diunduh")
        return
    
    st.divider()
    st.markdown(f"### 📥 Unduh Hasil ({context.capitalize()})")
    
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        format_options = ["PDF", "CSV", "ZIP Gambar"]
        selected_format = st.selectbox(
            "Format Unduhan:",
            format_options,
            key=f"format_{context}"
        )
        
        nama_pengguna = ""
        if selected_format == "PDF":
            nama_pengguna = st.text_input(
                "Nama untuk Laporan:",
                value="Yafi Amri",
                key=f"nama_{context}"
            )

    with col2:
        if st.button(f"📤 Generate {selected_format}", key=f"btn_{context}"):
            try:
                if selected_format == "CSV":
                    csv_data = export_csv(data)
                    b64 = base64.b64encode(csv_data).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="hasil_{context}.csv">💾 Klik untuk mengunduh CSV</a>'
                
                elif selected_format == "ZIP Gambar":
                    zip_path = export_zip(data)
                    with open(zip_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/zip;base64,{b64}" download="gambar_{context}.zip">📦 Klik untuk mengunduh ZIP</a>'
                
                elif selected_format == "PDF":
                    pdf_path = export_pdf(data, nama_pengguna)
                    with open(pdf_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="laporan_{context}.pdf">📄 Klik untuk mengunduh PDF</a>'
                
                st.markdown(href, unsafe_allow_html=True)
                st.success("Unduhan siap! Klik link di atas.")
            
            except Exception as e:
                st.error(f"Gagal membuat unduhan: {str(e)}")
                st.exception(e)



# pages/history.py
import streamlit as st
import pandas as pd
import os
from utils.export import export_csv, export_pdf, export_zip
from utils.download import download_controller
from utils.layout import apply_global_styles

apply_global_styles()

st.image("assets/logo_transparent.png", width=100)
st.title("📖 Riwayat Analisis Awan")

csv_path = "temps/history/riwayat.csv"
if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
    st.error("Belum ada hasil analisis yang tersimpan.")
    st.stop()

df = pd.read_csv(csv_path)
if df.empty:
    st.warning("Riwayat analisis masih kosong.")
    st.stop()

with st.expander("🔍 Filter Data"):
    filters = {}
    for col in ["timestamp", "coverage", "oktaf", "jenis_awan", "kondisi_langit"]:
        options = sorted(df[col].unique())
        filters[col] = st.multiselect(f"{col.replace('_', ' ').title()}:", options)

    filtered_df = df.copy()
    for col, selected in filters.items():
        if selected:
            filtered_df = filtered_df[filtered_df[col].isin(selected)]

with st.expander("↕️ Sortir Data"):
    sort_col = st.selectbox("Urutkan Berdasarkan Kolom:", df.columns.tolist(), index=0)
    sort_asc = st.radio("Urutan:", ["Naik", "Turun"], horizontal=True) == "Naik"
    filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc)

per_page = st.selectbox("Jumlah data per halaman:", [5, 10, 20, 50], index=1)
total_rows = len(filtered_df)
total_pages = max(1, (total_rows + per_page - 1) // per_page)
page = st.number_input("Halaman ke:", min_value=1, max_value=total_pages, step=1)

start = (page - 1) * per_page
end = start + per_page
paginated_df = filtered_df.iloc[start:end]

top_col1, top_col2, top_col3 = st.columns([0.6, 0.2, 0.2])
with top_col2:
    per_page = st.selectbox("Jumlah per halaman:", [5, 10, 20, 50], key="perpage")
with top_col3:
    current_page = st.number_input("Halaman ke:", min_value=1, max_value=total_pages, step=1, key="curpage")

header_cols = st.columns([0.10, 0.15, 0.10, 0.10, 0.15, 0.10, 0.15, 0.15])
headers = ["Pilih Semua", "Waktu Analisis", "Original", "Overlay", "Coverage", "Oktaf", "Jenis Awan", "Kondisi Langit"]

with header_cols[0]:
    st.markdown("**Pilih Semua**")
    select_all = st.checkbox("", value=len(selected) == len(paginated_df), key="select_all")

new_selected = set()

for col, title in zip(header_cols[1:], headers[1:]):
    col.markdown(f"<div style='text-align: center; font-weight: bold'>{title}</div>", unsafe_allow_html=True)

for i, row in paginated_df.iterrows():
    checked = select_all or i in selected
    cols = st.columns([0.10, 0.15, 0.10, 0.10, 0.15, 0.10, 0.15, 0.15])
    with cols[0]:
        if st.checkbox("", value=checked, key=f"check_{i}"):
            new_selected.add(i)
    with cols[1]:
        st.markdown(f"<div style='text-align: center'>{row['timestamp']}</div>", unsafe_allow_html=True)
    with cols[2]:
        try:
            st.image(row["original_path"], width=100)
        except:
            st.warning("Gambar hilang")
    with cols[3]:
        try:
            st.image(row["overlay_path"], width=100)
        except:
            st.warning("Overlay hilang")
    with cols[4]:
        st.markdown(f"<div style='text-align: center'><b>{row['coverage']}%</b></div>", unsafe_allow_html=True)
    with cols[5]:
        st.markdown(f"<div style='text-align: center'>{row['oktaf']} oktaf</div>", unsafe_allow_html=True)
    with cols[6]:
        st.markdown(f"<div style='text-align: center'>{row['jenis_awan']}</div>", unsafe_allow_html=True)
    with cols[7]:
        st.markdown(f"<div style='text-align: center'>{row['kondisi_langit']}</div>", unsafe_allow_html=True)

st.session_state["selected_rows"] = new_selected

if new_selected:
    subset = df.loc[list(new_selected)].copy()
    report_data = [{
        "timestamp": row["timestamp"],
        "name": row["nama_gambar"],
        "coverage": row["coverage"],
        "oktaf": row["oktaf"],
        "jenis_awan": row["jenis_awan"],
        "kondisi_langit": row["kondisi_langit"],
        "original_path": row["original_path"],
        "overlay_path": row["overlay_path"],
        "top_preds": []
    } for _, row in subset.iterrows()]

    col3, col_spacer, col5 = st.columns([0.25, 0.5, 0.25])
    with col3:
        download_controller(report_data, context="history")

    with col5:
        if st.button("🗑️ Hapus Data yang Dipilih", type="primary"):
            for _, row in subset.iterrows():
                for path in [row["original_path"], row["overlay_path"]]:
                    try:
                        os.remove(path)
                    except:
                        pass
            df = df.drop(index=list(new_selected)).reset_index(drop=True)
            df.to_csv(csv_path, index=False)
            st.success("Data terpilih berhasil dihapus.")
            st.experimental_rerun()