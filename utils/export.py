# utils/export.py
import os
import pandas as pd
import zipfile
from io import BytesIO
from fpdf import FPDF
from datetime import datetime
from PIL import Image
import subprocess
import shutil
import tempfile
import cv2

def _find_executable(name: str) -> str:
    """Mencari path executable (ffmpeg.exe)."""
    if path := shutil.which(name): return path
    local_exe_path = os.path.join(os.getcwd(), f"{name}.exe")
    if os.path.exists(local_exe_path): return local_exe_path
    return name

FFMPEG_PATH = _find_executable("ffmpeg")

def _get_image_for_pdf(path: str) -> Image.Image:
    """Mengambil gambar pratinjau (frame pertama jika video) untuk disisipkan ke PDF."""
    if not path or pd.isna(path) or not os.path.exists(path): return None
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mpeg4']:
        try:
            command = [ FFMPEG_PATH, "-i", path, "-vframes", "1", "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-" ]
            proc = subprocess.run(command, capture_output=True)
            if proc.returncode == 0 and proc.stdout:
                cap = cv2.VideoCapture(path); width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
                if width > 0 and height > 0: return Image.frombytes('RGB', (width, height), proc.stdout)
            return None
        except Exception: return None
    else:
        try: return Image.open(path)
        except Exception: return None

def safe_text(text):
    """Fungsi untuk memastikan teks aman untuk FPDF."""
    return str(text).encode("latin-1", "replace").decode("latin-1")

def export_csv(data):
    """Ekspor data ke CSV, mengembalikan byte string (ini sudah benar untuk unduhan CSV)."""
    df = pd.DataFrame(data)
    report_df = df.rename(columns={"nama_file": "Nama File", "top_preds": "Detail Prediksi"})
    relevant_cols = ["timestamp", "Nama File", "coverage", "oktaf", "jenis_awan", "kondisi_langit", "Detail Prediksi"]
    final_cols = [col for col in relevant_cols if col in report_df.columns]
    return report_df[final_cols].to_csv(index=False).encode("utf-8")

def export_pdf(data, nama_pengguna="Pengguna", output_path=None):
    """
    Buat PDF laporan, SIMPAN KE DISK, dan KEMBALIKAN PATH FILE.
    Struktur sama persis seperti skrip lama Anda.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)
    logo_path = "assets/logo_transparent.png"
    primary_color, secondary_color = (41, 128, 185), (100, 100, 100)
    
    # Halaman Sampul
    pdf.add_page()
    if os.path.exists(logo_path): pdf.image(logo_path, x=(pdf.w - 100)/2, y=40, w=100)
    pdf.set_y(160); pdf.set_font("helvetica", 'B', 22); pdf.set_text_color(*primary_color)
    pdf.cell(0, 12, "Laporan Hasil Deteksi Awan Berbasis AI", ln=1, align='C')
    pdf.ln(5); pdf.set_text_color(*secondary_color)
    for label, value in [("Disusun oleh: ", nama_pengguna), ("Dicetak pada: ", datetime.now().strftime('%Y-%m-%d %H:%M WIB'))]:
        pdf.set_font("helvetica", 'B', 14); label_w = pdf.get_string_width(label)
        pdf.set_x((pdf.w - pdf.get_string_width(label + value)) / 2); pdf.cell(label_w, 8, label, ln=0)
        pdf.set_font("helvetica", '', 14); pdf.cell(0, 8, value, ln=1)
    pdf.set_y(-25); pdf.set_font("helvetica", '', 8); pdf.set_text_color(*primary_color)
    pdf.cell(0, 5, "© AI-Based Cloud Detection", 0, 1, 'L')
    pdf.set_font("helvetica", 'I', 8); pdf.set_text_color(*secondary_color)
    pdf.cell(0, 5, "Yafi Amri - Meteorologi ITB", 0, 0, 'L')
    
    # Konten untuk setiap hasil analisis
    for item in data:
        pdf.add_page()
        file_name = item.get('nama_file', item.get('name', 'Unknown'))
        pdf.set_text_color(*primary_color); pdf.set_font("helvetica", 'B', 16)
        pdf.cell(40, 10, "Hasil Analisis:", ln=0); pdf.set_font("helvetica", '', 16)
        pdf.cell(0, 10, f" {safe_text(file_name)}", ln=1)
        pdf.set_text_color(*secondary_color); pdf.set_font("helvetica", 'B', 12)
        pdf.cell(40, 8, "Waktu Analisis:", ln=0); pdf.set_font("helvetica", '', 12)
        pdf.cell(0, 8, f" {item.get('timestamp', '')}", ln=1); pdf.ln(5)

        img_width, spacing, y_pos = 90, 10, 40
        original_preview, overlay_preview = _get_image_for_pdf(item.get("original_path")), _get_image_for_pdf(item.get("overlay_path"))

        for preview, x_pos, caption in [(original_preview, 10, "Citra Asli"), (overlay_preview, 10 + img_width + spacing, "Overlay Segmentasi")]:
            if preview:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    preview.save(tmp.name, "PNG")
                    pdf.image(tmp.name, x=x_pos, y=y_pos, w=img_width)
                os.remove(tmp.name)
            else:
                pdf.rect(x_pos, y_pos, img_width, img_width, 'D')
            pdf.set_xy(x_pos, y_pos + img_width); pdf.set_font("helvetica", 'I', 10)
            pdf.cell(img_width, 10, caption, align='C')

        pdf.set_y(y_pos + img_width + 15)
        pdf.set_font("helvetica", 'B', 16); pdf.set_text_color(*primary_color); pdf.cell(0, 10, "Hasil Segmentasi", ln=1)
        col_widths_seg = [95, 95]
        pdf.set_fill_color(*primary_color); pdf.set_text_color(255, 255, 255); pdf.set_font("helvetica", 'B', 12)
        pdf.cell(col_widths_seg[0], 10, "Parameter", 1, 0, 'C', fill=True); pdf.cell(col_widths_seg[1], 10, "Nilai", 1, 1, 'C', fill=True)
        pdf.set_text_color(*secondary_color); pdf.set_font("helvetica", '', 12)
        rows_seg = [("Tutupan Awan", f"{item.get('coverage', 0):.2f}%"), ("Nilai Oktaf", str(item.get('oktaf', '0'))), ("Kondisi Langit", safe_text(item.get('kondisi_langit', '')))]
        for row in rows_seg:
            pdf.cell(col_widths_seg[0], 10, row[0], 1, 0, 'C'); pdf.cell(col_widths_seg[1], 10, row[1], 1, 1, 'C')

        pdf.ln(5); pdf.set_font("helvetica", 'B', 16); pdf.set_text_color(*primary_color); pdf.cell(0, 10, "Hasil Klasifikasi", ln=1)
        col_widths_cls = [95, 95]
        pdf.set_fill_color(*primary_color); pdf.set_text_color(255, 255, 255); pdf.set_font("helvetica", 'B', 12)
        pdf.cell(col_widths_cls[0], 8, "Jenis Awan", 1, 0, 'C', fill=True); pdf.cell(col_widths_cls[1], 8, "Confidence", 1, 1, 'C', fill=True)
        pdf.set_text_color(*secondary_color)
        
        top_preds_str = item.get("top_preds", "")
        top_preds = []
        if isinstance(top_preds_str, str) and top_preds_str:
            try:
                preds = [p.strip().split(' (') for p in top_preds_str.split(';')]
                top_preds = [(label, float(score.replace('%)',''))/100) for label, score in preds if len(score) > 0]
            except: top_preds = []
        elif isinstance(item.get("top_preds"), list):
            top_preds = item.get("top_preds")
        
        for i, (label, score) in enumerate(top_preds):
            pdf.set_font("helvetica", 'B' if i == 0 else '', 12)
            pdf.cell(col_widths_cls[0], 10, safe_text(label), 1, 0, 'C'); pdf.cell(col_widths_cls[1], 10, f"{score*100:.1f}%", 1, 1, 'C')
        if not top_preds:
            pdf.cell(col_widths_cls[0], 10, "N/A", 1, 0, 'C'); pdf.cell(col_widths_cls[1], 10, "N/A", 1, 1, 'C')

        pdf.set_y(-25); pdf.set_font("helvetica", '', 8); pdf.set_text_color(*primary_color)
        pdf.cell(0, 5, "© AI-Based Cloud Detection", 0, 1, 'L')
        pdf.set_font("helvetica", 'I', 8); pdf.set_text_color(*secondary_color)
        pdf.cell(0, 5, "Yafi Amri - Meteorologi ITB", 0, 0, 'L'); pdf.cell(0, 5, f"Halaman {pdf.page_no()-1}", 0, 0, 'R')
    
    # PERBAIKAN: Menyimpan ke file dan mengembalikan PATH, bukan bytes
    if output_path is None:
        os.makedirs("temps", exist_ok=True)
        output_path = f"temps/report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    
    pdf.output(output_path)
    return output_path

def export_zip(data, output_path=None):
    """Kompresi file ke ZIP, SIMPAN KE DISK, dan KEMBALIKAN PATH FILE."""
    # PERBAIKAN: Menyimpan ke file dan mengembalikan PATH, bukan bytes
    if output_path is None:
        os.makedirs("temps", exist_ok=True)
        output_path = f"temps/export_{datetime.now().strftime('%Y%m%d%H%M%S')}.zip"
        
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in data:
            for key in ["original_path", "overlay_path", "mask_path"]:
                path = item.get(key)
                if path and pd.notna(path) and os.path.exists(path):
                    folder_name = key.split('_')[0]
                    zipf.write(path, arcname=f"{folder_name}/{os.path.basename(path)}")
    
    return output_path