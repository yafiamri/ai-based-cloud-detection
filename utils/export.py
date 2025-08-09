# utils/export.py
import os
import re
import pandas as pd
import zipfile
import tempfile
import shutil
import plotly.express as px
from datetime import datetime
from fpdf import FPDF
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple

# Impor konfigurasi terpusat
from .config import config
from .media import get_preview_as_pil

# Ambil seksi konfigurasi yang relevan untuk mempermudah akses
PATHS = config.get('paths', {})
APP_CONFIG = config.get('app', {})
PDF_CONFIG = config.get('pdf_report', {})
UI_CONFIG = config.get('ui', {})

# --- Fungsi Helper ---

def _safe_text(text: Any) -> str:
    """Memastikan teks aman untuk FPDF dengan mengganti karakter non-latin."""
    return str(text).encode("latin-1", "replace").decode("latin-1")

def _get_image_for_pdf(path: str, max_size: tuple = (512, 512)) -> Optional[Image.Image]:
    """
    Pembungkus (wrapper) yang mengambil gambar pratinjau untuk PDF
    dan memastikan ukurannya optimal serta menangani placeholder.
    """
    pil_img = get_preview_as_pil(path, max_size=max_size)

    if not pil_img:
        placeholder_path = PATHS.get('placeholder')
        if placeholder_path and os.path.exists(placeholder_path):
            return Image.open(placeholder_path)
        return None
    
    return pil_img

def _create_dashboard_charts(data: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    Membuat gambar-gambar grafik (histogram dan pie) dan mengembalikan path-nya.
    Teks metrik sekarang akan dibuat langsung di PDF.
    """
    if data.empty:
        return None

    temp_dir = tempfile.mkdtemp()
    
    theme = UI_CONFIG.get('theme', {})
    primary_color_hex = theme.get('primary_color', '#1f77b4')
    
    # 1. Grafik Histogram
    try:
        fig1 = px.histogram(data, x="cloud_coverage", nbins=10, 
                            title="Distribusi Tutupan Awan", 
                            color_discrete_sequence=[primary_color_hex])
        fig1.update_layout(template="plotly_white", yaxis_title="Jumlah Analisis", 
                           xaxis_title="Tutupan Awan (%)", title_x=0.5, margin=dict(t=40, b=40))
        path1 = os.path.join(temp_dir, "hist.png")
        fig1.write_image(path1, width=700, height=400, scale=2)
    except Exception:
        path1 = None

    # 2. Grafik Pie
    try:
        fig2 = px.pie(data, names="dominant_cloud_type", title="Komposisi Jenis Awan", hole=0.4)
        fig2.update_layout(template="plotly_white", legend_title_text='Jenis Awan', 
                           title_x=0.5, margin=dict(t=40, b=40))
        path2 = os.path.join(temp_dir, "pie.png")
        fig2.write_image(path2, width=700, height=450, scale=2)
    except Exception:
        path2 = None
    
    return path1, path2

# --- Fungsi Ekspor Utama ---

def export_csv(data: List[Dict[str, Any]]) -> str:
    """
    Mengekspor data ke file CSV di direktori sementara dan mengembalikan path-nya.

    Args:
        data (List[Dict[str, Any]]): Daftar dictionary hasil analisis.

    Returns:
        str: Path ke file CSV yang telah dibuat.
    """
    report_dir = PATHS.get('report_dir', 'temp/reports')
    os.makedirs(report_dir, exist_ok=True)
    
    df = pd.DataFrame(data)
    relevant_cols = [
        "analyzed_at", "source_filename", "media_type", "cloud_coverage",
        "okta_value", "sky_condition", "dominant_cloud_type", "classification_details"
    ]
    header_map = {
        "analyzed_at": "Waktu Analisis", "source_filename": "Nama File",
        "media_type": "Tipe Media", "cloud_coverage": "Tutupan Awan (%)",
        "okta_value": "Nilai Okta", "sky_condition": "Kondisi Langit",
        "dominant_cloud_type": "Jenis Awan Dominan", "classification_details": "Detail Klasifikasi"
    }
    
    export_df = df[[col for col in relevant_cols if col in df.columns]].copy()
    export_df.rename(columns=header_map, inplace=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + "_UTC"
    output_path = os.path.join(report_dir, f"metadata_{timestamp}.csv")
    export_df.to_csv(output_path, index=False, encoding='utf-8')
    return output_path

def export_zip(data: List[Dict[str, Any]]) -> str:
    """
    Mengompres semua file artefak ke dalam satu file ZIP dengan struktur
    yang sama persis seperti di folder sistem (data/archive).
    """
    report_dir = PATHS.get('report_dir', 'temp/reports')
    os.makedirs(report_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + "_UTC"
    zip_path = os.path.join(report_dir, f"archive_{timestamp}.zip")

    # Tentukan path dasar arsip dan temp untuk membuat path relatif
    archive_base_path = PATHS.get('archive_dir', 'data/archive')
    temp_base_path = os.path.join(PATHS.get('temp_dir', 'temp'), "live_session_artefacts")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in data:
            # Loop melalui setiap jenis artefak
            for key in ["original_path", "overlay_path", "mask_path"]:
                path = item.get(key)
                if not (path and os.path.exists(path)):
                    continue

                # Cek apakah path berasal dari direktori sementara atau arsip
                if temp_base_path in path:
                    base_to_strip = temp_base_path
                else:
                    base_to_strip = archive_base_path
                
                # Buat path relatif (arcname) dengan benar
                arcname = os.path.relpath(path, base_to_strip)

                if os.path.isdir(path):
                    # Tambahkan semua file di dalam direktori ini ke zip
                    for root, _, files in os.walk(path):
                        for file in files:
                            full_disk_path = os.path.join(root, file)
                            # Buat path relatif dari file di dalam subfolder
                            relative_file_path = os.path.relpath(full_disk_path, path)
                            final_arcname = os.path.join(arcname, relative_file_path)
                            zipf.write(full_disk_path, final_arcname)
                else:
                    # Kasus untuk file tunggal
                    arcname = os.path.relpath(path, archive_base_path)
                    zipf.write(path, arcname)
                    
    return zip_path

def export_pdf(data: List[Dict[str, Any]], user_name: str) -> str:
    """Membuat laporan PDF yang komprehensif dari hasil analisis."""
    report_dir = PATHS.get('report_dir', 'temp/reports')
    os.makedirs(report_dir, exist_ok=True)
    
    # --- Mengambil semua parameter dari config ---
    logo_path = PATHS.get('logo')
    theme = UI_CONFIG.get('theme', {})
    primary_color = tuple(int(theme.get('primary_color', '#1f77b4')[i:i+2], 16) for i in (1, 3, 5))
    secondary_color = tuple(int(theme.get('card_border_dark', '#30363d')[i:i+2], 16) for i in (1, 3, 5))
    cover_title = PDF_CONFIG.get('cover_title')
    footer_text = PDF_CONFIG.get('footer_text')
    author_name = APP_CONFIG.get('author')
    author_affiliation = APP_CONFIG.get('affiliation')

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)
    
    # --- Halaman Sampul ---
    pdf.add_page()
    if logo_path and os.path.exists(logo_path):
        pdf.image(logo_path, x=(pdf.w - 100)/2, y=40, w=100)
    pdf.set_y(160)
    pdf.set_font("helvetica", 'B', 18)
    pdf.set_text_color(*primary_color)
    pdf.cell(0, 12, _safe_text(cover_title), ln=1, align='C')
    pdf.ln(5)
    pdf.set_text_color(*secondary_color)
    
    info_list = [("Dianalisis oleh: ", user_name), ("Dicetak pada: ", datetime.now().strftime('%Y-%m-%d %H:%M') + " UTC")]
    for label, value in info_list:
        pdf.set_font("helvetica", 'B', 14)
        pdf.set_x((pdf.w - pdf.get_string_width(label + value)) / 2)
        pdf.cell(pdf.get_string_width(label), 8, label, ln=0)
        pdf.set_font("helvetica", '', 14)
        pdf.cell(0, 8, _safe_text(value), ln=1)
        
    pdf.set_y(-25)
    pdf.set_font("helvetica", '', 8)
    pdf.set_text_color(*primary_color)
    pdf.cell(0, 5, _safe_text(footer_text), 0, 1, 'L')
    pdf.set_font("helvetica", 'I', 8)
    pdf.set_text_color(*secondary_color)
    pdf.cell(0, 5, _safe_text(f"{author_name} - {author_affiliation}"), 0, 0, 'L')

    # --- Halaman Dasbor Statistik ---
    df_data = pd.DataFrame(data)
    chart_paths = _create_dashboard_charts(df_data)
    
    if chart_paths:
        hist_path, pie_path = chart_paths
        pdf.add_page()
        pdf.set_font("helvetica", 'B', 18)
        pdf.set_text_color(*primary_color)
        pdf.cell(0, 15, "Dasbor Rangkuman Statistik", ln=True, align='C')
        
        # Gambar teks metrik langsung ke PDF
        y_pos = 30
        jumlah_total = len(df_data)
        type_counts = df_data['media_type'].value_counts()
        breakdown_parts = [f"{count} {media_type.replace('_', ' ').title()}{'s' if count > 1 else ''}" for media_type, count in type_counts.items()]
        breakdown_text = ", ".join(breakdown_parts)

        pdf.set_font("helvetica", '', 14); pdf.set_text_color(*secondary_color)
        pdf.set_y(y_pos); pdf.cell(0, 8, "Total Citra Dinalisis", align='C', ln=1)
        y_pos += 8
        
        pdf.set_font("helvetica", 'B', 48); pdf.set_text_color(*primary_color)
        pdf.set_y(y_pos); pdf.cell(0, 25, str(jumlah_total), align='C', ln=1)
        y_pos += 25
        
        pdf.set_font("helvetica", 'I', 10); pdf.set_text_color(128, 128, 128)
        pdf.set_y(y_pos); pdf.cell(0, 8, breakdown_text, align='C', ln=1)
        
        # Tempatkan gambar grafik di bawah teks
        y_charts_start = y_pos + 15
        
        # Inisialisasi posisi Y untuk grafik berikutnya
        next_y_pos = y_charts_start

        # Buka gambar untuk mendapatkan ukurannya dalam piksel
        with Image.open(hist_path) as img:
            px_w, px_h = img.size
        # Tentukan lebar gambar di PDF
        pdf_img_width = pdf.w - 50
        # Hitung tinggi gambar di PDF sesuai rasio aspeknya
        pdf_img_height = pdf_img_width * (px_h / px_w)

        if hist_path:
            # Tempatkan gambar pertama
            pdf.image(hist_path, x=25, y=next_y_pos, w=pdf_img_width)
            # Perbarui posisi Y untuk grafik berikutnya
            next_y_pos += pdf_img_height

        if pie_path:
            # Beri jarak antar grafik
            space_between_charts = 10
            # Tempatkan gambar kedua di posisi Y yang sudah diperbarui + margin
            pdf.image(pie_path, x=25, y=next_y_pos + space_between_charts, w=pdf_img_width)
        
        # Footer di halaman dasbor
        pdf.set_y(-25)
        pdf.set_font("helvetica", '', 8)
        pdf.set_text_color(*primary_color)
        pdf.cell(0, 5, _safe_text(footer_text), 0, 1, 'L')
        pdf.set_font("helvetica", 'I', 8)
        pdf.set_text_color(*secondary_color)
        pdf.cell(0, 5, _safe_text(f"{author_name} - {author_affiliation}"), 0, 0, 'L')
        pdf.cell(0, 5, f"Halaman {pdf.page_no() - 1}", 0, 0, 'R')
        
        # Hapus direktori temporary
        if hist_path:
             shutil.rmtree(os.path.dirname(hist_path))

    # --- Konten untuk setiap hasil analisis ---
    with tempfile.TemporaryDirectory() as tmpdir: # Direktori sementara untuk semua gambar
        for i, item in enumerate(data):
            pdf.add_page()
            
            # --- Header Halaman ---
            file_name = item.get('source_filename', 'Unknown')
            pdf.set_text_color(*primary_color); pdf.set_font("helvetica", 'B', 16)
            pdf.cell(40, 10, "Hasil Analisis:", ln=0); pdf.set_font("helvetica", '', 16)
            pdf.multi_cell(0, 10, f" {_safe_text(file_name)}")
            
            # Ambil dan format waktu analisis
            raw_timestamp = item.get('analyzed_at')
            formatted_time = "N/A"  # Nilai default jika timestamp tidak ada

            if raw_timestamp:
                try:
                    # Ubah string ISO ke objek datetime, lalu format ulang
                    dt_object = datetime.fromisoformat(str(raw_timestamp))
                    formatted_time = dt_object.strftime('%Y-%m-%d %H:%M')
                except (ValueError, TypeError):
                    # Fallback jika format tidak valid atau bukan string
                    formatted_time = str(raw_timestamp)

            pdf.set_text_color(*secondary_color); pdf.set_font("helvetica", 'B', 12)
            pdf.cell(40, 8, "Waktu Analisis:", ln=0); pdf.set_font("helvetica", '', 12)
            pdf.cell(0, 8, f" {formatted_time} UTC", ln=1); pdf.ln(5)

            # --- Gambar Pratinjau ---
            img_width, spacing, y_pos = 90, 10, pdf.get_y()
            original_preview = _get_image_for_pdf(item.get("original_path"), max_size=(512, 512))
            overlay_preview = _get_image_for_pdf(item.get("overlay_path"), max_size=(512, 512))
            
            # Tentukan tinggi sel untuk judul
            caption_height = 10

            for preview, x_pos, caption in [(original_preview, 10, "Citra Asli"), (overlay_preview, 10 + img_width + spacing, "Overlay Segmentasi")]:
                # 1. Tempatkan judulnya terlebih dahulu
                pdf.set_xy(x_pos, y_pos)
                pdf.set_font("helvetica", 'I', 10)
                pdf.cell(img_width, caption_height, caption, align='C')

                # 2. Hitung posisi Y untuk gambar (tepat di bawah judul)
                image_y_pos = y_pos + caption_height

                # 3. Tempatkan gambarnya
                if preview:
                    temp_path = os.path.join(tmpdir, f"img_{i}_{caption.replace(' ', '')}.png")
                    preview.save(temp_path, "PNG")
                    pdf.image(temp_path, x=x_pos, y=image_y_pos, w=img_width)
                else:
                    pdf.rect(x_pos, image_y_pos, img_width, img_width, 'D')

            # --- Tabel Hasil Segmentasi ---
            pdf.set_y(y_pos + img_width + 15)
            pdf.set_font("helvetica", 'B', 16); pdf.set_text_color(*primary_color); pdf.cell(0, 10, "Hasil Segmentasi", ln=1)
            
            pdf.set_fill_color(*primary_color); pdf.set_text_color(255, 255, 255); pdf.set_font("helvetica", 'B', 12)
            pdf.cell(95, 8, "Parameter", 1, 0, 'C', fill=True); pdf.cell(95, 8, "Nilai", 1, 1, 'C', fill=True)
            
            pdf.set_text_color(*secondary_color); pdf.set_font("helvetica", 'B', 12)
            rows_seg = [
                ("Tutupan Awan", f"{item.get('cloud_coverage', 0):.2f}%"),
                ("Nilai Okta", str(item.get('okta_value', '0'))),
                ("Kondisi Langit", _safe_text(item.get('sky_condition', '')))
            ]
            for row in rows_seg:
                pdf.cell(95, 8, row[0], 1, 0, 'C'); pdf.cell(95, 8, row[1], 1, 1, 'C')

            # --- Tabel Hasil Klasifikasi ---
            pdf.ln(5); pdf.set_font("helvetica", 'B', 16); pdf.set_text_color(*primary_color); pdf.cell(0, 10, "Hasil Klasifikasi", ln=1)
            
            pdf.set_fill_color(*primary_color); pdf.set_text_color(255, 255, 255); pdf.set_font("helvetica", 'B', 12)
            pdf.cell(95, 8, "Jenis Awan", 1, 0, 'C', fill=True); pdf.cell(95, 8, "Keyakinan", 1, 1, 'C', fill=True)
            pdf.set_text_color(*secondary_color)
            
            # Parsing detail klasifikasi dari string
            preds_str = item.get("classification_details", "")
            preds_list = []
            if preds_str:
                try:
                    parts = preds_str.split('; ')
                    for part in parts:
                        match = re.match(r"(.+)\s\((\d+\.\d+)%\)", part)
                        if match:
                            preds_list.append((match.group(1), float(match.group(2))))
                except:
                    preds_list = [] # Fallback jika parsing gagal
            
            if not preds_list:
                pdf.cell(95, 8, "N/A", 1, 0, 'C'); pdf.cell(95, 8, "N/A", 1, 1, 'C')
            else:
                for idx, (label, score) in enumerate(preds_list):
                    pdf.set_font("helvetica", 'B' if idx == 0 else '', 12)
                    pdf.cell(95, 8, _safe_text(label), 1, 0, 'C'); pdf.cell(95, 8, f"{score:.2f}%", 1, 1, 'C')

            # --- Footer Halaman ---
            pdf.set_y(-25)
            pdf.set_font("helvetica", '', 8); pdf.set_text_color(*primary_color)
            pdf.cell(0, 5, _safe_text(footer_text), 0, 1, 'L')
            pdf.set_font("helvetica", 'I', 8); pdf.set_text_color(*secondary_color)
            pdf.cell(0, 5, _safe_text(f"{author_name} - {author_affiliation}"), 0, 0, 'L')
            pdf.cell(0, 5, f"Halaman {pdf.page_no() - 1}", 0, 0, 'R')

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') + "_UTC"
    output_path = os.path.join(report_dir, f"report_{timestamp}.pdf")
    pdf.output(output_path)
    return output_path