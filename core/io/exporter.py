# Lokasi File: core/io/exporter.py
"""
Menyediakan fungsionalitas untuk mengekspor data hasil analisis
ke berbagai format (CSV, ZIP, PDF) menggunakan skema data standar.
"""

from __future__ import annotations
import csv
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List
from fpdf import FPDF
from datetime import datetime
import logging

log = logging.getLogger(__name__)

# Helper function untuk FPDF agar aman menangani karakter non-latin
def _safe_text(text: str) -> str:
    """Encode teks ke format yang aman untuk FPDF."""
    return text.encode('latin-1', 'replace').decode('latin-1')

def export_to_csv(selected_records: List[Dict[str, Any]]) -> bytes:
    """Mengekspor daftar hasil analisis ke format CSV dalam bentuk bytes."""
    output = BytesIO()
    string_buffer = ""
    
    header = [
        "timestamp", "file_name", "cloud_coverage_percent", "sky_oktas", 
        "sky_condition", "dominant_cloud_type", "top_predictions_str", "duration_seconds"
    ]
    string_buffer += ",".join(header) + "\n"
    
    for rec in selected_records:
        row = [
            str(rec.get("timestamp", "")),
            str(rec.get("file_name", "")),
            f"{rec.get('cloud_coverage_percent', 0.0):.2f}",
            str(rec.get("sky_oktas", 0)),
            str(rec.get("sky_condition", "")),
            str(rec.get("dominant_cloud_type", "")),
            str(rec.get("top_predictions_str", "")).replace(",", ";"), # Ganti koma agar tidak merusak CSV
            f"{rec.get('duration_seconds', 0.0):.2f}",
        ]
        # Pastikan setiap item di-quote untuk menangani koma di dalam data
        string_buffer += ",".join(f'"{item}"' for item in row) + "\n"
        
    return string_buffer.encode('utf-8')

def export_to_zip(selected_records: List[Dict[str, Any]], output_path: Path) -> Path:
    """Membuat arsip ZIP yang berisi semua file gambar dari hasil analisis yang dipilih."""
    path_keys_to_zip = ["original_path", "mask_path", "overlay_path", "preview_path", "roi_path"]
    
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Tambahkan CSV ke dalam ZIP
        csv_bytes = export_to_csv(selected_records)
        zf.writestr("hasil_analisis.csv", csv_bytes)

        for rec in selected_records:
            for path_key in path_keys_to_zip:
                file_str_path = rec.get(path_key, "")
                if file_str_path:
                    p = Path(file_str_path)
                    if p.is_file():
                        arc_dir = path_key.replace("_path", "")
                        zf.write(p, arcname=Path(arc_dir) / p.name)
    return output_path

def export_to_pdf(selected_records: List[Dict[str, Any]], author: str, output_path: Path) -> Path:
    """Buat PDF laporan komprehensif dari hasil analisis awan."""
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(10, 10, 10)
    logo_path = Path("assets/logo.png") # Pastikan path logo benar

    # Style configuration
    primary_color = (41, 128, 185)
    secondary_color = (80, 80, 80)
    
    # --- Cover Page ---
    pdf.add_page()
    if logo_path.is_file():
        pdf.image(str(logo_path), x=(210 - 100)/2, y=40, w=100) # 210mm adalah lebar A4
    pdf.set_y(150)
    pdf.set_font("Helvetica", 'B', 22)
    pdf.set_text_color(*primary_color)
    pdf.cell(0, 12, "Laporan Hasil Deteksi Awan Berbasis AI", ln=1, align='C')
    pdf.ln(10)
    pdf.set_font("Helvetica", '', 14)
    pdf.set_text_color(*secondary_color)
    pdf.cell(0, 8, _safe_text(f"Disusun oleh: {author}"), ln=1, align='C')
    pdf.cell(0, 8, f"Dicetak pada: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}", ln=1, align='C')
    
    # --- Konten untuk setiap hasil analisis ---
    for rec in selected_records:
        pdf.add_page()
        
        # Header Halaman
        pdf.set_font("Helvetica", 'B', 16)
        pdf.set_text_color(*primary_color)
        pdf.cell(0, 10, _safe_text(f"Analisis untuk: {rec.get('file_name', 'N/A')}"), ln=1)
        pdf.set_font("Helvetica", '', 12)
        pdf.set_text_color(*secondary_color)
        pdf.cell(0, 8, f"Waktu Analisis: {rec.get('timestamp', '-')}", ln=1)
        pdf.ln(5)

        # Gambar berdampingan
        img_width, y_pos, spacing = 90, pdf.get_y(), 10
        x_pos_1, x_pos_2 = 10, 10 + img_width + spacing
        
        original_path = rec.get("original_path")
        if original_path and Path(original_path).is_file():
            pdf.image(original_path, x=x_pos_1, y=y_pos, w=img_width)
        
        overlay_path = rec.get("overlay_path")
        if overlay_path and Path(overlay_path).is_file():
            pdf.image(overlay_path, x=x_pos_2, y=y_pos, w=img_width)

        pdf.set_xy(x_pos_1, y_pos + img_width)
        pdf.set_font("Helvetica", 'I', 10)
        pdf.cell(img_width, 8, "Gambar Asli", ln=False, align='C')
        pdf.set_xy(x_pos_2, y_pos + img_width)
        pdf.cell(img_width, 8, "Hasil Overlay Segmentasi", ln=True, align='C')

        # Tabel Hasil
        pdf.set_y(y_pos + img_width + 15)
        
        # Fungsi bantu untuk membuat baris tabel
        def create_table_row(label, value):
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(60, 8, _safe_text(label), border=1)
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 8, _safe_text(str(value)), border=1)

        # Data Segmentasi
        pdf.set_font("Helvetica", 'B', 14)
        pdf.set_text_color(*primary_color)
        pdf.cell(0, 10, "Hasil Segmentasi", ln=1)
        pdf.set_text_color(*secondary_color)
        create_table_row("Cakupan Awan", f"{rec.get('cloud_coverage_percent', 0.0):.2f}%")
        create_table_row("Kondisi Langit", f"{rec.get('sky_condition', '-')} ({rec.get('sky_oktas', '-')} Okta)")
        
        pdf.ln(5)
        
        # Data Klasifikasi
        pdf.set_font("Helvetica", 'B', 14)
        pdf.set_text_color(*primary_color)
        pdf.cell(0, 10, "Hasil Klasifikasi", ln=1)
        pdf.set_text_color(*secondary_color)
        create_table_row("Jenis Awan Dominan", rec.get('dominant_cloud_type', '-'))
        # Asumsi top_predictions adalah list of tuples [(label, score), ...]
        preds_list = rec.get('top_predictions', []) 
        if isinstance(preds_list, list) and preds_list:
            formatted_preds = "\n".join([f"- {label}: {score:.2%}" for label, score in preds_list])
            create_table_row("Prediksi Teratas", formatted_preds)
        else:
            create_table_row("Prediksi Teratas", rec.get('top_predictions_str', '-'))

    pdf.output(str(output_path))
    return output_path