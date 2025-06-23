# Lokasi File: core/io/file_manager.py
"""
Mengelola semua operasi I/O terkait penyimpanan file analisis
dan data riwayat dalam format CSV dengan skema penamaan standar.
"""

from __future__ import annotations
import pandas as pd
import shutil
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List
import logging

# Konfigurasi Logging
log = logging.getLogger(__name__)

# Definisi Path Konstan
ROOT = Path(__file__).resolve().parents[2]
TEMPS_DIR = ROOT / "temps"
HISTORY_DIR = TEMPS_DIR / "history"
CSV_PATH = HISTORY_DIR / "history.csv"

# Kolom CSV Standar (English, snake_case) - Versi Definitif
CSV_COLUMNS = [
    # 1. Identitas Analisis
    "timestamp", "file_name", "file_hash", "data_type",
    # 2. Parameter Analisis
    "segmentation_threshold", "classification_threshold", "video_frame_interval",
    # 3. Hasil Segmentasi
    "cloud_coverage_percent", "sky_oktas", "sky_condition",
    # 4. Hasil Klasifikasi
    "dominant_cloud_type", "top_predictions_str",
    # 5. Artefak & Kinerja
    "original_path", "mask_path", "overlay_path", "roi_path", "preview_path", 
    "duration_seconds"
]

def initialize_storage():
    """Memastikan semua direktori yang diperlukan untuk penyimpanan ada."""
    SUBDIRS = ["original", "mask", "overlay", "roi", "preview"]
    for subdir in SUBDIRS:
        (HISTORY_DIR / subdir).mkdir(parents=True, exist_ok=True)
    
    if not CSV_PATH.is_file():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_PATH, index=False)

initialize_storage()


def save_analysis_artifacts(result_data: Dict[str, Any]) -> Dict[str, str]:
    """Menyimpan semua artefak gambar (original, mask, dll.) ke disk."""
    saved_paths = {}
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for key, image_obj in result_data.get("images", {}).items():
        if image_obj:
            filename = f"{timestamp_str}_{key}_{result_data.get('file_name', 'unknown_file')}"
            save_path = HISTORY_DIR / key / filename
            try:
                image_obj.save(save_path)
                saved_paths[f"{key}_path"] = str(save_path)
            except Exception as e:
                log.error(f"Gagal menyimpan artefak '{key}': {e}")
                saved_paths[f"{key}_path"] = ""
    return saved_paths

def add_record_to_history(record: Dict[str, Any]):
    """Menambahkan satu baris record ke history.csv menggunakan skema standar."""
    try:
        # Memastikan hanya kolom yang valid yang ditulis
        record_to_write = {col: record.get(col) for col in CSV_COLUMNS}
        
        df = pd.read_csv(CSV_PATH)
        new_record_df = pd.DataFrame([record_to_write])
        df = pd.concat([df, new_record_df], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
    except Exception as e:
        log.error(f"Gagal menambahkan record ke {CSV_PATH}: {e}")

def get_history() -> List[Dict[str, Any]]:
    """Membaca seluruh file history.csv."""
    if not CSV_PATH.is_file():
        return []
    try:
        df = pd.read_csv(CSV_PATH).fillna('') # Ganti NaN dengan string kosong
        return df.to_dict('records')
    except Exception as e:
        log.error(f"Gagal membaca file riwayat {CSV_PATH}: {e}")
        return []

def check_if_hash_exists(file_hash: str) -> Optional[Dict[str, Any]]:
    """
    Memeriksa apakah hash sudah ada di history.csv.

    Args:
        file_hash: String hash MD5 yang akan diperiksa.

    Returns:
        Dictionary dari baris riwayat jika ditemukan, jika tidak None.
    """
    if not file_hash or not CSV_PATH.is_file():
        return None
    
    try:
        df = pd.read_csv(CSV_PATH)
        # Cari baris yang cocok dengan hash
        result = df[df['file_hash'] == file_hash]
        if not result.empty:
            # Kembalikan baris pertama yang ditemukan sebagai dictionary
            return result.iloc[0].to_dict()
        return None
    except Exception as e:
        log.error(f"Gagal memeriksa hash di {CSV_PATH}: {e}")
        return None

def delete_history_entry(file_hash: str):
    """Menghapus entri dari CSV berdasarkan file_hash dan menghapus file terkait."""
    if not CSV_PATH.is_file():
        return

    df = pd.read_csv(CSV_PATH)
    # Gunakan file_hash sebagai identifier unik
    entry_to_delete = df[df['file_hash'] == file_hash]

    if not entry_to_delete.empty:
        path_keys_to_delete = [
            "original_path", "mask_path", "overlay_path", "roi_path", "preview_path"
        ]
        for _, row in entry_to_delete.iterrows():
            for col in path_keys_to_delete:
                if pd.notna(row.get(col)) and row[col] != '':
                    file_path = Path(row[col])
                    if file_path.is_file():
                        try:
                            file_path.unlink()
                        except OSError as e:
                            log.error(f"Gagal menghapus file {file_path}: {e}")
        
        df = df[df['file_hash'] != file_hash]
        df.to_csv(CSV_PATH, index=False)