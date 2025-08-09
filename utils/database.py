# utils/database.py
import sqlite3
import pandas as pd
import os
import hashlib
import json
from typing import Dict, Any, List, Optional
import streamlit as st

# Impor konfigurasi yang sudah dimuat
from .config import config

# Ambil path database dari file konfigurasi
DB_PATH = config.get('paths', {}).get('database_file', 'data/history.db')

def get_pipeline_version_hash() -> str:
    """Membuat 'sidik jari' untuk versi pipeline analisis saat ini."""
    try:
        version_params = {
            "cls_model": config['models']['classification']['weight_path'],
            "seg_model": config['models']['segmentation']['weight_path'],
            "seg_input_size": config['models']['segmentation']['input_size'],
            "cls_threshold": config['analysis']['confidence_threshold'],
            "seg_threshold": config['analysis']['segmentation_threshold']
        }
        params_string = json.dumps(version_params, sort_keys=True)
        return hashlib.sha256(params_string.encode()).hexdigest()
    except KeyError as e:
        st.error(f"Kunci konfigurasi hilang untuk membuat hash pipeline: {e}.")
        return "invalid_config"

def get_db_connection() -> Optional[sqlite3.Connection]:
    """Membuat dan mengembalikan koneksi ke database SQLite."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        st.error(f"Gagal terhubung ke database di {DB_PATH}: {e}")
        return None

def init_db():
    """Menginisialisasi tabel 'history'."""
    query = """
    CREATE TABLE IF NOT EXISTS history (
        -- Kunci & Versi
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        pipeline_version_hash TEXT NOT NULL,
        file_hash TEXT NOT NULL,
        analysis_hash TEXT NOT NULL,
        
        -- Metadata Sumber File
        source_filename TEXT NOT NULL,
        media_type TEXT,
        file_size_bytes INTEGER,
        
        -- Metadata Proses Analisis
        analyzed_at TEXT NOT NULL,
        analysis_duration_sec REAL,
        
        -- Hasil Analisis
        cloud_coverage REAL,
        okta_value INTEGER,
        sky_condition TEXT,
        dominant_cloud_type TEXT,
        classification_details TEXT,
        
        -- Path Penyimpanan Artefak
        original_path TEXT,
        mask_path TEXT,
        overlay_path TEXT,

        -- Constraint untuk memastikan setiap analisis unik
        UNIQUE(analysis_hash)
    );
    """
    with get_db_connection() as conn:
        if conn:
            conn.cursor().execute(query)
            conn.commit()

def find_history(analysis_hash: str) -> Optional[Dict[str, Any]]:
    """Mencari entri riwayat berdasarkan HASH ANALISIS yang unik."""
    # DIUBAH: Fungsi ini sekarang jauh lebih sederhana
    query = "SELECT * FROM history WHERE analysis_hash = ?"
    with get_db_connection() as conn:
        if conn:
            row = conn.cursor().execute(query, (analysis_hash,)).fetchone()
            return dict(row) if row else None

def add_history_entry(entry: Dict[str, Any]):
    """Menambahkan satu entri hasil analisis ke dalam tabel history."""
    # DIUBAH: Query INSERT sekarang menyertakan analysis_hash
    query = """
    INSERT OR IGNORE INTO history (
        analysis_hash, pipeline_version_hash, file_hash, source_filename, media_type, file_size_bytes,
        analyzed_at, analysis_duration_sec, cloud_coverage, okta_value, sky_condition,
        dominant_cloud_type, classification_details, original_path, mask_path, overlay_path
    ) VALUES (
        :analysis_hash, :pipeline_version_hash, :file_hash, :source_filename, :media_type, :file_size_bytes,
        :analyzed_at, :analysis_duration_sec, :cloud_coverage, :okta_value, :sky_condition,
        :dominant_cloud_type, :classification_details, :original_path, :mask_path, :overlay_path
    )
    """
    with get_db_connection() as conn:
        if conn:
            conn.cursor().execute(query, entry)
            conn.commit()

def get_history_df() -> pd.DataFrame:
    """Mengambil semua data riwayat dari database."""
    try:
        with get_db_connection() as conn:
            return pd.read_sql_query("SELECT * FROM history", conn, index_col="id") if conn else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def delete_history_entries(ids: List[int]):
    """Menghapus entri riwayat berdasarkan daftar ID."""
    if not ids: return
    with get_db_connection() as conn:
        if conn:
            placeholders = ','.join('?' for _ in ids)
            conn.cursor().execute(f"DELETE FROM history WHERE id IN ({placeholders})", ids)
            conn.commit()

def get_paths_for_deletion(ids: List[int]) -> List[str]:
    """Mengambil semua path file yang terkait dengan ID entri yang akan dihapus."""
    if not ids: return []
    with get_db_connection() as conn:
        if conn:
            query = "SELECT original_path, mask_path, overlay_path FROM history WHERE id IN ({})".format(','.join('?' for _ in ids))
            cursor = conn.cursor().execute(query, ids)
            paths = {path for row in cursor.fetchall() for path in row if path and pd.notna(path)}
            return list(paths)
    return []

# Inisialisasi DB saat modul diimpor untuk memastikan tabel selalu ada
init_db()