# utils/system.py
import os
import time
from typing import List

# Impor konfigurasi terpusat
from .config import config

def cleanup_temp_files(age_hours: int = 1) -> None:
    """
    Membersihkan file dan direktori kosong di dalam folder temp utama
    yang lebih tua dari batas usia yang ditentukan.

    Fungsi ini akan menelusuri subdirektori yang relevan seperti 'reports',
    'downloads', dan 'live_session_artefacts' secara rekursif.

    Args:
        age_hours (int): Batas usia file dalam jam sebelum dihapus.
    """
    # Ambil direktori temp utama dari config
    temp_dir = config.get('paths', {}).get('temp_dir', 'temp')
    
    # Langsung keluar jika direktori temp tidak ada
    if not os.path.isdir(temp_dir):
        return

    try:
        age_seconds = age_hours * 3600
        now = time.time()
        deleted_files_count = 0
        deleted_dirs_count = 0

        # --- TAHAP 1: Hapus semua FILE lama di dalam temp_dir dan subfolder-nya ---
        for root, _, files in os.walk(temp_dir):
            for name in files:
                file_path = os.path.join(root, name)
                try:
                    if os.path.isfile(file_path):
                        if (now - os.path.getmtime(file_path)) > age_seconds:
                            os.remove(file_path)
                            deleted_files_count += 1
                except FileNotFoundError:
                    continue

        # --- TAHAP 2: Hapus semua DIREKTORI KOSONG dari dalam ke luar ---
        for root, _, _ in os.walk(temp_dir, topdown=False):
            # Pengaman: Jangan hapus folder temp utama itu sendiri
            if root == temp_dir:
                continue
            
            try:
                if not os.listdir(root):
                    os.rmdir(root)
                    deleted_dirs_count += 1
            except OSError:
                continue

        if deleted_files_count > 0 or deleted_dirs_count > 0:
            print(f"Pembersihan otomatis: {deleted_files_count} file dan {deleted_dirs_count} direktori kosong telah dihapus dari folder '{temp_dir}'.")
            
    except Exception as e:
        print(f"Error saat membersihkan file sementara: {e}")