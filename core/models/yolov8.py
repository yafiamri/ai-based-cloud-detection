# Lokasi File: core/models/yolov8.py
"""
Mendefinisikan pemuat untuk model klasifikasi awan YOLOv8.

File ini bertanggung jawab HANYA untuk:
1. Menyediakan fungsi untuk memuat bobot model (.pt) yang sudah dilatih.
"""

from __future__ import annotations
from typing import Optional
from ultralytics import YOLO
import torch
from pathlib import Path
import logging

log = logging.getLogger(__name__)

# --- PERUBAHAN KUNCI: Gunakan Path.cwd() yang sudah terbukti andal ---
PROJECT_ROOT = Path.cwd()

# =============================================================================
# FUNGSI PEMUAT MODEL (MODEL LOADER)
# =============================================================================

def get_model(weight_path: str | Path | None = None) -> Optional[YOLO]:
    """
    Membuat instance model klasifikasi YOLOv8 dan memuat bobot.

    Args:
        weight_path: Path menuju file bobot model .pt.

    Returns:
        Objek model YOLO. Mengembalikan None jika gagal.
    """
    if weight_path is None:
        weight_path = PROJECT_ROOT / "models" / "yolov8.pt"

    try:
        log.info(f"Mencoba memuat bobot model dari path: {weight_path}")
        if not Path(weight_path).is_file():
            raise FileNotFoundError
        model = YOLO(str(weight_path))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        log.info(f"Model klasifikasi YOLOv8 berhasil dimuat di device: {device}.")
        return model
    except FileNotFoundError:
        log.error(f"FATAL: File bobot model TIDAK DITEMUKAN di '{weight_path}'.")
        return None
    except Exception as e:
        log.error(f"FATAL: Gagal saat memuat model YOLOv8: {e}")
        return None