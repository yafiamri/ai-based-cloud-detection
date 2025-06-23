# Lokasi File: core/utils/hash_utils.py
"""
Menyediakan fungsi murni untuk menghasilkan hash unik (MD5)
dari konten gambar dan video, dengan menyertakan parameter analisis.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import hashlib
import logging

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

def hash_image(
    image: Image.Image,
    seg_threshold: float,
    cls_threshold: float,
    roi_mask: Optional[np.ndarray] = None
) -> str:
    """Menghitung hash MD5 unik berdasarkan konten gambar dan parameter analisis.

    Fungsi ini menghasilkan "sidik jari" digital untuk sebuah proses analisis
    gambar. Dengan memasukkan nilai threshold dan ROI ke dalam hash, fungsi ini
    memastikan bahwa analisis terhadap gambar yang sama dengan parameter yang
    berbeda akan menghasilkan hash yang unik.

    Args:
        image (Image.Image): Objek gambar input dari PIL.
        seg_threshold (float): Ambang batas (0.0-1.0) yang digunakan untuk segmentasi.
        cls_threshold (float): Ambang batas (0.0-1.0) yang digunakan для klasifikasi.
        roi_mask (Optional[np.ndarray]): Mask NumPy opsional untuk Region of Interest.

    Returns:
        str: String MD5 heksadesimal (32 karakter) yang unik untuk kombinasi
             gambar dan parameter ini. Mengembalikan string kosong jika terjadi error.
    """
    try:
        payload = bytearray()
        # 1. Tambahkan data gambar
        payload.extend(image.resize((256, 256)).tobytes())

        # 2. Tambahkan data ROI jika ada
        if roi_mask is not None:
            resized_mask = cv2.resize(roi_mask.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
            payload.extend(resized_mask.tobytes())
            
        # 3. Tambahkan data parameter threshold
        param_str = f"seg{seg_threshold:.2f}_cls{cls_threshold:.2f}"
        payload.extend(param_str.encode())

        return hashlib.md5(payload).hexdigest()
    except Exception as e:
        log.error(f"Gagal saat hashing gambar: {e}")
        return ""

def hash_video(
    video_path: Path, 
    frame_interval: int, 
    seg_threshold: float, 
    cls_threshold: float
) -> Tuple[str, float]:
    """Menghasilkan hash unik untuk analisis video berdasarkan parameter yang diberikan.

    Hashing ini tidak memproses seluruh file video demi efisiensi. Sebaliknya,
    ia mengambil sampel dari beberapa frame kunci (awal, tengah, akhir) dan
    menggabungkannya dengan metadata video serta semua parameter analisis yang
    dapat diubah pengguna (interval dan thresholds).

    Args:
        video_path (Path): Path menuju file video input.
        frame_interval (int): Interval pengambilan frame analisis (dalam detik).
        seg_threshold (float): Ambang batas yang digunakan untuk segmentasi setiap frame.
        cls_threshold (float): Ambang batas yang digunakan untuk klasifikasi setiap frame.

    Returns:
        Tuple[str, float]: Sebuah tuple yang berisi:
            - string_hash_md5 (str): Hash MD5 heksadesimal yang unik.
            - durasi_video_detik (float): Total durasi video dalam detik.
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError("Tidak dapat membuka video.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps if fps > 0 else 0.0

        # 1. Ambil sampel dari frame kunci untuk hash
        frame_indices = {0, total_frames // 2, total_frames - 1}
        frames_data = bytearray()
        for idx in sorted(list(frame_indices)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames_data.extend(cv2.resize(frame, (128, 128)).tobytes())
        cap.release()

        # 2. Tambahkan metadata DAN SEMUA PARAMETER ke payload hash
        param_str = (
            f"frames{total_frames}_fps{fps:.2f}_interval{frame_interval}s_"
            f"seg{seg_threshold:.2f}_cls{cls_threshold:.2f}"
        )
        frames_data.extend(param_str.encode())

        digest = hashlib.md5(frames_data).hexdigest()
        return digest, duration
    except Exception as e:
        log.error(f"Gagal saat hashing video {video_path}: {e}")
        return "", 0.0