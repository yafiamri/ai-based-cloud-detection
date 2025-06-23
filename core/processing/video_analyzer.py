# Lokasi File: core/processing/video_analyzer.py
"""
Orkestrator untuk menganalisis file video menggunakan metode overlay masker
pada video asli, kini terintegrasi penuh dengan sistem hashing dan parameter.
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
import time
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import Counter
import logging

# Impor dari modul-modul lain di dalam 'core'
from . import image_analyzer
from core.utils import ffmpeg_wrapper, hash_utils
from core.io import file_manager

log = logging.getLogger(__name__)

def _create_transparent_mask(
    binary_mask_np: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0), # Merah
    opacity: int = 128 # 0-255 (sekitar 50% transparan)
) -> Image.Image:
    """
    Mengubah mask biner (0 atau 255) menjadi gambar RGBA transparan.
    Area awan akan berwarna, area lain akan transparan.
    """
    h, w = binary_mask_np.shape
    # Buat kanvas RGBA 4-channel yang sepenuhnya transparan
    transparent_image = np.zeros((h, w, 4), dtype=np.uint8)
    
    # Tentukan di mana lokasi awan (piksel bernilai > 0)
    cloud_locations = binary_mask_np > 0
    
    # Isi warna (R,G,B) dan tingkat opacity (A) hanya di lokasi awan
    transparent_image[cloud_locations] = [*color, opacity]
    
    return Image.fromarray(transparent_image, "RGBA")

def _aggregate_video_results(frame_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mengagregasi hasil dari semua frame yang dianalisis menjadi ringkasan yang kaya."""
    if not frame_results:
        return {}
    
    avg_coverage = np.mean([r.get("cloud_coverage_percent", 0) for r in frame_results])
    
    # Ambil nilai yang paling sering muncul (modus) untuk data kualitatif
    all_dominant_types = [r.get("dominant_cloud_type") for r in frame_results if r.get("dominant_cloud_type")]
    dominant_cloud_type = Counter(all_dominant_types).most_common(1)[0][0] if all_dominant_types else "N/A"
    
    all_sky_conditions = [r.get("sky_condition") for r in frame_results if r.get("sky_condition")]
    sky_condition = Counter(all_sky_conditions).most_common(1)[0][0] if all_sky_conditions else "N/A"

    # Ambil nilai rata-rata yang dibulatkan untuk okta
    avg_oktas = round(np.mean([r.get("sky_oktas", 0) for r in frame_results]))

    return {
        "average_cloud_coverage_percent": avg_coverage,
        "overall_dominant_cloud_type": dominant_cloud_type,
        "representative_sky_condition": sky_condition,
        "representative_sky_oktas": int(avg_oktas),
        "processed_frame_count": len(frame_results),
    }

def analyze_video(
    video_path: Path,
    seg_model: Any,
    seg_device: Any,
    cls_model: Any,
    *,
    frame_interval_seconds: int = 1,
    seg_threshold: float = 0.5,
    cls_threshold: float = 0.2,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Orkestrasi lengkap untuk analisis file video dengan metode overlay."""
    start_time = time.time()
    
    # 1. Hashing di awal, menyertakan SEMUA parameter
    file_hash, video_duration = hash_utils.hash_video(
        video_path=video_path,
        frame_interval=frame_interval_seconds,
        seg_threshold=seg_threshold,
        cls_threshold=cls_threshold
    )
    
    # Cek riwayat untuk analisis yang identik
    existing_result = file_manager.check_if_hash_exists(file_hash)
    if existing_result:
        log.info(f"Analisis identik untuk {video_path.name} ditemukan di riwayat. Mengembalikan hasil lama.")
        return existing_result

    # Jika tidak ada, lanjutkan analisis...
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Tidak dapat membuka file video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_step = int(fps * frame_interval_seconds)
    if frame_step == 0: frame_step = 1

    temp_mask_dir = file_manager.TEMPS_DIR / f"video_masks_{file_hash[:10]}"
    temp_mask_dir.mkdir(exist_ok=True)
    
    frame_results = []
    current_frame_idx = 0

    try:
        while cap.isOpened():
            ret, frame_bgr = cap.read()
            if not ret: break
            
            if current_frame_idx % frame_step == 0:
                if progress_callback:
                    progress_callback(current_frame_idx / total_frames)
                
                frame_pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
                
                # 2. Meneruskan parameter threshold ke image_analyzer
                result_package = image_analyzer.analyze_image(
                    image=frame_pil,
                    file_name=f"{video_path.stem}_frame_{current_frame_idx}",
                    seg_model=seg_model, seg_device=seg_device, cls_model=cls_model,
                    seg_threshold=seg_threshold,
                    cls_threshold=cls_threshold
                )
                
                frame_results.append(result_package)
                
                mask_pil = result_package["images"]["mask"]
                transparent_mask = _create_transparent_mask(np.array(mask_pil))
                mask_path = temp_mask_dir / f"mask_{current_frame_idx:06d}.png"
                transparent_mask.save(mask_path)

            current_frame_idx += 1
    finally:
        cap.release()
    
    if not frame_results:
        log.warning("Tidak ada frame yang diproses dari video.")
        shutil.rmtree(temp_mask_dir)
        return None

    aggregated_results = _aggregate_video_results(frame_results)
    
    video_output_name = f"result_{file_hash[:10]}_{video_path.name}"
    video_output_path = file_manager.HISTORY_DIR / "overlay" / video_output_name
    
    log.info("Memulai proses overlay video dengan FFmpeg...")
    if any(temp_mask_dir.iterdir()):
         ffmpeg_wrapper.overlay_image_sequence(
             base_video_path=video_path,
             image_sequence_dir=temp_mask_dir,
             output_path=video_output_path,
             fps=int(fps)
         )

    shutil.rmtree(temp_mask_dir)
    
    # 3. Susun Paket Hasil Akhir yang Lengkap sesuai skema CSV
    final_package = {
        "file_name": video_path.name,
        "file_hash": file_hash,
        "data_type": "Video",
        "segmentation_threshold": seg_threshold,
        "classification_threshold": cls_threshold,
        "video_frame_interval": frame_interval_seconds,
        "cloud_coverage_percent": aggregated_results.get("average_cloud_coverage_percent"),
        "sky_oktas": aggregated_results.get("representative_sky_oktas"),
        "sky_condition": aggregated_results.get("representative_sky_condition"),
        "dominant_cloud_type": aggregated_results.get("overall_dominant_cloud_type"),
        "top_predictions_str": "N/A untuk video agregat", # Tidak relevan untuk video
        "duration_seconds": time.time() - start_time,
        "original_video_path": str(video_path), # Untuk ditampilkan di UI
        "output_video_path": str(video_output_path) if video_output_path.exists() else None,
        "metrics": aggregated_results, # Data agregat mentah
    }

    return final_package