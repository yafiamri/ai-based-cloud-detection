# Lokasi File: core/processing/image_analyzer.py
"""
Orkestrator untuk menganalisis gambar tunggal.
Fungsi utama 'analyze_image' adalah "otak" dari alur kerja analisis gambar.
"""

from __future__ import annotations
import torch
import cv2
import numpy as np
from PIL import Image
import time
from typing import Any, Dict, Optional, Tuple, List

# Impor dari modul-modul yang telah kita standarisasi
from core.roi import auto_roi, manual_roi
from core.utils import image_utils, hash_utils

# --- HELPER FUNCTIONS YANG DIPINDAHKAN KE SINI ---

def _prepare_input_tensor(image_np: np.ndarray, target_size: int = 512) -> torch.Tensor:
    """Pra-pemrosesan: Mengubah NumPy array menjadi tensor siap inferensi."""
    img_float = image_np.astype(np.float32) / 255.0
    resized = cv2.resize(img_float, (target_size, target_size))
    # Ubah HWC (Tinggi, Lebar, Channel) -> CHW (Channel, Tinggi, Lebar)
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0)
    return tensor

@torch.no_grad()
def _run_segmentation(model: Any, device: torch.device, input_tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Menjalankan inferensi segmentasi dan mengembalikan mask biner."""
    input_tensor = input_tensor.to(device)
    # Asumsi model mengembalikan dict dengan kunci 'out'
    pred = model(input_tensor)["out"].squeeze().cpu().numpy()
    return (pred > threshold).astype(np.uint8)

@torch.no_grad()
def _run_classification(model: Any, image_np: np.ndarray, threshold: float = 0.5) -> List[Tuple[str, float]]:
    """Menjalankan inferensi klasifikasi dan mengembalikan daftar prediksi."""
    results = model.predict(source=image_np, verbose=False)
    probs = results[0].probs
    # Dapatkan nama kelas dari model
    class_names = results[0].names
    preds = [
        (class_names[i], float(p))
        for i, p in enumerate(probs.data)
        if p > threshold
    ]
    return sorted(preds, key=lambda x: x[1], reverse=True)


def _calculate_metrics(segmentation_mask: np.ndarray, roi_mask: np.ndarray) -> Dict[str, Any]:
    """Menghitung metrik dari hasil segmentasi."""
    # Pastikan roi_mask adalah biner (0 atau 1)
    roi_binary = (roi_mask > 0).astype(np.uint8)
    # Hitung area di dalam ROI
    total_roi_pixels = np.sum(roi_binary)
    # Hitung area awan di dalam ROI
    cloud_pixels_in_roi = np.sum(segmentation_mask * roi_binary)
    coverage = (cloud_pixels_in_roi / total_roi_pixels) * 100.0 if total_roi_pixels > 0 else 0.0
    oktas = min(8, round((coverage / 100) * 8))
    # Mapping dari Okta ke Kondisi Langit
    sky_condition_map = {
        0: "Cerah (Clear)",
        1: "Hampir Cerah (Mostly Clear)",
        2: "Agak Berawan (Slightly Cloudy)",
        3: "Berawan Sebagian (Partly Cloudy)",
        4: "Berawan Sebagian (Partly Cloudy)",
        5: "Berawan (Cloudy)",
        6: "Berawan (Cloudy)",
        7: "Hampir Tertutup (Mostly Overcast)",
        8: "Tertutup (Overcast)",
    }
    return {
        "cloud_coverage_percent": coverage,
        "sky_oktas": oktas,
        "sky_condition": sky_condition_map.get(oktas, "N/A"),
    }

# --- FUNGSI UTAMA ANALYZER ---

def analyze_image(
    image: Image.Image,
    file_name: str,
    seg_model: Any,
    seg_device: torch.device,
    cls_model: Any,
    *,
    roi_mode: str = "auto",
    canvas_data: Optional[Dict[str, Any]] = None,
    seg_threshold: float = 0.5,
    cls_threshold: float = 0.2,
) -> Dict[str, Any]:
    """
    Orkestrasi lengkap untuk analisis gambar tunggal.
    Menerima gambar, menjalankan semua model, dan mengembalikan satu dictionary hasil.

    Args:
        image: Objek gambar PIL.Image input.
        seg_model: Model segmentasi yang sudah dimuat.
        seg_device: Device untuk model segmentasi ('cuda' atau 'cpu').
        cls_model: Model klasifikasi yang sudah dimuat.
        roi_mode: Mode ROI ('auto' atau 'manual').
        canvas_data: Data dari streamlit-drawable-canvas jika mode manual.
        seg_threshold: Threshold untuk binerisasi mask segmentasi.
        cls_threshold: Threshold untuk menampilkan hasil klasifikasi.

    Returns:
        Sebuah dictionary komprehensif berisi semua hasil analisis (metrik dan gambar).
    """
    start_time = time.time()
    
    # 0. Konversi ke format NumPy (BGR untuk OpenCV, RGB untuk lainnya)
    image_rgb_np = np.array(image.convert("RGB"))
    image_bgr_np = cv2.cvtColor(image_rgb_np, cv2.COLOR_RGB2BGR)

    # 1. Hasilkan ROI Mask
    roi_mask = None
    if roi_mode == "manual" and canvas_data:
        st.toast(f"Menerapkan ROI manual untuk {file_name}...", icon="✍️")
        try:
            # Pastikan format data canvas benar
            orig_h, orig_w = canvas_data["original_size"][::-1] # Balik (w,h) -> (h,w)
            canvas_w, canvas_h = canvas_data["canvas_size"]
            # Panggil fungsi ROI manual kita dari 'core'
            roi_mask = canvas_to_mask(
                json_data=canvas_data["json_data"],
                orig_size=(orig_h, orig_w),
                canvas_size=(canvas_w, canvas_h)
            )
        except Exception as e:
            log.error(f"Gagal membuat mask dari kanvas: {e}")
            # Fallback ke ROI otomatis jika manual gagal
            roi_mask = auto_roi.detect_circle_roi(image_bgr_np)
    # Jika mode adalah otomatis atau manual gagal, jalankan auto_roi
    if roi_mask is None:
        roi_mask = auto_roi.detect_circle_roi(image_bgr_np)
    # Ubah ROI mask menjadi 0 atau 1 untuk kalkulasi
    roi_mask_binary = (roi_mask > 0).astype(np.uint8)

    # 2. Hasilkan Hash Unik (PENYESUAIAN KUNCI)
    # Hash kini menyertakan gambar, ROI, dan kedua threshold.
    file_hash = hash_utils.hash_image(
        image=image,
        roi_mask=roi_mask_binary,
        seg_threshold=seg_threshold,
        cls_threshold=cls_threshold
    )

    # 3. Jalankan Segmentasi
    input_tensor = _prepare_input_tensor(image_rgb_np)
    seg_mask = _run_segmentation(seg_model, seg_device, input_tensor, seg_threshold)
    # Resize mask ke ukuran gambar asli
    seg_mask_resized = cv2.resize(seg_mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
    
    # 4. Jalankan Klasifikasi
    # Menerapkan ROI sebelum klasifikasi bisa meningkatkan akurasi
    image_with_roi = cv2.bitwise_and(image_rgb_np, image_rgb_np, mask=roi_mask_binary)
    predictions = _run_classification(cls_model, image_with_roi, cls_threshold)
    
    # 5. Hitung Metrik
    metrics = _calculate_metrics(seg_mask_resized, roi_mask_binary)
    
    # 6. Hasilkan Gambar Output (sebagai objek PIL, bukan file)
    seg_mask_pil = Image.fromarray(seg_mask_resized * 255, "L")
    roi_mask_pil = Image.fromarray(roi_mask, "L")
    overlay_pil = image_utils.overlay_mask(image, seg_mask_resized, color=(255, 0, 0), alpha=0.5)
    preview_pil = image_utils.resize_and_pad(image_bgr_np, (256, 256))
    preview_pil = Image.fromarray(cv2.cvtColor(preview_pil, cv2.COLOR_BGR2RGB))
    
    # 7. Susun Hasil Akhir (Paket Data Lengkap)
    result_package = {
        # Identitas & Parameter
        "file_name": file_name,
        "file_hash": file_hash,
        "data_type": "Gambar",
        "segmentation_threshold": seg_threshold,
        "classification_threshold": cls_threshold,
        
        # Hasil Segmentasi
        "cloud_coverage_percent": metrics["cloud_coverage_percent"],
        "sky_oktas": metrics["sky_oktas"],
        "sky_condition": metrics["sky_condition"],
        
        # Hasil Klasifikasi
        "dominant_cloud_type": predictions[0][0] if predictions else "Tidak Terdeteksi",
        "top_predictions": predictions,
        "top_predictions_str": "; ".join([f"{label} ({p:.1%})" for label, p in predictions]),
        
        # Kinerja
        "duration_seconds": time.time() - start_time,
        
        # Aset Gambar
        "images": {
            "original": image,
            "mask": seg_mask_pil,
            "overlay": overlay_pil,
            "roi": roi_mask_pil,
            "preview": preview_pil,
        }
    }
    
    return result_package