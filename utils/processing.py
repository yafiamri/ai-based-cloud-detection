# utils/processing.py
import hashlib
import cv2
import numpy as np
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Any, Optional, IO, Tuple

# Impor dari modul utilitas lain dan konfigurasi
from .config import config
from .segmentation import prepare_input_tensor, predict_segmentation, detect_circle_roi
from .classification import predict_classification

# Ambil konfigurasi yang relevan
ANALYSIS_CONFIG = config.get('analysis', {})
PDF_CONFIG = config.get('pdf_report', {})
MODEL_CONFIG = config.get('models', {})

def get_file_hash(file_like_object: IO[bytes]) -> str:
    """
    HANYA menghitung hash SHA-256 dari konten sebuah file.
    Digunakan untuk identifikasi file unik di antrean UI.
    """
    sha256_hash = hashlib.sha256()
    file_like_object.seek(0)
    for byte_block in iter(lambda: file_like_object.read(4096), b""):
        sha256_hash.update(byte_block)
    file_like_object.seek(0)
    return sha256_hash.hexdigest()

def get_analysis_hash(
    file_hash: str,
    pipeline_hash: str,
    config: Dict[str, Any]
) -> str:
    """
    HANYA membuat hash unik untuk sebuah tugas analisis berdasarkan
    kombinasi hash file dan konfigurasinya. Digunakan untuk caching.
    """
    unique_parts = [
        f"file:{file_hash}",
        f"pipeline:{pipeline_hash}"
    ]
    
    roi_method = config.get('roi_method', 'Otomatis')
    unique_parts.append(f"roi_method:{roi_method}")
    
    if "Manual" in roi_method:
        canvas_data = config.get('canvas')
        json_str = str(canvas_data.json_data) if canvas_data and hasattr(canvas_data, 'json_data') else "no_canvas"
        unique_parts.append(f"canvas_data:{json_str}")
        
    if 'interval' in config:
        interval = config.get('interval', 5)
        unique_parts.append(f"interval:{interval}")
        
    combined_string = "|".join(unique_parts)
    return hashlib.sha256(combined_string.encode('utf-8')).hexdigest()

def create_enhanced_overlay(
    original_image: Image.Image,
    segmentation_mask: np.ndarray,
    roi_mask: np.ndarray,
    as_sticker: bool = False  # <-- PERUBAHAN: Tambah parameter baru
) -> Image.Image:
    """
    Membuat gambar overlay yang fleksibel.
    Bisa menghasilkan gambar jadi (opak) atau stiker transparan (RGBA).

    Args:
        original_image (Image.Image): Gambar asli, digunakan untuk dimensi.
        segmentation_mask (np.ndarray): Mask biner (0/1) dari segmentasi awan.
        roi_mask (np.ndarray): Mask biner (0/1) dari Region of Interest.
        as_sticker (bool): Jika True, hasilkan stiker transparan (RGBA).
                           Jika False, hasilkan gambar jadi (RGB).

    Returns:
        Image.Image: Gambar overlay final.
    """
    # Tentukan kanvas dasar berdasarkan mode
    if as_sticker:
        # Mode Stiker: Mulai dengan kanvas yang sepenuhnya transparan
        final_image = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    else:
        # Mode Normal: Mulai dengan gambar asli
        final_image = original_image.copy().convert("RGBA")
    
    # 2. Buat "stiker" transparan untuk overlay segmentasi (Logika ini tetap sama)
    overlay_sticker = Image.new("RGBA", final_image.size)
    draw = ImageDraw.Draw(overlay_sticker)
    
    red_color_transparent = (255, 0, 0, 100) # RGBA, alpha=100
    
    cloud_mask_pil = Image.fromarray((segmentation_mask * 255).astype(np.uint8))
    draw.bitmap((0, 0), cloud_mask_pil, fill=red_color_transparent)

    # Tempelkan stiker segmentasi ke kanvas utama
    final_image.alpha_composite(overlay_sticker)

    # 3. Gambar outline ROI berwarna kuning di atasnya
    draw_final = ImageDraw.Draw(final_image)
    
    roi_mask_pil = Image.fromarray((roi_mask * 255).astype(np.uint8))
    
    contours, _ = cv2.findContours(np.array(roi_mask_pil), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        polygon = cnt.flatten().tolist()
        if len(polygon) >= 2:
            draw_final.line(polygon, fill=(255, 255, 0, 180), width=3, joint="curve")
    
    # 4. Tambahkan watermark
    watermark_text = config.get('pdf_report', {}).get('footer_text', '© AI-Based Cloud Detection')
    img_w, img_h = final_image.size
    
    try:
        # 1. Buat teks di kanvas besar untuk menjaga kualitas (resolusi tinggi)
        font_watermark = ImageFont.truetype("arial.ttf", size=50) # Ukuran font besar & tetap
        
        # Dapatkan ukuran teks yang sebenarnya di kanvas besar
        text_bbox = font_watermark.getbbox(watermark_text)
        text_w, text_h = text_bbox[2], text_bbox[3]

        # Buat gambar watermark transparan
        watermark_img = Image.new('RGBA', (text_w, text_h), (255, 255, 255, 0))
        draw_watermark = ImageDraw.Draw(watermark_img)
        draw_watermark.text((0, 0), watermark_text, font=font_watermark, fill=(255, 255, 255, 180))

        # 2. Hitung ukuran target (selalu 1/3 dari lebar gambar utama)
        img_w, img_h = final_image.size
        target_w = img_w // 3
        # Hitung tinggi target secara proporsional
        target_h = int(target_w * (text_h / text_w))

        # 3. Resize gambar watermark ke ukuran target
        watermark_resized = watermark_img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        # 4. Tentukan posisi penempelan di pojok kanan bawah
        position = (img_w - target_w - 15, img_h - target_h - 15) # 15px padding

        # 5. Tempelkan watermark yang sudah di-resize ke gambar utama
        final_image.paste(watermark_resized, position, watermark_resized)

    except Exception as e:
        print(f"Gagal membuat watermark: {e}")
        # Fallback ke metode lama jika ada error (misal font tidak ditemukan)
        try:
            font = ImageFont.load_default()
            draw_final.text((img_w - 150, img_h - 30), watermark_text, font=font, fill=(255, 255, 255, 180))
        except:
            pass # Abaikan jika fallback juga gagal

    # Kembalikan format yang benar sesuai mode
    if as_sticker:
        return final_image  # Kembalikan sebagai RGBA (transparan)
    else:
        return final_image.convert("RGB")  # Kembalikan sebagai RGB (opak)

def analyze_single_image(
    image: Image.Image,
    seg_model: nn.Module,
    cls_model: Any,
    user_roi_mask: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Menganalisis satu gambar untuk segmentasi dan klasifikasi.
    Fungsi ini HANYA menghitung dan MENGEMBALIKAN hasil, tidak menyimpan file.

    Args:
        image (Image.Image): Gambar input dalam format PIL.
        seg_model (nn.Module): Model segmentasi yang sudah dimuat.
        cls_model (Any): Model klasifikasi (YOLO) yang sudah dimuat.
        user_roi_mask (Optional[np.ndarray]): Mask ROI dari pengguna (jika ada).

    Returns:
        Dict[str, Any]: Dictionary berisi semua hasil analisis.
    """
    # Simpan versi integer (uint8) dari gambar
    np_img_uint8 = np.array(image)
    # Normalisasi gambar asli dalam bentuk float [0, 1]
    np_img_float = np.array(image) / 255.0
    
    # Tentukan ROI: gunakan dari pengguna jika ada, jika tidak, deteksi otomatis
    roi_mask = user_roi_mask if user_roi_mask is not None else detect_circle_roi(np_img_float)
    
    # Proses Segmentasi
    # 1. Siapkan tensor input dari gambar asli
    #    (gunakan gambar asli untuk menjaga kualitas)
    tensor = prepare_input_tensor(np_img_float)
    pred_seg = predict_segmentation(seg_model, tensor)
    pred_seg_resized = cv2.resize(
        pred_seg, (image.width, image.height), interpolation=cv2.INTER_NEAREST
    )

    # 2. Ambil dimensi dari kedua mask
    h1, w1 = pred_seg_resized.shape
    h2, w2 = roi_mask.shape
    
    # 3. Jika ukurannya tidak sama, sinkronkan dengan memotong ke ukuran terkecil
    if h1 != h2 or w1 != w2:
        min_h = min(h1, h2)
        min_w = min(w1, w2)
        
        pred_seg_resized = pred_seg_resized[:min_h, :min_w]
        roi_mask = roi_mask[:min_h, :min_w]

    # 4. Terapkan ROI ke mask segmentasi
    final_mask = pred_seg_resized * roi_mask
    
    # 5. Hitung Metrik
    #    Hitung jumlah piksel awan dan ROI untuk menghitung coverage
    cloud_pixels = np.sum(final_mask)
    roi_pixels = np.sum(roi_mask)
    coverage = (cloud_pixels / roi_pixels * 100) if roi_pixels > 0 else 0
    okta = int(round((coverage / 100) * 8))

    # 6. Tentukan kondisi langit berdasarkan nilai Okta
    sky_conditions = ANALYSIS_CONFIG.get('sky_conditions', [])
    sky_condition_index = min(okta // 2, len(sky_conditions) - 1)
    sky_condition = sky_conditions[sky_condition_index]

    # Proses Klasifikasi
    # 1. Buat kanvas hitam seukuran citra asli (uint8)
    masked_image_np = np.zeros_like(np_img_uint8)
    if roi_mask is not None and np.any(roi_mask): # Pastikan roi_mask tidak kosong
        # 2. Salin piksel dari citra asli (uint8) ke kanvas hitam, HANYA di area ROI
        # `roi_mask > 0` akan membuat 'pintu' boolean, hanya piksel dengan
        # mask berwarna putih yang akan disalin.
        masked_image_np[roi_mask > 0] = np_img_uint8[roi_mask > 0]

    # "Penjaga": Cek apakah gambar hasil masking sepenuhnya hitam
    if np.max(masked_image_np) == 0:
        # Jika ya, jangan panggil model. Berikan hasil yang sudah ditentukan.
        preds = []
        dominant_cloud_type = "Tidak Terdeteksi"
        classification_details = "Tidak ada ROI di dalam gambar."
        all_cloud_types = MODEL_CONFIG.get('classification', {}).get('class_names', [])
        cloud_type_confidences = {label: np.nan for label in all_cloud_types}

    else: # Jika tidak hitam, baru konversi dan kirim ke model klasifikasi
        # 3. Konversi kembali ke format PIL Image
        #    (masking sudah dilakukan, jadi ini aman)
        image_for_classification = Image.fromarray(masked_image_np)

        # 4. Lakukan klasifikasi pada citra yang sudah dipotong sesuai ROI
        preds = predict_classification(cls_model, image_for_classification)
        
        # 5. Ambil hasil klasifikasi
        dominant_cloud_type = preds[0][0] if preds else "Tidak Terdeteksi"
        classification_details = "; ".join([f"{label} ({conf*100:.2f}%)" for label, conf in preds])
        cloud_type_confidences = {label: conf for label, conf in preds}

    return {
        "cloud_coverage": coverage,
        "okta_value": okta,
        "sky_condition": sky_condition,
        "dominant_cloud_type": dominant_cloud_type,
        "classification_details": classification_details,
        "cloud_type_confidences": cloud_type_confidences,
        "raw_predictions": preds,
        "segmentation_mask": final_mask,
        "roi_mask": roi_mask
    }