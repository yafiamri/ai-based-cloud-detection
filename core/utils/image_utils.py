# Lokasi File: core/utils/image_utils.py
"""
Berisi fungsi-fungsi utilitas murni untuk manipulasi gambar
yang tidak bergantung pada framework UI manapun.
"""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from typing import Tuple
from pathlib import Path
import logging

log = logging.getLogger(__name__)

def resize_and_pad(
    image_np: np.ndarray, 
    target_size: Tuple[int, int] = (256, 256)
) -> np.ndarray:
    """
    Mengubah ukuran gambar ke target_size dengan padding untuk menjaga rasio aspek.

    Args:
        image_np: Gambar dalam format NumPy array (dari cv2.imread, BGR).
        target_size: Tuple (lebar, tinggi) untuk ukuran output.

    Returns:
        Gambar NumPy array yang telah diubah ukurannya dengan padding.
    """
    h, w = image_np.shape[:2]
    target_w, target_h = target_size
    
    ratio = min(target_w / w, target_h / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    
    resized_image = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Buat kanvas baru dengan warna hitam dan tempelkan gambar di tengah
    # Menentukan channel dari gambar input
    if len(image_np.shape) == 3:
        padded_image = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    else: # Grayscale
        padded_image = np.zeros((target_h, target_w), dtype=np.uint8)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    padded_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_image
    
    return padded_image

def overlay_mask(
    image: Image.Image, 
    mask: np.ndarray, 
    color: Tuple[int, int, int] = (255, 0, 0), 
    alpha: float = 0.5
) -> Image.Image:
    """
    Menumpuk mask segmentasi di atas gambar asli.

    Args:
        image: Gambar asli dalam format PIL Image.
        mask: Mask biner (0 dan 1 atau 0 dan 255) dengan channel tunggal.
        color: Warna mask dalam format (R, G, B).
        alpha: Tingkat transparansi overlay (0.0 - 1.0).

    Returns:
        Gambar hasil overlay dalam format PIL Image.
    """
    # Pastikan gambar dalam format RGBA untuk proses blending
    image_rgba = image.convert("RGBA")
    
    # Buat gambar overlay berwarna dari mask
    colored_mask_img = Image.new("RGBA", image.size, color)
    
    # Konversi mask NumPy menjadi PIL Image (jika perlu)
    if isinstance(mask, np.ndarray):
        # Pastikan mask adalah biner 0 atau 255
        if mask.max() == 1:
            mask = mask * 255
        mask_pil = Image.fromarray(mask.astype(np.uint8), 'L')
    else:
        mask_pil = mask

    # Gabungkan gambar overlay berwarna dengan gambar asli, menggunakan mask sebagai panduan
    overlay_image = Image.composite(colored_mask_img, image_rgba, mask_pil)
    
    # Lakukan blending dengan gambar asli untuk efek transparansi
    blended_image = Image.blend(image_rgba, overlay_image, alpha)
    
    return blended_image.convert("RGB")

def load_demo_images(
    demo_dir: Union[str, Path] = "assets/demo"
) -> List[Tuple[str, Image.Image]]:
    """
    Muat semua gambar demo dari folder demo.

    Args:
        demo_dir: Path ke direktori demo (jpg/png).

    Returns:
        List of (filename, PIL.Image).
    """
    demo_path = Path(demo_dir)
    if not demo_path.exists():
        log.warning(f"Direktori demo tidak ditemukan: {demo_path}")
        return []

    result: List[Tuple[str, Image.Image]] = []
    for file in sorted(demo_path.iterdir()):
        if file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            img = Image.open(file).convert("RGB")
            result.append((file.name, img))
        except Exception as e:
            log.warning(f"Gagal memuat demo image {file}: {e}")
    return result