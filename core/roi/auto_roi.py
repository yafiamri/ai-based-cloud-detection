# core/roi/auto_roi.py
"""
Auto-ROI: deteksi lingkaran (all-sky) terbesar di citra.
"""

from __future__ import annotations
import logging
from typing import Any

import cv2
import numpy as np

log = logging.getLogger(__name__)


def detect_circle_roi(image_rgb: np.ndarray) -> np.ndarray:
    """
    Deteksi lingkaran ROI terbesar menggunakan Hough-lik.

    Args:
        image_rgb: Citra RGB uint8 shape (H, W, 3).

    Returns:
        Mask uint8 {0,1} shape (H, W). Jika gagal, kembalikan mask all-ones.
    """
    try:
        if image_rgb.size == 0:
            raise ValueError("Empty image")

        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        # Gaussian blur adaptif
        min_dim = min(gray.shape)
        k_blur = max(5, (min_dim // 100) * 2 + 1)
        blur = cv2.GaussianBlur(gray, (k_blur, k_blur), 0)

        # Otsu threshold + closing
        _, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        k_close = max(3, min_dim // 100)
        kernel = np.ones((k_close, k_close), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Cari kontur terbesar
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = [c for c in contours if cv2.contourArea(c) > 100]
        if not contours:
            return np.ones_like(gray, dtype=np.uint8)

        # Enclosing circle
        cnt = max(contours, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)

        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (int(cx), int(cy)), int(r), 1, thickness=-1)
        return mask

    except Exception as e:
        log.error(f"[Auto-ROI] Error: {e}")
        # Fallback: mask penuh
        return np.ones(image_rgb.shape[:2], dtype=np.uint8)