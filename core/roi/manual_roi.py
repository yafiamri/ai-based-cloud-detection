# core/roi/manual_roi.py
"""
Manual-ROI: konversi objek dari streamlit_drawable_canvas → binary mask.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

def canvas_to_mask(
    json_data: Dict[str, Any],
    orig_size: Tuple[int, int],
    canvas_size: Tuple[int, int],
) -> np.ndarray:
    """
    Buat mask biner dari shapes canvas.

    Args:
        json_data: Output JSON dari streamlit_drawable_canvas, berisi 'objects'.
        orig_size: Ukuran asli citra sebagai (height, width).
        canvas_size: Ukuran kanvas sebagai (width_canvas, height_canvas).

    Returns:
        Mask uint8 {0,1} shape orig_size.
    """
    mask = np.zeros(orig_size, dtype=np.uint8)
    try:
        objs = json_data.get("objects", [])
        if not objs:
            return mask

        orig_h, orig_w = orig_size
        canvas_w, canvas_h = canvas_size
        scale_x = orig_w / canvas_w
        scale_y = orig_h / canvas_h

        for obj in objs:
            typ = obj.get("type")
            if typ == "rect":
                x = int(obj["left"] * scale_x)
                y = int(obj["top"] * scale_y)
                w = int(obj["width"] * scale_x)
                h = int(obj["height"] * scale_y)
                cv2.rectangle(mask, (x, y), (x + w, y + h), color=1, thickness=-1)

            elif typ == "path":
                pts = [
                    (int(x * scale_x), int(y * scale_y))
                    for x, y in obj.get("path", [])
                ]
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [np.array(pts, np.int32)], color=1)

            elif typ == "line":
                x1 = int((obj.get("x1", 0) + obj.get("left", 0)) * scale_x)
                y1 = int((obj.get("y1", 0) + obj.get("top", 0)) * scale_y)
                x2 = int((obj.get("x2", 0) + obj.get("left", 0)) * scale_x)
                y2 = int((obj.get("y2", 0) + obj.get("top", 0)) * scale_y)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                r = int(np.hypot(x2 - x1, y2 - y1) / 2)
                if r > 0:
                    cv2.circle(mask, (cx, cy), r, color=1, thickness=-1)

    except Exception as e:
        log.error(f"[Manual-ROI] Error parsing canvas data: {e}")

    return mask