# utils/roi.py
import os
import cv2
import logging
import numpy as np
from PIL import Image
from typing import Optional, Union, Dict, Tuple

def detect_circle_roi(image_np: np.ndarray) -> np.ndarray:
    """Deteksi ROI lingkaran dengan validasi input dan optimasi parameter"""
    try:
        if image_np.size == 0 or len(image_np.shape) not in [2, 3]:
            logging.error("Input gambar tidak valid")
            return np.ones((512, 512), dtype=np.uint8)

        if image_np.dtype == np.float32:
            gray = cv2.cvtColor((image_np * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np

        blur_size = max(5, int(min(gray.shape)//100*2-1))
        blur = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel_size = max(3, int(min(gray.shape)//100))
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]

        if not valid_contours:
            return np.ones_like(gray, dtype=np.uint8)

        largest = max(valid_contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        (x, y), radius = cv2.minEnclosingCircle(hull)

        mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.circle(mask, (int(x), int(y)), int(radius*0.95), 255, -1)

        return (mask > 0).astype(np.uint8)

    except Exception as e:
        logging.error(f"Deteksi gagal: {str(e)}")
        return np.ones((512, 512), dtype=np.uint8)

def canvas_to_mask(
    canvas_result: Dict,
    orig_size: Tuple[int, int],
    canvas_size: Tuple[int, int]
) -> np.ndarray:
    """Konversi canvas ke mask dengan aspect ratio preservation"""
    mask = np.zeros(orig_size, dtype=np.uint8)
    try:
        if not canvas_result or "objects" not in canvas_result:
            return mask

        orig_h, orig_w = orig_size
        canvas_w, canvas_h = canvas_size

        scale_x = orig_w / canvas_w
        scale_y = orig_h / canvas_h

        for obj in canvas_result["objects"]:
            if obj["type"] == "rect":
                x = int(obj["left"] * scale_x)
                y = int(obj["top"] * scale_y)
                w = int(obj["width"] * scale_x)
                h = int(obj["height"] * scale_y)
                cv2.rectangle(mask, (x, y), (x+w, y+h), 1, -1)

            elif obj["type"] == "path" and obj.get("path"):
                points = []
                for item in obj["path"]:
                    if isinstance(item, list) and len(item) >= 3:
                        px = int(item[1] * scale_x)
                        py = int(item[2] * scale_y)
                        points.append([px, py])
                if len(points) >= 3:
                    cv2.fillPoly(mask, [np.array(points)], 1)

            elif obj["type"] == "line":
                x1 = int((obj["x1"] + obj["left"]) * scale_x)
                y1 = int((obj["y1"] + obj["top"]) * scale_y)
                x2 = int((obj["x2"] + obj["left"]) * scale_x)
                y2 = int((obj["y2"] + obj["top"]) * scale_y)
                cx, cy = (x1+x2)//2, (y1+y2)//2
                radius = int(np.hypot(x2-x1, y2-y1) // 2)
                if 0 <= cx < orig_w and 0 <= cy < orig_h and radius > 0:
                    cv2.circle(mask, (cx, cy), radius, 1, -1)

    except Exception as e:
        logging.error(f"Konversi gagal: {str(e)}")

    return mask

def generate_roi_mask(
    image: Union[np.ndarray, Image.Image],
    mode: str = "auto",
    canvas_data: Optional[Dict] = None,
    target_size: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None
) -> np.ndarray:
    """API utama untuk generasi ROI mask"""
    try:
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()

        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)

        if mode == "auto":
            roi_mask = detect_circle_roi(image_np)
        else:
            if canvas_data and "json_data" in canvas_data:
                roi_mask = canvas_to_mask(
                    canvas_data["json_data"],
                    orig_size=image_np.shape[:2],
                    canvas_size=(
                        canvas_data.get("width", 512),
                        canvas_data.get("height", 512)
                    )
                )
            else:
                logging.warning("Mode manual dipilih tapi data canvas tidak valid")
                roi_mask = detect_circle_roi(image_np)

        if target_size:
            roi_mask = cv2.resize(
                roi_mask.astype(np.uint8),
                target_size,
                interpolation=cv2.INTER_NEAREST
            )

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, (roi_mask * 255).astype(np.uint8))

        return roi_mask.astype(np.uint8)

    except Exception as e:
        logging.error(f"Generasi ROI gagal: {str(e)}")
        return np.ones(image_np.shape[:2], dtype=np.uint8)