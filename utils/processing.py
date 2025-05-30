# utils/processing.py
import os
import sys
import cv2
import logging
import shutil
import hashlib
import subprocess
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Dict, List, Tuple

from utils.segmentation import prepare_input_tensor, predict_segmentation
from utils.classification import predict_classification
from utils.image import is_duplicate_hash, hash_image, hash_video, append_to_history, load_result_from_history
from utils.roi import generate_roi_mask

# Konfigurasi global
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG_PATH = os.path.join(PROJECT_ROOT, "ffmpeg", "bin", "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
logger = logging.getLogger(__name__)

# 🧩 Helper Functions
def generate_filename(ext: str = "jpg") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = hashlib.md5(timestamp.encode()).hexdigest()[:8]
    return f"{timestamp}_{unique}.{ext}"

def save_image(img: Image.Image, subdir: str, ext: str = "jpg") -> str:
    os.makedirs(os.path.join("temps", "history", subdir), exist_ok=True)
    filename = generate_filename(ext)
    path = os.path.join("temps", "history", subdir, filename)
    img.save(path)
    return path

def extract_roi_image(np_img: np.ndarray, mask_roi: np.ndarray) -> Image.Image:
    roi_rgb = np_img * mask_roi[..., None]
    return Image.fromarray((roi_rgb * 255).astype(np.uint8))

def calculate_metrics(pred: np.ndarray, mask_roi: np.ndarray) -> Tuple[float, str]:
    awan_pixels = np.sum(pred * mask_roi)
    total_pixels = np.sum(mask_roi)
    coverage = 100 * awan_pixels / total_pixels if total_pixels > 0 else 0.0
    oktaf = min(8, int(round((coverage / 100) * 8)))
    kondisi_idx = min(oktaf // 2, 5)
    kondisi_list = ["Cerah", "Sebagian Cerah", "Sebagian Berawan", "Berawan", "Hampir Tertutup", "Tertutup"]
    return coverage, kondisi_list[kondisi_idx]

def initialize_video_writer(path, fourcc, height, width, fps):
    if width % 2 != 0: width += 1
    if height % 2 != 0: height += 1
    return cv2.VideoWriter(path, fourcc, fps, (width, height))

def process_ffmpeg_conversion(input_path: str):
    output_path = input_path.replace("_raw.mp4", ".mp4")
    try:
        subprocess.run([
            FFMPEG_PATH, "-y", "-i", input_path,
            "-vcodec", "libx264", "-crf", "23", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", output_path
        ], check=True)
        os.remove(input_path)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode() if e.stderr else e}")

def aggregate_predictions(preds_list: List[List[Tuple[str, float]]]) -> Tuple[str, List[Tuple[str, float]]]:
    counter = {}
    for preds in preds_list:
        for label, conf in preds:
            counter.setdefault(label, []).append(conf)
    avg_conf = [(label, np.mean(conf)) for label, conf in counter.items()]
    sorted_preds = sorted(avg_conf, key=lambda x: x[1], reverse=True)
    return (sorted_preds[0][0] if sorted_preds else "-", sorted_preds[:3])

# 🖼️ Gambar: analyze_image()
def analyze_image(name: str, img: Image.Image, roi_type: str, canvas_data, seg_model, cls_model,
                  seg_threshold=0.5, cls_threshold=0.25) -> Tuple[Optional[Dict], Optional[str]]:
    try:
        np_img = np.array(img) / 255.0
        h, w = img.height, img.width
        roi_mask = generate_roi_mask(np_img, mode="auto" if roi_type == "Otomatis" else "manual",
                                     canvas_data=canvas_data, target_size=(w, h))
        if roi_mask.sum() == 0:
            return None, f"⚠️ ROI kosong untuk {name}"

        # Simpan ROI
        roi_img = Image.fromarray((roi_mask * 255).astype(np.uint8))
        roi_path = save_image(roi_img, "roi", ext="png")

        # Cek hash dan duplikat
        file_hash = hash_image(img, roi_mask)
        if is_duplicate_hash(file_hash):
            return load_result_from_history(file_hash), None

        # Segmentasi
        tensor = prepare_input_tensor(np_img * roi_mask[..., None])
        pred = predict_segmentation(seg_model, tensor, seg_threshold)
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        # Klasifikasi
        roi_img_cls = extract_roi_image(np_img, roi_mask)
        roi_cls_path = save_image(roi_img_cls, "roi")
        preds = predict_classification(cls_model, roi_cls_path, threshold=cls_threshold)

        # Metrik segmentasi
        coverage, kondisi = calculate_metrics(pred, roi_mask)

        # Simpan hasil visual
        original_path = save_image(img, "original")
        mask_path = save_image(Image.fromarray((pred * 255).astype(np.uint8)), "mask", ext="png")
        overlay_np = np.where((pred * roi_mask)[..., None], 0.6 * np_img + 0.4 * np.array([1, 0, 0]), np_img)
        overlay_img = Image.fromarray((overlay_np * 255).astype(np.uint8))
        
        try:
            logo = Image.open(os.path.join(PROJECT_ROOT, "assets", "logo.png")).convert("RGBA")
            logo_scale = 0.15  # 15% dari lebar gambar
            logo_width = int(overlay_img.width * logo_scale)
            logo_ratio = logo.width / logo.height
            logo_size = (logo_width, int(logo_width / logo_ratio))
            logo = logo.resize(logo_size, Image.LANCZOS)
        
            if overlay_img.mode != "RGBA":
                overlay_img = overlay_img.convert("RGBA")
        
            pos = (overlay_img.width - logo.width - 10, overlay_img.height - logo.height - 10)
            overlay_img.paste(logo, pos, mask=logo)
            overlay_img = overlay_img.convert("RGB")
        except Exception as e:
            logger.error(f"Gagal menambahkan watermark gambar: {str(e)}")
            
        overlay_path = save_image(overlay_img, "overlay")

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "original_image_path": original_path,
            "mask_image_path": mask_path,
            "overlay_image_path": overlay_path,
            "path_roi": roi_path,
            "path_preview": "",
            "coverage": round(coverage, 2),
            "oktaf": min(8, int(round((coverage / 100) * 8))),
            "kondisi_langit": kondisi,
            "jenis_awan": preds[0][0] if preds else "-",
            "top_preds": preds,
            "durasi": 0.0,
            "hash": file_hash
        }

        append_to_history(result)
        return result, None

    except Exception as e:
        logger.error(f"Error analyzing image {name}: {str(e)}", exc_info=True)
        return None, str(e)

# 🎥 Video: analyze_video()
def analyze_video(video_path: str, seg_model, cls_model, interval_detik: int = 1,
                  roi_mask: Optional[np.ndarray] = None, roi_mode: str = "auto") -> Tuple[List[Dict], Optional[str]]:
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], "🚨 Video tidak dapat dibuka"

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval_detik)
        basename = os.path.splitext(os.path.basename(video_path))[0]
        hash_vid, durasi = hash_video(video_path)

        if is_duplicate_hash(hash_vid):
            return [load_result_from_history(hash_vid)], None

        frame_data = {
            "coverages": [],
            "kondisi_all": [],
            "top_preds_all": [],
            "preview_path": ""
        }

        # Output paths
        raw_overlay_path = f"temps/history/overlay/{basename}_overlay_raw.mp4"
        raw_mask_path = f"temps/history/mask/{basename}_mask_raw.mp4"
        final_overlay_path = raw_overlay_path.replace("_raw.mp4", ".mp4")
        final_mask_path = raw_mask_path.replace("_raw.mp4", ".mp4")
        original_target_path = f"temps/history/original/{basename}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_overlay = None
        out_mask = None
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue

            result = process_single_frame(frame, seg_model, cls_model, roi_mask, roi_mode, frame_idx)
            if result:
                if frame_idx == 0:
                    frame_data['preview_path'] = result['preview_path']
                if out_overlay is None or out_mask is None:
                    h, w = result['overlay_frame'].shape[:2]
                    out_overlay = initialize_video_writer(raw_overlay_path, fourcc, h, w, fps=1.0 / interval_detik)
                    out_mask = initialize_video_writer(raw_mask_path, fourcc, h, w, fps=1.0 / interval_detik)
                out_overlay.write(result['overlay_frame'])
                out_mask.write(cv2.cvtColor(result['mask_frame'], cv2.COLOR_GRAY2BGR))

                frame_data['coverages'].append(result['coverage'])
                frame_data['kondisi_all'].append(result['kondisi'])
                frame_data['top_preds_all'].append(result['preds'])

            frame_idx += 1

        cap.release()
        if out_overlay: out_overlay.release()
        if out_mask: out_mask.release()

        # Konversi hasil video
        process_ffmpeg_conversion(raw_overlay_path)
        process_ffmpeg_conversion(raw_mask_path)

        shutil.copy(video_path, original_target_path)

        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name": f"{basename}.mp4",
            "original_video_path": original_target_path,
            "preview_video_path": frame_data['preview_path'],
            "overlay_video_path": final_overlay_path,
            "mask_video_path": final_mask_path,
            "path_roi": "",
            "path_preview": frame_data['preview_path'],
            "coverage": round(np.mean(frame_data['coverages']), 2) if frame_data['coverages'] else 0.0,
            "oktaf": int(round(np.mean(frame_data['coverages']) / 100 * 8)) if frame_data['coverages'] else 0,
            "kondisi_langit": max(set(frame_data['kondisi_all']), key=frame_data['kondisi_all'].count) if frame_data['kondisi_all'] else "-",
            "jenis_awan": aggregate_predictions(frame_data['top_preds_all'])[0],
            "top_preds": aggregate_predictions(frame_data['top_preds_all'])[1],
            "durasi": round(durasi, 2),
            "hash": hash_vid
        }

        append_to_history(result)
        return [result], None

    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}", exc_info=True)
        return [], str(e)

def process_single_frame(frame, seg_model, cls_model, roi_mask, roi_mode, frame_idx):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        np_img = frame_rgb / 255.0
        img_pil = Image.fromarray(frame_rgb)

        if roi_mask is None:
            roi_mask_final = generate_roi_mask(np_img, mode=roi_mode, canvas_data=None, target_size=(img_pil.width, img_pil.height))
        else:
            roi_mask_final = cv2.resize(roi_mask.astype(np.uint8), (img_pil.width, img_pil.height), interpolation=cv2.INTER_NEAREST)

        if roi_mask_final.sum() == 0:
            return None

        tensor = prepare_input_tensor(np_img * roi_mask_final[..., None])
        pred = predict_segmentation(seg_model, tensor)
        pred = cv2.resize(pred, img_pil.size, interpolation=cv2.INTER_NEAREST)
        coverage, kondisi = calculate_metrics(pred, roi_mask_final)

        roi_img = extract_roi_image(np_img, roi_mask_final)
        roi_path = save_image(roi_img, "roi")
        preds = predict_classification(cls_model, roi_path)

        overlay_np = np.where((pred * roi_mask_final)[..., None], 0.6 * np_img + 0.4 * np.array([1,0,0]), np_img)
        overlay_frame = cv2.cvtColor((overlay_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        try:
            logo = Image.open(os.path.join(PROJECT_ROOT, "assets", "logo.png")).convert("RGBA")
            logo_scale = 0.15  # 15% dari lebar frame
            logo_width = int(overlay_frame.shape[1] * logo_scale)
            logo_ratio = logo.width / logo.height
            logo_size = (logo_width, int(logo_width / logo_ratio))
            logo = logo.resize(logo_size, Image.LANCZOS)
        
            frame_pil = Image.fromarray(cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
            pos = (frame_pil.width - logo.width - 10, frame_pil.height - logo.height - 10)
            frame_pil.paste(logo, pos, mask=logo)
        
            overlay_frame = cv2.cvtColor(np.array(frame_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Gagal menambahkan watermark video frame: {str(e)}")

        preview_path = ""
        if frame_idx == 0:
            preview_img = Image.fromarray((np_img * 255).astype(np.uint8))
            preview_path = save_image(preview_img, "preview")

        return {
            "coverage": coverage,
            "kondisi": kondisi,
            "preds": preds,
            "overlay_frame": overlay_frame,
            "mask_frame": (pred * 255).astype(np.uint8),
            "preview_path": preview_path
        }

    except Exception as e:
        logger.error(f"Frame {frame_idx} error: {str(e)}", exc_info=True)
        return None