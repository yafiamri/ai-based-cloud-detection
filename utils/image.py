# utils/image.py
import os
import cv2
import zipfile
import hashlib
import shutil
import logging
import numpy as np
import pandas as pd
from PIL import Image
from typing import List, Tuple, Optional, Union
from tempfile import TemporaryDirectory

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_temps():
    """Membersihkan folder temps dengan preservasi history"""
    try:
        if os.path.exists("temps"):
            for item in os.listdir("temps"):
                if item == "history":
                    continue
                path = os.path.join("temps", item)
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                    else:
                        shutil.rmtree(path)
                except Exception as e:
                    logger.error(f"Gagal menghapus {path}: {str(e)}")
    except Exception as e:
        logger.error(f"Error dalam clear_temps: {str(e)}")

def load_demo_images(demo_dir: str = "assets/demo") -> List[Tuple[str, Image.Image]]:
    """Memuat gambar demo dengan validasi format dan error handling"""
    images = []
    try:
        if not os.path.exists(demo_dir):
            raise FileNotFoundError(f"Folder demo {demo_dir} tidak ditemukan")

        for fname in os.listdir(demo_dir):
            try:
                if fname.lower().endswith(("jpg", "jpeg", "png")):
                    img_path = os.path.join(demo_dir, fname)
                    with Image.open(img_path) as img:
                        images.append((fname, img.convert("RGB")))
            except Exception as e:
                logger.warning(f"Gagal memuat {fname}: {str(e)}")
                continue

    except Exception as e:
        logger.error(f"Error load_demo_images: {str(e)}")

    return images

def load_uploaded_files(uploaded_files) -> Tuple[List[Tuple[str, Image.Image]], List[Tuple[str, bytes]]]:
    """Mengelola file upload dengan support ZIP, gambar, dan video"""
    images = []
    videos = []
    for file in uploaded_files:
        try:
            if file.name.lower().endswith('.zip'):
                with TemporaryDirectory() as tmpdir:
                    with zipfile.ZipFile(file) as zip_ref:
                        zip_ref.extractall(tmpdir)
                        for root, _, files in os.walk(tmpdir):
                            for fname in files:
                                path = os.path.join(root, fname)
                                if fname.lower().endswith(('jpg', 'jpeg', 'png')):
                                    with Image.open(path) as img:
                                        images.append((fname, img.convert("RGB")))
                                elif fname.lower().endswith('mp4'):
                                    with open(path, 'rb') as f:
                                        videos.append((fname, f.read()))
            elif file.name.lower().endswith(('jpg', 'jpeg', 'png')):
                with Image.open(file) as img:
                    images.append((file.name, img.convert("RGB")))
            elif file.name.lower().endswith('mp4'):
                videos.append((file.name, file.read()))
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")

    return images, videos

def hash_image(img: Image.Image, mask_roi: Optional[np.ndarray] = None) -> str:
    """Generate hash unik dengan optimasi resize dan error handling"""
    try:
        base_img = img.resize((256, 256))
        img_bytes = base_img.tobytes()

        if mask_roi is not None:
            mask_resized = cv2.resize(mask_roi.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
            img_bytes += mask_resized.tobytes()

        return hashlib.md5(img_bytes).hexdigest()
    except Exception as e:
        logger.error(f"Hash generation error: {str(e)}")
        return ""

def hash_video(video_path: str) -> str:
    """Hash video dari frame awal, akhir, dan durasi"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Gagal membuka video untuk hashing: {video_path}")
            return ""

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0

        frames = []
        for frame_id in [0, total_frames - 1]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (256, 256))
                frames.append(frame.tobytes())

        cap.release()
        combined = b''.join(frames) + f"{total_frames}_{fps}_{duration:.2f}".encode()
        return hashlib.md5(combined).hexdigest(), duration
    except Exception as e:
        logger.error(f"Hash video gagal: {str(e)}")
        return ""

def is_duplicate_hash(hash_value: str, csv_path: str = "temps/history/riwayat.csv") -> bool:
    """Cek duplikat dengan chunksize untuk file besar"""
    try:
        if not os.path.exists(csv_path):
            return False

        for chunk in pd.read_csv(csv_path, chunksize=1000, usecols=['hash']):
            if hash_value in chunk['hash'].values:
                return True
        return False
    except Exception as e:
        logger.error(f"Duplicate check error: {str(e)}")
        return False

def append_to_history(result: dict, csv_path: str = "temps/history/riwayat.csv"):
    """Menyimpan hasil analisis ke CSV riwayat dan menimpa hasil lama jika hash sama"""
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        def safe_path(value):
            return value if value and os.path.exists(str(value)) else ""

        new_row = {
            "timestamp": result.get("timestamp", ""),
            "nama_file": result.get("name", ""),
            "path_original": safe_path(result.get("original_image_path") or result.get("original_video_path")),
            "path_overlay": safe_path(result.get("overlay_image_path") or result.get("overlay_video_path")),
            "path_mask": safe_path(result.get("mask_image_path") or result.get("mask_video_path")),
            "path_roi": safe_path(result.get("path_roi", "")),
            "path_preview": safe_path(result.get("path_preview", "")),
            "coverage": round(result.get("coverage", 0.0), 2),
            "oktaf": int(result.get("oktaf", 0)),
            "kondisi_langit": result.get("kondisi_langit", ""),
            "jenis_awan": result.get("jenis_awan", ""),
            "top_preds": '; '.join([f"{l} ({c:.4f})" for l, c in result.get("top_preds", [])]),
            "durasi": round(result.get("durasi", 0.0), 2),
            "hash": result.get("hash", "")
        }

        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df[df["hash"] != result.get("hash", "")]
        else:
            df = pd.DataFrame()

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(csv_path, index=False, encoding='utf-8')

    except Exception as e:
        logger.error(f"Gagal menyimpan riwayat: {str(e)}")

def load_result_from_history(hash_value: str, csv_path: str = "temps/history/riwayat.csv") -> Optional[dict]:
    """Memuat hasil sebelumnya berdasarkan hash"""
    try:
        if not os.path.exists(csv_path):
            return None

        for chunk in pd.read_csv(csv_path, chunksize=1000):
            if 'hash' not in chunk.columns:
                continue

            match = chunk[chunk['hash'] == hash_value]
            if not match.empty:
                row = match.iloc[0].to_dict()
                return {
                    "timestamp": row.get("timestamp", ""),
                    "name": row.get("nama_file", ""),
                    "original_path": [p for p in str(row.get("path_original", "")).split(';') if p],
                    "overlay_path": [p for p in str(row.get("path_overlay", "")).split(';') if p],
                    "mask_path": [p for p in str(row.get("path_mask", "")).split(';') if p],
                    "path_roi": row.get("path_roi", ""),
                    "path_preview": row.get("path_preview", ""),
                    "coverage": float(row.get("coverage", 0.0)),
                    "oktaf": int(row.get("oktaf", 0)),
                    "kondisi_langit": row.get("kondisi_langit", ""),
                    "jenis_awan": row.get("jenis_awan", ""),
                    "top_preds": [
                        (p.split(" (")[0], float(p.split("(")[1].replace(")", "")))
                        for p in str(row.get("top_preds", "")).split(';') if '(' in p and ')' in p
                    ],
                    "durasi": float(row.get("durasi", 0.0)),
                    "hash": row.get("hash", "")
                }
        return None
    except Exception as e:
        logger.error(f"Gagal memuat riwayat: {str(e)}")
        return None