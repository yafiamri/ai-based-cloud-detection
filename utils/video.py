# utils/video.py
import streamlit as st
import os, tempfile, json, subprocess, shutil
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime
import cv2

from .segmentation import predict_segmentation, prepare_input_tensor
from .classification import predict_classification
from .image import analyze_image

def _find_executable(name: str) -> str:
    """Mencari path executable (ffmpeg.exe atau ffprobe.exe)."""
    if path := shutil.which(name): return path
    local_exe_path = os.path.join(os.getcwd(), f"{name}.exe")
    if os.path.exists(local_exe_path): return local_exe_path
    raise FileNotFoundError(f"{name}.exe tidak ditemukan. Pastikan ia ada di PATH sistem atau di direktori utama proyek.")

FFMPEG_PATH = _find_executable("ffmpeg")
FFPROBE_PATH = _find_executable("ffprobe")

def get_video_info(video_file):
    """
    Mendapatkan durasi dan frame pertama dari video.
    Menggunakan session_state untuk cache yang lebih andal.
    """
    try:
        # Jika objek memiliki atribut .size, gunakan itu.
        # Jika tidak (artinya ini adalah BytesIO), dapatkan ukurannya dari buffer.
        file_size = video_file.size if hasattr(video_file, 'size') else len(video_file.getvalue())
        file_key = f"video_info_{video_file.name}_{file_size}"
    except Exception:
        # Fallback jika terjadi error saat mendapatkan ukuran
        file_key = f"video_info_{video_file.name}"

    if file_key not in st.session_state:
        try:
            # Pastikan pointer berada di awal untuk BytesIO
            video_file.seek(0)
            video_bytes = video_file.getvalue()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(video_bytes); temp_video_path = tmp.name
            
            command = [ FFPROBE_PATH, "-v", "quiet", "-print_format", "json", "-show_streams", temp_video_path ]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            video_info = json.loads(result.stdout)['streams'][0]
            duration, width, height = float(video_info['duration']), int(video_info['width']), int(video_info['height'])

            command_frame = [ FFMPEG_PATH, "-i", temp_video_path, "-vframes", "1", "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-" ]
            frame_data = subprocess.run(command_frame, capture_output=True, check=True).stdout
            first_frame = Image.frombytes('RGB', (width, height), frame_data)
            os.remove(temp_video_path)
            st.session_state[file_key] = (duration, first_frame)
        except (subprocess.CalledProcessError, IndexError, FileNotFoundError) as e:
            st.error(f"Gagal memproses video: {e}"); st.session_state[file_key] = (0, None)
    
    return st.session_state[file_key]

def _create_transparent_mask(binary_mask_np: np.ndarray, color: tuple = (255, 0, 0), opacity: int = 128) -> Image.Image:
    """Mengubah mask biner menjadi gambar RGBA transparan."""
    h, w = binary_mask_np.shape
    transparent_image = np.zeros((h, w, 4), dtype=np.uint8)
    cloud_locations = binary_mask_np > 0
    transparent_image[cloud_locations] = [*color, opacity]
    return Image.fromarray(transparent_image, "RGBA")

def analyze_video(video_file, mask_roi_user, frame_interval, seg_model, cls_model, progress_bar, step_counter, total_steps):
    """
    Fungsi utama yang dioptimalkan:
    1. Ekstrak frame & analisis untuk mengumpulkan "stiker" overlay.
    2. Simpan setiap mask hitam-putih sebagai gambar untuk arsip.
    3. Gunakan satu perintah FFmpeg untuk komposisi video overlay akhir.
    """
    video_bytes = video_file.getvalue()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(video_bytes)
        temp_video_path = tmp.name

    cap = cv2.VideoCapture(temp_video_path)
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS) or 30.0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if mask_roi_user is None: mask_roi = np.ones((height, width), dtype=np.uint8)
    else: mask_roi = cv2.resize(mask_roi_user.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)

    temp_overlay_dir = tempfile.mkdtemp()
    
    safe_interval = max(1, frame_interval)
    frames_to_analyze_indices = [int(i * fps) for i in range(0, int(total_frames / fps), safe_interval)]
    
    analysis_results, classification_results_list = {}, []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_base_name = os.path.splitext(video_file.name)[0].replace(" ", "_")
    output_dir = "temps/history"
    mask_archive_dir = os.path.join(output_dir, "masks", f"{timestamp}_{video_base_name}")
    os.makedirs(mask_archive_dir, exist_ok=True)
    
    for i, frame_idx in enumerate(frames_to_analyze_indices):
        step_counter += 1
        progress_bar.progress(step_counter / total_steps, text=f"Menganalisis berkas: {video_file.name} (frame {i+1}/{len(frames_to_analyze_indices)})")
        cap = cv2.VideoCapture(temp_video_path); cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx); ret, frame = cap.read(); cap.release()
        if not ret: continue
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        tensor = prepare_input_tensor(np.array(img) / 255.0); pred_seg = predict_segmentation(seg_model, tensor)
        pred_seg_resized = cv2.resize(pred_seg.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        
        final_mask = pred_seg_resized * mask_roi
        
        # Buat "stiker" overlay transparan dan simpan ke folder sementara
        transparent_mask = _create_transparent_mask(final_mask)
        transparent_mask.save(os.path.join(temp_overlay_dir, f"overlay_{i:06d}.png"))
        
        # PERBAIKAN: Simpan mask hitam-putih sebagai gambar untuk arsip
        mask_image = Image.fromarray(final_mask * 255)
        mask_image.save(os.path.join(mask_archive_dir, f"mask_frame_{i:06d}.png"))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_img: tmp_img_path = tmp_img.name; img.save(tmp_img_path)
        preds_cls = predict_classification(cls_model, tmp_img_path); os.remove(tmp_img_path)
        
        coverage = 100 * final_mask.sum() / mask_roi.sum() if mask_roi.sum() > 0 else 0
        analysis_results[i] = {'coverage': coverage, 'cls_preds': preds_cls}
        if preds_cls: classification_results_list.append(preds_cls)

    avg_coverage = np.mean([res['coverage'] for res in analysis_results.values()]) if analysis_results else 0
    avg_oktaf = int(round((avg_coverage / 100) * 8))
    kondisi_langit = ["Cerah", "Sebagian Cerah", "Sebagian Berawan", "Berawan", "Hampir Tertutup", "Tertutup"][min(avg_oktaf // 2, 5)]
    if classification_results_list:
        flat_list_of_preds = [item for sublist in classification_results_list for item in sublist]
        all_preds_df = pd.DataFrame(flat_list_of_preds, columns=["awan", "skor"])
        avg_preds = all_preds_df.groupby("awan")["skor"].mean().sort_values(ascending=False)
        top_preds_avg = list(zip(avg_preds.index, avg_preds.values)); jenis_awan_avg = top_preds_avg[0][0] if top_preds_avg else "Tidak Terdeteksi"
    else: top_preds_avg, jenis_awan_avg = [], "Tidak Terdeteksi"

    progress_bar.progress(1.0, f"Menyusun video hasil akhir untuk {video_file.name}")

    os.makedirs(f"{output_dir}/images", exist_ok=True); os.makedirs(f"{output_dir}/overlays", exist_ok=True)
    original_path, overlay_path = f"{output_dir}/images/{timestamp}_{video_base_name}.mp4", f"{output_dir}/overlays/{timestamp}_{video_base_name}.mp4"
    with open(original_path, "wb") as f: f.write(video_bytes)

    framerate_for_overlay = len(frames_to_analyze_indices) / (total_frames / fps) if total_frames > 0 and fps > 0 else 1.0
    command = [
        FFMPEG_PATH, '-y', '-i', temp_video_path,
        '-framerate', str(framerate_for_overlay), '-i', os.path.join(temp_overlay_dir, 'overlay_%06d.png'),
        '-filter_complex', "[0:v][1:v]overlay=shortest=1",
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'veryfast', overlay_path
    ]
    subprocess.run(command, check=True, capture_output=True)

    shutil.rmtree(temp_overlay_dir)
    os.remove(temp_video_path)
    
    # Path mask sekarang merujuk ke folder arsip gambar
    mask_path = mask_archive_dir
    
    return {"result": {"original_path": original_path, "mask_path": str(mask_path), "overlay_path": overlay_path, "coverage": avg_coverage, "oktaf": avg_oktaf, "kondisi_langit": kondisi_langit, "jenis_awan": jenis_awan_avg, "top_preds": top_preds_avg}, "step_counter": step_counter}