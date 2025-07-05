# utils/image.py
from PIL import Image
import os, zipfile, tempfile, io
import numpy as np
from datetime import datetime
import cv2

from .segmentation import prepare_input_tensor, predict_segmentation, detect_circle_roi
from .classification import predict_classification

def extract_media_from_zip(zip_file):
    """
    Ekstrak semua berkas gambar dan video dari berkas ZIP,
    termasuk dari subfolder. Mengembalikan daftar file-like objects (BytesIO).
    """
    media_files = []
    # Gunakan BytesIO agar tidak perlu menulis ke disk
    with zipfile.ZipFile(io.BytesIO(zip_file.getvalue()), 'r') as zip_ref:
        for member in zip_ref.infolist():
            if member.is_dir():
                continue
            
            file_name = os.path.basename(member.filename)
            if not file_name: continue

            supported_extensions = ('.jpeg', '.jpg', '.png', '.webp', '.avi', '.mov', '.mp4', '.mpeg4')
            if file_name.lower().endswith(supported_extensions):
                file_data = zip_ref.read(member.filename)
                file_buffer = io.BytesIO(file_data)
                file_buffer.name = file_name
                media_files.append(file_buffer)
    return media_files

def load_uploaded_images(uploaded_files):
    """Membaca gambar dari berkas unggahan (non-ZIP)."""
    images = []
    for file in uploaded_files:
        if not file.name.lower().endswith(".zip"):
             try:
                # Mundurkan pointer file ke awal setelah dibaca
                file.seek(0)
                images.append((file.name, Image.open(file).convert("RGB")))
             except Exception:
                 continue
    return images

def load_demo_images(demo_dir="assets/demo"):
    """Memuat gambar contoh dari folder demo_dir."""
    return [
        (f, Image.open(os.path.join(demo_dir, f)).convert("RGB"))
        for f in os.listdir(demo_dir)
        if f.lower().endswith(("jpeg", "jpg", "png", "webp"))
    ]

def create_image_overlay(original_image: Image.Image, mask: np.ndarray, color: tuple = (255, 0, 0), opacity: int = 128) -> Image.Image:
    """Membuat gambar overlay dengan menempelkan warna transparan di atas gambar asli."""
    overlay_sticker = Image.new("RGBA", original_image.size)
    overlay_pixels = np.array(overlay_sticker)
    cloud_locations = mask > 0
    overlay_pixels[cloud_locations] = [*color, opacity]
    overlay_sticker = Image.fromarray(overlay_pixels, "RGBA")
    
    final_image = original_image.convert("RGBA")
    final_image.paste(overlay_sticker, (0, 0), overlay_sticker)
    
    return final_image.convert("RGB")

def analyze_image(img, name, mask_user, seg_model, cls_model, progress_bar, step_counter, total_steps):
    """Fungsi utama untuk menganalisis satu berkas gambar."""
    step_counter += 1
    progress_bar.progress(step_counter / total_steps, text=f"Menganalisis berkas: {name}")
    
    np_img = np.array(img) / 255.0
    mask_roi = detect_circle_roi(np_img) if mask_user is None else mask_user
    
    tensor = prepare_input_tensor(np_img)
    pred_seg = predict_segmentation(seg_model, tensor)
    pred_seg = cv2.resize(pred_seg.astype(np.uint8), (img.width, img.height), interpolation=cv2.INTER_NEAREST)
    
    final_mask = pred_seg * mask_roi
    
    awan, total = final_mask.sum(), mask_roi.sum()
    coverage = 100 * awan / total if total > 0 else 0
    oktaf = int(round((coverage / 100) * 8))
    kondisi = ["Cerah", "Sebagian Cerah", "Sebagian Berawan", "Berawan", "Hampir Tertutup", "Tertutup"][min(oktaf // 2, 5)]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(name)[0].replace(" ", "_")
    output_dir = "temps/history"
    os.makedirs(f"{output_dir}/images", exist_ok=True)
    os.makedirs(f"{output_dir}/masks", exist_ok=True)
    os.makedirs(f"{output_dir}/overlays", exist_ok=True)
    
    img_path = f"{output_dir}/images/{timestamp}_{base_name}.png"
    mask_path = f"{output_dir}/masks/{timestamp}_{base_name}.png"
    overlay_path = f"{output_dir}/overlays/{timestamp}_{base_name}.png"
    
    img.save(img_path)
    Image.fromarray(final_mask * 255).save(mask_path)
    
    overlay_image = create_image_overlay(img, final_mask)
    overlay_image.save(overlay_path)
    
    preds = predict_classification(cls_model, img_path)
    jenis = preds[0][0] if preds else "Tidak Terdeteksi"
    
    result = {
        "name": name, "original_path": img_path, "mask_path": mask_path, 
        "overlay_path": overlay_path, "coverage": coverage, "oktaf": oktaf,
        "kondisi_langit": kondisi, "jenis_awan": jenis, "top_preds": preds
    }
    return {"result": result, "step_counter": step_counter}