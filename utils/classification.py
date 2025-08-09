# utils/classification.py
import os
import streamlit as st
import gdown
from ultralytics import YOLO
from typing import List, Tuple

# Impor konfigurasi terpusat
from .config import config

# Ambil konfigurasi spesifik untuk model ini agar kode lebih bersih
MODEL_CONFIG = config.get('models', {}).get('classification', {})
ANALYSIS_CONFIG = config.get('analysis', {})


@st.cache_resource
def load_classification_model() -> YOLO:
    """
    Memuat model klasifikasi YOLOv8.
    Fungsi ini di-cache oleh Streamlit untuk memastikan model hanya dimuat sekali.
    Akan mengunduh bobot model dari Google Drive jika tidak ditemukan secara lokal.

    Returns:
        YOLO: Objek model YOLO yang sudah dimuat dan siap digunakan.
    """
    weight_path = MODEL_CONFIG.get('weight_path')
    drive_id = MODEL_CONFIG.get('drive_id')

    if not os.path.exists(weight_path):
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={drive_id}"
        st.toast(f"Mengunduh bobot model klasifikasi...")
        gdown.download(url, weight_path, quiet=False)
    
    return YOLO(weight_path)


def predict_classification(model: YOLO, image_path: str) -> List[Tuple[str, float]]:
    """
    Melakukan prediksi klasifikasi pada satu gambar dan mengembalikan hasil teratas.

    Args:
        model (YOLO): Model YOLO yang sudah dimuat.
        image_path (str): Path ke gambar yang akan diprediksi.

    Returns:
        List[Tuple[str, float]]: Daftar tuple berisi (nama_kelas, confidence_score).
    """
    class_names = MODEL_CONFIG.get('class_names', [])
    top_k = ANALYSIS_CONFIG.get('top_k_preds', 3)
    conf_threshold = ANALYSIS_CONFIG.get('confidence_threshold', 0.05)

    result = model.predict(image_path, verbose=False)[0]
    probs = result.probs.data.tolist()
    
    preds_with_names = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
    
    filtered_preds = [
        (label, conf) for label, conf in preds_with_names if conf > conf_threshold
    ][:top_k]
    
    return filtered_preds


def format_predictions(preds: List[Tuple[str, float]]) -> str:
    """
    Memformat hasil prediksi menjadi string yang mudah dibaca untuk ditampilkan di UI.

    Args:
        preds (List[Tuple[str, float]]): Hasil dari `predict_classification`.

    Returns:
        str: String yang sudah diformat untuk ditampilkan.
    """
    if not preds:
        return "Tidak Terdeteksi"
        
    return "\n".join([
        f"- **{label}** ({conf*100:.1f}%)" if i == 0 else f"- {label} ({conf*100:.1f}%)"
        for i, (label, conf) in enumerate(preds)
    ])