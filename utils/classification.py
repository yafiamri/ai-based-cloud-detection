# utils/classification.py
from ultralytics import YOLO
import torch

class_names = [
    'Cumulus',
    'Altocumulus / Cirrocumulus',
    'Cirrus / Cirrostratus',
    'Clear Sky',
    'Stratocumulus / Stratus / Altostratus',
    'Cumulonimbus / Nimbostratus',
    'Mixed Cloud'
]

def load_classification_model(weight_path="models/yolov8.pt"):
    """Memuat model klasifikasi dengan konfigurasi device"""
    try:
        model = YOLO(weight_path)
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return model
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model klasifikasi: {str(e)}")

def predict_classification(model, image_path, threshold=0.25):
    """Prediksi klasifikasi dengan filtering threshold"""
    results = model.predict(image_path, imgsz=512, verbose=False)
    probs = results[0].probs.data.tolist()
    return sorted(
        [(class_names[i], prob) for i, prob in enumerate(probs) if prob > threshold],
        key=lambda x: x[1],
        reverse=True
    )

def format_predictions(preds, max_display=3):
    """Format hasil prediksi untuk ditampilkan"""
    return "\n".join(
        f"{'**' if i==0 else ''}{label}: {conf*100:.1f}%{'**' if i==0 else ''}" 
        for i, (label, conf) in enumerate(preds[:max_display])
    )