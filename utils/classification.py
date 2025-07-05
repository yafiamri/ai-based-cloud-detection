# utils/classification.py
import os
from ultralytics import YOLO

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
    drive_id = "1qG1nvsCBPxOPtiE2Po8yDS521SjfZisI"
    if not os.path.exists(weight_path):
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, weight_path, quiet=False)
    return YOLO(weight_path)

def predict_classification(model, image_path, top_k=3):
    result = model.predict(image_path, verbose=False)[0]
    probs = result.probs.data.tolist()
    top_preds = sorted(zip(class_names, probs), key=lambda x: x[1], reverse=True)
    filtered_preds = [(label, conf) for label, conf in top_preds if conf > 0.05][:top_k]
    return filtered_preds

def format_predictions(preds):
    return "\n".join([
        f"- **{label}** ({conf*100:.1f}%)" if i == 0 else f"- {label} ({conf*100:.1f}%)"
        for i, (label, conf) in enumerate(preds)
    ])