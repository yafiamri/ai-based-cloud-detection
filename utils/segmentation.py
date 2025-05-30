# utils/segmentation.py
import torch
import numpy as np
import cv2
from models.clouddeeplabv3_architecture import CloudDeepLabV3Plus

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_segmentation_model(weight_path="models/clouddeeplabv3.pth"):
    """Memuat model segmentasi dengan manajemen device"""
    model = CloudDeepLabV3Plus()
    try:
        model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Gagal memuat model segmentasi: {str(e)}")

def prepare_input_tensor(image_np: np.ndarray):
    """Mempersiapkan input tensor dengan normalisasi"""
    if image_np.max() > 1.0:
        image_np = image_np / 255.0
    img_resized = cv2.resize(image_np, (512, 512))
    return torch.from_numpy(img_resized.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

def predict_segmentation(model, input_tensor, threshold=0.5):
    """Prediksi segmentasi dengan thresholding"""
    with torch.no_grad():
        pred = model(input_tensor)["out"].squeeze().cpu().numpy()
    return (pred > threshold).astype(np.uint8)