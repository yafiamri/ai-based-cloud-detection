# utils/segmentation.py
import os
import streamlit as st
import gdown
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
import cv2
from typing import Tuple, Dict, Any, List

# Impor konfigurasi terpusat
from .config import config

# Ambil konfigurasi spesifik untuk model ini agar kode lebih bersih
SEG_MODEL_CONFIG = config.get('models', {}).get('segmentation', {})
ANALYSIS_CONFIG = config.get('analysis', {})

# --- Arsitektur Model (Hanya penambahan dokumentasi & integrasi config) ---

def gn(num_channels: int) -> nn.GroupNorm:
    """Helper function untuk membuat layer Group Normalization."""
    return nn.GroupNorm(num_groups=32, num_channels=num_channels)

class HRFSASPP(nn.Module):
    """Implementasi modul HRFS-ASPP."""
    def __init__(self, in_channels: int, out_channels: int):
        super(HRFSASPP, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.branch3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.branch4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1),
            gn(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size(2), x.size(3)
        out1, out2, out3, out4 = self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)
        out5 = F.interpolate(self.global_pool(x), size=(h, w), mode='bilinear', align_corners=False)
        residual = self.residual(x)
        concat = torch.cat([out1, out2, out3, out4, out5], dim=1)
        return self.fusion(concat) + residual

class AFAM(nn.Module):
    """Implementasi modul A-FAM."""
    def __init__(self, low_channels: int, high_channels: int, out_channels: int):
        super(AFAM, self).__init__()
        self.conv_low = nn.Conv2d(low_channels, out_channels, kernel_size=1)
        self.conv_high = nn.Conv2d(high_channels, out_channels, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1), gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            gn(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, low_feat: torch.Tensor, high_feat: torch.Tensor) -> torch.Tensor:
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=True)
        low, high = self.conv_low(low_feat), self.conv_high(high_feat)
        attn = self.attention(torch.cat([low, high], dim=1))
        fused = attn * low + (1 - attn) * high
        return self.fusion(fused)

class CloudDeepLabV3Plus(nn.Module):
    """Arsitektur utama model segmentasi CloudDeepLabV3+, menggunakan backbone EfficientNetV2."""
    def __init__(self):
        super(CloudDeepLabV3Plus, self).__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s", features_only=True, pretrained=True)
        self.hrfs_aspp = HRFSASPP(in_channels=256, out_channels=256)
        self.afam = AFAM(low_channels=24, high_channels=256, out_channels=256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1), gn(128),
            nn.ReLU(inplace=True), nn.Dropout(p=0.1), nn.Conv2d(128, 1, kernel_size=1)
        )
        output_size = tuple(SEG_MODEL_CONFIG.get('input_size', [512, 512]))
        self.final_upsample = lambda x: F.interpolate(x, size=output_size, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        low_feat, high_feat = features[0], features[4]
        x = self.hrfs_aspp(high_feat)
        x = self.afam(low_feat, x)
        x = self.decoder(x)
        x = self.final_upsample(x)
        return {"out": x}

# --- Fungsi-Fungsi Utilitas ---

@st.cache_resource
def load_segmentation_model() -> CloudDeepLabV3Plus:
    """
    Memuat model segmentasi CloudDeepLabV3+.
    Fungsi ini di-cache untuk performa, akan mengunduh bobot jika tidak ditemukan.

    Returns:
        CloudDeepLabV3Plus: Objek model PyTorch yang sudah dimuat dan siap digunakan.
    """
    weight_path = SEG_MODEL_CONFIG.get('weight_path')
    drive_id = SEG_MODEL_CONFIG.get('drive_id')

    if not os.path.exists(weight_path):
        os.makedirs(os.path.dirname(weight_path), exist_ok=True)
        url = f"https://drive.google.com/uc?id={drive_id}"
        st.toast(f"Mengunduh bobot model segmentasi...")
        gdown.download(url, weight_path, quiet=False)
        
    model = CloudDeepLabV3Plus()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

def prepare_input_tensor(image_np: np.ndarray) -> torch.Tensor:
    """
    Mengubah gambar numpy array menjadi tensor PyTorch yang siap untuk model.

    Args:
        image_np (np.ndarray): Gambar input (H, W, C), dalam rentang float 0-1.

    Returns:
        torch.Tensor: Tensor PyTorch dengan shape (1, C, H, W).
    """
    input_size = tuple(SEG_MODEL_CONFIG.get('input_size', [512, 512]))
    img_uint8 = (image_np * 255).astype(np.uint8)
    img_resized = cv2.resize(img_uint8, input_size) / 255.0
    return torch.from_numpy(img_resized.transpose(2, 0, 1)).unsqueeze(0).float()

def predict_segmentation(model: CloudDeepLabV3Plus, input_tensor: torch.Tensor) -> np.ndarray:
    """
    Melakukan prediksi segmentasi pada satu tensor gambar.

    Args:
        model (CloudDeepLabV3Plus): Model segmentasi yang sudah dimuat.
        input_tensor (torch.Tensor): Tensor input dari `prepare_input_tensor`.

    Returns:
        np.ndarray: Mask biner (0 atau 1) hasil segmentasi.
    """
    threshold = ANALYSIS_CONFIG.get('segmentation_threshold', 0.5)
    with torch.no_grad():
        pred = model(input_tensor)["out"].squeeze().cpu().numpy()
    return (pred > threshold).astype(np.uint8)

def detect_circle_roi(image_np: np.ndarray) -> np.ndarray:
    """
    Mendeteksi ROI melingkar dari gambar (misalnya, dari lensa fisheye).

    Args:
        image_np (np.ndarray): Gambar input (H, W, C), dalam rentang float 0-1.

    Returns:
        np.ndarray: Mask biner (0 atau 1) dengan area lingkaran berwarna putih.
    """
    img_uint8 = (image_np * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blur, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.ones((h, w), dtype=np.uint8)
        
    largest_contour = max(contours, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(radius), 1, -1)
    return mask

def canvas_to_mask(canvas_result: Any, height: int, width: int) -> np.ndarray:
    """
    Mengonversi hasil dari streamlit-drawable-canvas menjadi mask numpy.

    Args:
        canvas_result (Any): Objek hasil dari `st_canvas`.
        height (int): Tinggi gambar asli.
        width (int): Lebar gambar asli.

    Returns:
        np.ndarray: Mask biner (0 atau 1) berdasarkan gambar pengguna.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if not (canvas_result and canvas_result.json_data and canvas_result.json_data.get("objects")):
        return mask

    canvas_h, canvas_w = canvas_result.image_data.shape[:2]
    scale_x, scale_y = width / canvas_w, height / canvas_h
    
    for obj in canvas_result.json_data["objects"]:
        obj_type = obj.get("type")
        if obj_type == "rect":
            l, t = int(obj["left"] * scale_x), int(obj["top"] * scale_y)
            w, h = int(obj["width"] * scale_x), int(obj["height"] * scale_y)
            mask[t:t+h, l:l+w] = 1
        elif obj_type == "path" and obj.get("path"):
            coords = [[int(item[1] * scale_x), int(item[2] * scale_y)] 
                      for item in obj["path"] if isinstance(item, list) and len(item) >= 3]
            if len(coords) >= 3:
                cv2.fillPoly(mask, [np.array(coords, dtype=np.int32)], 1)
        elif obj_type == "line":
            left, top = float(obj.get("left", 0)), float(obj.get("top", 0))
            x1, y1 = int((left + obj["x1"]) * scale_x), int((top + obj["y1"]) * scale_y)
            x2, y2 = int((left + obj["x2"]) * scale_x), int((top + obj["y2"]) * scale_y)
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            radius = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2)
            if 0 <= cx < width and 0 <= cy < height and radius > 0:
                cv2.circle(mask, (cx, cy), radius, 1, -1)
    return mask