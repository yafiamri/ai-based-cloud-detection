# Lokasi File: core/models/clouddeeplabv3.py
"""
Mendefinisikan arsitektur dan pemuat untuk model segmentasi CloudDeepLabV3+.

File ini bertanggung jawab HANYA untuk:
1. Mendefinisikan kelas-kelas arsitektur neural network.
2. Menyediakan fungsi untuk memuat bobot model (.pth) yang sudah dilatih.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple
from pathlib import Path
import logging

log = logging.getLogger(__name__)

# --- PERUBAHAN KUNCI: Gunakan Path.cwd() yang sudah terbukti andal ---
PROJECT_ROOT = Path.cwd()
    
# =============================================================================
# 1. DEFINISI ARSITEKTUR MODEL (SESUAI DENGAN KODE ANDA YANG BENAR)
# =============================================================================

def gn(num_channels):
    """Helper function untuk membuat layer Group Normalization."""
    return nn.GroupNorm(num_groups=32, num_channels=num_channels)

class HRFSASPP(nn.Module):
    """Modul High-Resolution Feature Squeeze Atrous Spatial Pyramid Pooling."""
    def __init__(self, in_channels, out_channels):
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
    def forward(self, x):
        h, w = x.size(2), x.size(3)
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        out5 = self.global_pool(x)
        out5 = F.interpolate(out5, size=(h, w), mode='bilinear', align_corners=False)
        residual = self.residual(x)
        concat = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.fusion(concat) + residual
        return out

class AFAM(nn.Module):
    """Modul Adjacent Feature Aggregation Module."""
    def __init__(self, low_channels, high_channels, out_channels):
        super(AFAM, self).__init__()
        self.conv_low = nn.Conv2d(low_channels, out_channels, kernel_size=1)
        self.conv_high = nn.Conv2d(high_channels, out_channels, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=1),
            gn(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            gn(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, low_feat, high_feat):
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=True)
        low = self.conv_low(low_feat)
        high = self.conv_high(high_feat)
        concat = torch.cat([low, high], dim=1)
        attn = self.attention(concat)
        fused = attn * low + (1 - attn) * high
        out = self.fusion(fused)
        return out

class CloudDeepLabV3Plus(nn.Module):
    """Arsitektur utama model CloudDeepLabV3+ (Full Version)."""
    def __init__(self):
        super(CloudDeepLabV3Plus, self).__init__()
        self.backbone = timm.create_model("tf_efficientnetv2_s", features_only=True, pretrained=True)
        self.hrfs_aspp = HRFSASPP(in_channels=256, out_channels=256)
        self.afam = AFAM(low_channels=24, high_channels=256, out_channels=256)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            gn(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        # Ambil fitur dari backbone dengan benar
        # Perhatikan perubahan pada `forward` di bawah
    def forward(self, x):
        input_size = x.shape[2:]
        features = self.backbone(x)
        # Berdasarkan arsitektur EfficientNetV2-S di `timm`,
        # fitur level rendah dan tinggi biasanya ada di indeks tertentu.
        low_feat = features[0]     # 24 channels
        high_feat = features[4]    # 256 channels
        
        x = self.hrfs_aspp(high_feat)
        x = self.afam(low_feat, x)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        return {"out": x}

# =============================================================================
# 2. FUNGSI PEMUAT MODEL (MODEL LOADER)
# =============================================================================

def get_model(weight_path: str | Path | None = None) -> tuple[Optional[CloudDeepLabV3Plus], Optional[torch.device]]:
    """
    Membuat instance model CloudDeepLabV3Plus dan memuat bobot yang sudah dilatih.

    Args:
        weight_path: Path menuju file bobot model .pth.

    Returns:
        Tuple berisi (model, device). Mengembalikan (None, None) jika gagal.
    """
    if weight_path is None:
        weight_path = PROJECT_ROOT / "models" / "clouddeeplabv3.pth"
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CloudDeepLabV3Plus()
        log.info(f"Mencoba memuat bobot model dari path: {weight_path}")
        if not Path(weight_path).is_file():
            raise FileNotFoundError
        state_dict = torch.load(weight_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        log.info("Model segmentasi CloudDeepLabV3+ berhasil dimuat.")
        return model, device
    except FileNotFoundError:
        log.error(f"FATAL: File bobot model TIDAK DITEMUKAN di '{weight_path}'.")
        return None, None
    except Exception as e:
        log.error(f"FATAL: Gagal saat memuat model segmentasi: {e}")
        return None, None