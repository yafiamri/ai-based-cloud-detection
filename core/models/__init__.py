# Lokasi File: core/models/__init__.py
"""
File inisialisasi untuk paket core.models.

File ini berfungsi untuk:
1. Menandai direktori ini sebagai paket Python.
2. Menyediakan akses mudah ke fungsi-fungsi utama dari modul di dalamnya,
   sehingga impor menjadi lebih bersih.
"""

from .clouddeeplabv3 import get_model as get_segmentation_model
from .yolov8 import get_model as get_classification_model

# Mendefinisikan apa yang akan diimpor saat seseorang melakukan 'from core.models import *'
# Ini adalah praktik yang baik untuk kejelasan.
__all__ = [
    "get_segmentation_model",
    "get_classification_model"
]