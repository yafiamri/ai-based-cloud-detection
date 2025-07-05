# Struktur Modular - File: README.md
# 🌤️ Deteksi Awan Berbasis AI (Streamlit App)

Aplikasi ini merupakan purwarupa sistem pendeteksian awan berbasis Artificial Intelligence menggunakan model:
- **CloudDeepLabV3+** untuk segmentasi tutupan awan
- **YOLOv8-cls** untuk klasifikasi jenis awan

## 🚀 Fitur
- Upload gambar atau ZIP berisi citra langit
- Pilihan ROI otomatis/manual (lingkaran, kotak, poligon)
- Visualisasi segmentasi awan dan overlay hasil
- Perhitungan tutupan awan dalam persen dan oktaf
- Klasifikasi jenis awan dominan
- Ekspor hasil analisis ke **CSV dan PDF**

## 🧱 Struktur Folder
```
📁 pages/
📁 utils/
📁 assets/
📁 temps/
app.py
requirements.txt
README.md
```

## ▶️ Cara Menjalankan
Jalankan aplikasi dengan:
```bash
streamlit run app.py
```

Atau deploy langsung di [streamlit.io](https://streamlit.io/cloud).

## 🔗 Download Model
Jika model terlalu besar untuk GitHub, gunakan `gdown` untuk mengunduh dari Google Drive:
```python
import gdown

# Contoh (ganti dengan ID file kamu)
url = "https://drive.google.com/uc?id=FILE_ID"
out = "models/clouddeeplabv3.pth"
gdown.download(url, out, quiet=False)
```

## 🧠 Model Referensi
- `CloudDeepLabV3+`: Li et al. (2023)
- `YOLOv8-cls`: Luo et al. (2024)