# pages/3_Penjelasan_Model.py
import streamlit as st

from utils.layout import apply_global_styles, render_page_header, render_sidebar_footer, section_divider

# --- 1. Konfigurasi Halaman & Gaya ---
st.set_page_config(page_title="Penjelasan Model", layout="wide")
apply_global_styles()
render_sidebar_footer()

# --- 2. Render Halaman ---
render_page_header("Penjelasan Model Deteksi Awan")

st.markdown("""
Sistem ini menggunakan dua model *deep learning* utama untuk menganalisis citra langit berbasis darat:

1.  **CloudDeepLabV3+** — untuk segmentasi tutupan awan.
2.  **YOLOv8** — untuk klasifikasi jenis awan.

Kedua model ini diintegrasikan dalam satu sistem untuk memberikan hasil analisis yang objektif, cepat, dan akurat.
""")

# --- Detail Model ---
section_divider("CloudDeepLabV3+ — Segmentasi Tutupan Awan", "☁️")
st.markdown("""
CloudDeepLabV3+ adalah model segmentasi citra yang dirancang untuk memisahkan piksel-piksel awan dari latar belakang langit.

**Komponen utama:**
- **EfficientNetV2-S** sebagai *backbone* untuk ekstraksi fitur yang efisien.
- **ASPP (Atrous Spatial Pyramid Pooling)** untuk menangkap fitur multi-skala, memungkinkan model melihat objek dalam berbagai ukuran.
- **Decoder** untuk merekonstruksi citra dan menghasilkan *mask* segmentasi awan yang presisi.

Model ini bekerja dengan memberikan *output mask* biner: setiap piksel diberi label **awan (1)** atau **bukan awan (0)**.
""")

section_divider("YOLOv8 — Klasifikasi Jenis Awan", "🌤️")
st.markdown("""
YOLOv8 adalah model deteksi dan klasifikasi *real-time* yang telah dilatih secara khusus untuk mengenali jenis-jenis awan dari citra.

**Jenis awan yang dapat diklasifikasi:**
- Cumulus
- Cirrus / Cirrostratus
- Stratocumulus / Stratus / Altostratus
- Cumulonimbus / Nimbostratus
- Altocumulus / Cirrocumulus
- Mixed Cloud
- Clear Sky

Model akan memberikan prediksi beberapa jenis awan teratas beserta *confidence score* (tingkat kepercayaan) untuk setiap prediksi.
""")

# --- Alur Kerja ---
section_divider("Alur Kerja Sistem", "🔁")
st.markdown("""
1.  **Input:** Gambar atau video langit diunggah atau dipilih dari demo.
2.  **ROI:** *Region of Interest* (area amatan) ditentukan secara otomatis atau digambar manual oleh pengguna.
3.  **Segmentasi:** CloudDeepLabV3+ memisahkan area awan dari langit di dalam ROI yang telah ditentukan.
4.  **Klasifikasi:** YOLOv8 mengidentifikasi jenis awan dari gambar asli.
5.  **Output:** Visualisasi tutupan awan (% dan oktaf), jenis awan, dan laporan yang dapat diekspor ke berbagai format (PDF, CSV, ZIP).
""")

# --- Referensi ---
section_divider("Referensi Ilmiah", "📖")
st.markdown("""
- Li, S., Wang, M., Wu, J., Sun, S., & Zhuang, Z. (2023). *CloudDeepLabV3+: A lightweight ground-based cloud segmentation method*. International Journal of Remote Sensing, 44(15), 4836–4856. https://doi.org/10.1080/01431161.2023.2240034
- Luo, J., Pan, Y., Su, D., Zhong, J., Wu, L., Zhao, W., ... & Wang, Y. (2024). *Innovative cloud quantification: Deep learning classification and finite-sector clustering for ground-based all-sky imaging*. Atmospheric Measurement Techniques, 17(12), 3765–3781. https://doi.org/10.5194/amt-17-3765-2024
- Lv, Q., Li, Q., Chen, K., Lu, Y., & Wang, L. (2022). *Classification of ground-based cloud images by contrastive self-supervised learning*. Remote Sensing, 14(22), 5821. https://doi.org/10.3390/rs14225821
- Hu, F., Hou, B., Zhu, W., Zhu, Y., & Zhang, Q. (2023). *Cloudy-net: A deep CNN for joint segmentation and classification of cloud images*. Atmosphere, 14(9), 1405. https://doi.org/10.3390/atmos14091405
- Zhu, W., Chen, T., Hou, B., et al. (2022). *Classification of ground-based cloud images by improved combined convolutional network*. Applied Sciences, 12(3), 1570. https://doi.org/10.3390/app12031570
- Kazantzidis, A., Tzoumanikas, P., et al. (2012). *Cloud detection and classification with whole-sky images*. Atmospheric Research, 113, 80–88. https://doi.org/10.1016/j.atmosres.2012.05.005
- Niu, Y., Song, J., Zou, L., et al. (2024). *Cloud detection using sky images based on Clear Sky Library and superpixel local threshold*. Renewable Energy, 226, 120452. https://doi.org/10.1016/j.renene.2024.120452
""")