# AI-Based Cloud Detection (ABCD) Web App

![A screenshot of the AI-Based Cloud Detection application interface, showing a dark mode theme with a beautiful semi-transparent cartoon cloud background. The main panel displays the results for a cloud image, including the original photo, a segmentation overlay, and a set of metrics like Cloud Coverage and Dominant Cloud Type.](assets/display.png)
Aplikasi web interaktif yang dibangun dengan Streamlit untuk mendeteksi tutupan awan (segmentasi) dan menentukan jenis awan (klasifikasi) secara otomatis dari berbagai sumber citra menggunakan model *deep learning*.

---

## 📜 Tentang Proyek

Proyek ini merupakan implementasi dari sistem deteksi awan berbasis AI yang menyajikan kemampuan analisis meteorologi tingkat lanjut dalam sebuah antarmuka yang ramah pengguna. Aplikasi ini mampu memproses gambar statis, video *timelapse*, hingga siaran langsung (*live stream*), menjadikannya alat yang fleksibel untuk penelitian, pendidikan, atau hobi.

Tujuan utamanya adalah untuk menjembatani kesenjangan antara model AI yang kompleks dengan kebutuhan analisis praktis, memungkinkan pengguna untuk mendapatkan data kuantitatif dan kualitatif tentang kondisi langit dengan mudah.

## ✨ Fitur Utama

-   **Analisis Multi-Sumber**:
    -   🖼️ **Gambar**: Unggah satu atau banyak gambar (.jpg, .png, .webp).
    -   📹 **Video**: Analisis video (.mp4, .mov) dengan interval *frame* yang dapat disesuaikan.
    -   🔗 **URL**: Ambil media langsung dari tautan web, termasuk video YouTube dan file Google Drive.
    -   🗂️ **Arsip ZIP**: Unggah file `.zip` berisi kumpulan gambar dan video untuk diproses sekaligus.
-   **🛰️ Monitoring Real-Time**:
    -   Analisis kondisi awan secara langsung dari siaran *live stream* YouTube dan platform lainnya.
    -   Interval pemantauan yang dapat dikonfigurasi.
-   **🎨 Kontrol Analisis Penuh (Region of Interest - ROI)**:
    -   **Mode Otomatis**: Deteksi cerdas area langit berbentuk lingkaran, ideal untuk citra *fisheye*.
    -   **Mode Manual**: Gambar ROI (kotak, poligon, lingkaran) Anda sendiri untuk fokus pada area spesifik dan mengabaikan objek penghalang.
-   **📊 Hasil Analisis Detail**:
    -   **Metrik Kuantitatif**: Dapatkan persentase tutupan awan, nilai Okta, dan kondisi langit.
    -   **Hasil Klasifikasi**: Identifikasi jenis awan dominan beserta tingkat keyakinan untuk setiap kelas.
    -   **Visualisasi Overlay**: Lihat perbandingan berdampingan antara citra asli dan hasil segmentasi.
-   **📚 Manajemen Riwayat**:
    -   Semua hasil analisis disimpan secara persisten dalam database.
    -   Filter, sortir, dan kelola data riwayat Anda dengan mudah melalui tabel interaktif.
-   **📥 Ekspor Laporan Profesional**:
    -   Unduh hasil analisis dalam format **PDF** yang rapi, data mentah **CSV**, atau kumpulan semua file (asli, masker, overlay) dalam format **ZIP**.

## 🛠️ Tumpukan Teknologi (Tech Stack)

-   **Framework Aplikasi**: Streamlit
-   **Backend & Pemrosesan**: Python
-   **Model AI**:
    -   **Segmentasi**: PyTorch (implementasi modifikasi dari CloudDeepLabV3+)
    -   **Klasifikasi**: Ultralytics YOLOv8
-   **Pemrosesan Citra/Video**: OpenCV, Pillow
-   **Analisis Data & Visualisasi**: Pandas, Plotly Express
-   **Manajemen Data**: SQLite

## 🏗️ Arsitektur Sistem

Sistem ini menggunakan arsitektur *client-server* yang difasilitasi oleh Streamlit, memisahkan antarmuka pengguna (frontend) dari logika pemrosesan AI (backend).

1.  **Model Segmentasi (CloudDeepLabV3+ Modifikasi)**: Model ini bertugas untuk mengklasifikasikan setiap piksel dalam citra sebagai "awan" atau "langit". Ini menghasilkan peta segmentasi yang digunakan untuk menghitung tutupan awan secara presisi. Arsitektur telah disederhanakan secara pragmatis dari literatur aslinya untuk mencapai keseimbangan antara akurasi dan efisiensi komputasi.
2.  **Model Klasifikasi (YOLOv8x-cls)**: Setelah area awan diketahui, model ini mengidentifikasi jenis awan dominan dalam *Region of Interest* (ROI). Ini memastikan bahwa klasifikasi hanya dilakukan pada area langit yang relevan.
3.  **Alur Kerja Analisis**: Untuk setiap citra, sistem pertama-tama menentukan ROI (otomatis atau manual). Kemudian, model segmentasi dan klasifikasi dijalankan. Hasil kuantitatif dan visual kemudian digabungkan dan disimpan ke database.

## 🚀 Memulai (Getting Started)

Untuk menjalankan aplikasi ini di lingkungan lokal Anda, ikuti langkah-langkah berikut.

### Prasyarat

-   Python 3.9+
-   Git
-   Conda (Direkomendasikan) atau `venv`

### Instalasi

1.  **Clone repositori ini:**
    ```bash
    git clone https://github.com/yafiamri/ai-based-cloud-detection.git
    cd ai-based-cloud-detection
    ```

2.  **Buat dan aktifkan environment Conda:**
    ```bash
    conda create -n abcd python=3.10
    conda activate abcd
    ```

3.  **Instal semua dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Unduh Model AI:**
    Aplikasi ini memerlukan bobot model yang telah dilatih sebelumnya. Letakkan file bobot di dalam direktori `models/`:
    -   `models/yolov8.pt`
    -   `models/clouddeeplabv3.pth`

5.  **Siapkan Aset:**
    Pastikan folder `assets/` berisi file yang diperlukan seperti `background.png`, `logo.png`, dan gambar-gambar demo.

### Menjalankan Aplikasi

Setelah instalasi selesai, jalankan aplikasi menggunakan perintah berikut dari direktori utama proyek:

```bash
streamlit run Beranda.py