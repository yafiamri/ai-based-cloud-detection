# pages/3_Panduan_Pengguna.py
import streamlit as st
from utils.layout import (
    apply_global_styles, 
    render_page_header, 
    render_sidebar_footer, 
    section_divider
)
from utils.config import config

# --- 1. Konfigurasi Halaman & Gaya ---
st.set_page_config(page_title=f"Panduan Pengguna - {config['app']['title']}", layout="wide")
apply_global_styles()
render_sidebar_footer()

# --- 2. Konten Halaman ---

# Header Utama
render_page_header("Panduan Pengguna")

# --- Seksi 1: Selamat Datang ---
section_divider("Selamat Datang di Aplikasi Deteksi Awan!", "👋")
st.markdown("""
Aplikasi ini adalah sebuah sistem purwarupa canggih yang dirancang untuk **mendeteksi tutupan dan mengklasifikasikan jenis awan** secara otomatis dari berbagai sumber citra menggunakan teknologi kecerdasan buatan (*Artificial Intelligence*). Dibangun dengan arsitektur *deep learning* modern, aplikasi ini bertujuan untuk menyediakan alat bantu yang efisien dan interaktif bagi para peneliti, mahasiswa, atau siapa pun yang tertarik dalam analisis citra langit.

Baik Anda menganalisis satu gambar, sebuah video *timelapse*, atau bahkan memantau siaran langsung, sistem ini dirancang untuk memberikan hasil yang detail dan kuantitatif dengan antarmuka yang mudah digunakan.
""")

# --- Seksi 2: Fitur Utama ---
section_divider("Fitur Unggulan Kami", "✨")
col1, col2 = st.columns(2)
with col1:
    st.info("**Analisis Multi-Sumber Fleksibel**", icon="📤")
    st.markdown("""
    - **Gambar**: Unggah satu atau banyak gambar (.jpg, .png) sekaligus.
    - **Video**: Analisis video (.mp4, .mov) dengan interval frame yang dapat disesuaikan.
    - **Arsip ZIP**: Cukup unggah satu file .zip, dan sistem akan mengekstrak semua media di dalamnya.
    - **URL**: Ambil media langsung dari tautan web, termasuk Google Drive dan video YouTube.
    """)

    st.info("**Monitoring Real-Time**", icon="📡")
    st.markdown("""
    - Pantau tutupan dan jenis awan secara langsung dari siaran *live stream* (misalnya dari YouTube Live).
    - Atur interval analisis untuk mendapatkan pembaruan secara berkala.
    """)

with col2:
    st.info("**Kontrol Analisis Penuh**", icon="🎨")
    st.markdown("""
    - **Mode Otomatis**: Biarkan AI secara cerdas mendeteksi area langit berbentuk lingkaran, cocok untuk citra *fisheye*.
    - **Mode Manual (ROI)**: Gambar **Region of Interest (ROI)** Anda sendiri (kotak, poligon, atau lingkaran) untuk fokus pada area spesifik dan mengecualikan objek penghalang seperti gedung atau pohon.
    """)

    st.info("**Manajemen Riwayat & Laporan**", icon="📚")
    st.markdown("""
    - **Riwayat Persisten**: Semua hasil analisis Anda disimpan dan dapat diakses kembali di halaman Riwayat.
    - **Filter & Sortir**: Kelola data Anda dengan mudah menggunakan fitur filter dan sortir.
    - **Ekspor Data**: Unduh hasil analisis Anda sebagai laporan **PDF** yang rapi, data mentah **CSV**, atau kumpulan semua file artefak dalam format **ZIP**.
    """)

# --- Seksi 3: Cara Menggunakan Aplikasi ---
section_divider("Cara Menggunakan Aplikasi", "🚀")

st.subheader("1. Halaman Deteksi Awan (Untuk Gambar & Video)")
st.markdown("""
Halaman ini adalah pusat untuk menganalisis berkas media statis. Alurnya dirancang seperti *wizard* yang memandu Anda langkah demi langkah.
""")

st.markdown("""
- **Langkah 1: Unggah Citra Langit**: Pilih gambar demo, unggah berkas dari komputer Anda, atau tempelkan URL. Semua berkas yang siap diproses akan muncul di area pratinjau.
- **Langkah 2: Konfigurasi Analisis**:
    - Pilih mode **"Otomatis"** untuk analisis cepat.
    - Pilih mode **"Manual per Berkas"** untuk kontrol presisi. Di sini Anda bisa menggambar ROI di atas setiap gambar dan mengatur interval analisis untuk video.
- **Langkah 3: Jalankan Analisis**: Tekan tombol **"Proses Semua Berkas"**. Sebuah bilah kemajuan (*progress bar*) akan muncul dan memberikan informasi *real-time* tentang proses analisis, termasuk kemajuan per *frame* untuk video.
- **Langkah 4: Lihat dan Unduh Hasil**: Setelah selesai, hasil akan ditampilkan secara terstruktur. Gunakan panel unduhan di bagian bawah untuk mengekspor laporan Anda.
""")

st.subheader("2. Halaman Live Monitoring")
st.markdown("""
Gunakan halaman ini untuk analisis berkelanjutan dari siaran langsung.
- **Langkah 1: Masukkan URL**: Tempelkan URL siaran langsung (misalnya, dari YouTube Live) dan konfirmasi. Sebuah pratinjau akan muncul jika URL valid.
- **Langkah 2: Konfigurasi**: Atur **interval** analisis (dalam detik) dan pilih **metode ROI** (otomatis atau manual).
- **Langkah 3: Jalankan Monitoring**: Tekan tombol **"Mulai Monitoring"**. Sebuah pesan info akan muncul menandakan proses sedang berjalan, dan hasil analisis akan diperbarui secara otomatis sesuai interval yang Anda tentukan. Tekan **"Hentikan Monitoring"** untuk mengakhiri sesi.
- **Langkah 4: Rangkuman Sesi**: Setelah dihentikan, sebuah dasbor rangkuman dari sesi monitoring tersebut akan ditampilkan, dan Anda dapat mengunduh hasilnya.
""")

st.subheader("3. Halaman Riwayat Analisis")
st.markdown("""
Halaman ini adalah pusat manajemen data Anda.
- **Filter & Pilih**: Gunakan kontrol di bagian atas untuk memfilter dan menyortir data. Centang kotak di setiap baris untuk memilih entri yang ingin Anda kelola.
- **Aksi Massal**: Setelah memilih satu atau lebih entri, panel **"Aksi untuk Data Terpilih"** akan aktif. Dari sini, Anda bisa mengunduh laporan gabungan atau menghapus entri dan semua file terkait secara permanen (dengan konfirmasi).
""")

# --- Seksi 4: Di Balik Layar ---
section_divider("Di Balik Layar: Teknologi yang Digunakan", "🔬")
st.markdown("""
Kekuatan aplikasi ini didukung oleh dua model *deep learning* canggih yang bekerja secara sinergis, diatur oleh arsitektur aplikasi web yang modern.
""")
col_tech1, col_tech2 = st.columns(2)
with col_tech1:
    st.success("🤖 Model Segmentasi: CloudDeepLabV3+", icon="🗺️")
    st.markdown("""
    Model ini bertugas untuk **mengukur tutupan awan**. Ia bekerja dengan menganalisis setiap piksel pada citra dan mengklasifikasikannya sebagai "awan" atau "langit".
    - **Arsitektur**: Menggunakan fondasi DeepLabV3+ dengan *backbone* **EfficientNetV2-S** yang ringan namun kuat.
    - **Fitur Kunci**: Dilengkapi dengan modul **ASPP** (*Atrous Spatial Pyramid Pooling*) untuk menangkap konteks awan dalam berbagai ukuran dan **A-FAM** (*Attention-based Feature Alignment Module*) untuk menghasilkan batas awan yang lebih tajam.
    - **Hasil**: Sebuah peta segmentasi presisi tinggi yang digunakan untuk menghitung persentase tutupan awan dan nilai Okta.
    """)
with col_tech2:
    st.success("🤖 Model Klasifikasi: YOLOv8", icon="🏷️")
    st.markdown("""
    Setelah area awan diketahui, model ini bertugas untuk **mengidentifikasi jenis awan** yang dominan.
    - **Arsitektur**: Menggunakan varian **YOLOv8x-cls**, yang terkenal dengan keseimbangan antara kecepatan dan akurasi.
    - **Fitur Kunci**: Memanfaatkan *backbone* **C2f** untuk ekstraksi fitur yang kaya dan *neck* **PAN-FPN** untuk menggabungkan informasi dari berbagai skala, memungkinkannya mengenali berbagai jenis awan.
    - **Hasil**: Prediksi jenis awan yang paling mungkin (seperti *Cumulus*, *Cirrus*, dll.) beserta tingkat keyakinannya.
    """)

st.info("**Arsitektur Aplikasi Streamlit**", icon="🕸️")
st.markdown("""
Aplikasi ini dibangun di atas kerangka kerja **Streamlit** dengan arsitektur **client-server**. Antarmuka pengguna (yang Anda lihat di *browser*) terpisah dari logika pemrosesan AI yang berjalan di *server*. Ini memungkinkan aplikasi untuk berjalan secara efisien baik di komputer lokal maupun saat di-*deploy* di *cloud*.
""")

# --- Footer ---
st.markdown("---")
st.markdown("Terima kasih telah menggunakan aplikasi ini! Jelajahi fiturnya dan semoga bermanfaat. 🚀")