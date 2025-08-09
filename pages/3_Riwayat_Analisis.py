# pages/3_Riwayat_Analisis.py
import streamlit as st
import pandas as pd
import os
import shutil

# Impor semua fondasi dari utils
from utils.config import config
from utils.database import get_history_df, delete_history_entries, get_paths_for_deletion
from utils.layout import apply_global_styles, render_page_header, render_sidebar_footer, section_divider, render_summary_dashboard
from utils.media import get_preview_as_base64
from utils.download import download_controller
    
# --- 1. Konfigurasi Halaman & Inisialisasi ---
st.set_page_config(page_title=f"Riwayat Analisis - {config['app']['title']}", layout="wide")
apply_global_styles()
render_sidebar_footer()

# Inisialisasi state untuk konfirmasi hapus
if "confirm_delete" not in st.session_state:
    st.session_state.confirm_delete = False
if "toast_message" not in st.session_state:
    st.session_state.toast_message = None
if st.session_state.toast_message:
    message, icon = st.session_state.toast_message
    st.toast(message, icon=icon)
    st.session_state.toast_message = None # Hapus setelah ditampilkan

# --- 2. Pemuatan Data & Tampilan Utama ---
render_page_header("Riwayat Analisis")
st.write("Kelola, lihat detail, dan ekspor semua hasil analisis yang telah tersimpan di dalam sistem.")

# Ambil data langsung dari database terpusat
df_history = get_history_df()

section_divider("Tabel Data Riwayat", "🗂️")

if df_history.empty:
    st.info("Belum ada riwayat analisis yang tersimpan. Silakan lakukan analisis di halaman 'Deteksi Awan' atau 'Live Monitoring'.")
    st.stop()

# --- 3. Filter & Kontrol Tabel ---
with st.expander("🔎 Filter & Sortir Data", expanded=False):
    # Kolom untuk filter
    filter_cols = {
        "Tipe Media": "media_type",
        "Kondisi Langit": "sky_condition",
        "Jenis Awan Dominan": "dominant_cloud_type"
    }
    
    # Kolom untuk sortir (semua kolom relevan)
    sortable_cols = {
        "Waktu Analisis": "analyzed_at",
        "Nama File": "source_filename",
        "Tipe Media": "media_type",
        "Tutupan Awan (%)": "cloud_coverage",
        "Nilai Okta": "okta_value"
    }

    filtered_df = df_history.copy()
    
    # Terapkan filter
    for label, col_name in filter_cols.items():
        if col_name in filtered_df.columns:
            options = sorted(filtered_df[col_name].dropna().unique())
            selected = st.multiselect(f"Filter berdasarkan {label}:", options)
            if selected:
                filtered_df = filtered_df[filtered_df[col_name].isin(selected)]

    # Terapkan sortir
    sort_col_label = st.selectbox("Urutkan berdasarkan:", list(sortable_cols.keys()))
    sort_ascending = st.radio("Urutan:", ["Terbaru (Turun)", "Terdahulu (Naik)"], horizontal=True) == "Terlama (Naik)"
    filtered_df = filtered_df.sort_values(by=sortable_cols[sort_col_label], ascending=sort_ascending)

# --- 4. Tampilan Tabel Interaktif dengan st.dataframe ---
st.info("Pilih baris pada tabel di bawah dengan mengklik kotak centang untuk melakukan aksi.")

# Buat kolom pratinjau PADA 'filtered_df' SETELAH SEMUA FILTER SELESAI
if not filtered_df.empty:
    with st.spinner("Mempersiapkan pratinjau gambar..."):
        filtered_df["preview"] = filtered_df["overlay_path"].apply(get_preview_as_base64)
else:
    # Buat kolom kosong jika hasil filter kosong untuk mencegah error
    filtered_df["preview"] = None

# Konfigurasi lengkap untuk semua kolom di tabel 'history.db'
column_config = {
    # --- Kolom ID & Hash (Biasanya untuk debugging) ---
    "id": st.column_config.NumberColumn(
        "ID",
        help="ID unik untuk setiap entri di database"
    ),
    "pipeline_version_hash": st.column_config.TextColumn(
        "Hash Pipeline",
        help="Sidik jari unik dari versi model dan kode yang digunakan untuk analisis"
    ),
    "file_hash": st.column_config.TextColumn(
        "Hash File",
        help="Sidik jari unik dari konten file media"
    ),
    "analysis_hash": st.column_config.TextColumn(
        "Hash Analisis",
        help="Sidik jari unik dari kombinasi file dan konfigurasi analisis"
    ),
    # --- Kolom Utama yang Umum Ditampilkan ---
    "preview": st.column_config.ImageColumn(
        "Pratinjau",
        help="Pratinjau visual dari gambar overlay hasil segmentasi"
    ),
    "source_filename": st.column_config.TextColumn(
        "Nama File",
        help="Nama file asli yang dianalisis"
    ),
    "media_type": st.column_config.SelectboxColumn(
        "Tipe Media",
        help="Jenis media yang diproses (Gambar, Video, atau Live Frame)",
        options=["image", "video", "live_frame"]
    ),
    "analyzed_at": st.column_config.DatetimeColumn(
        "Waktu Analisis (UTC)",
        help="Waktu saat analisis selesai dilakukan",
        format="YYYY-MM-DD HH:mm"
    ),
    # --- Kolom Hasil Analisis ---
    "cloud_coverage": st.column_config.ProgressColumn(
        "Tutupan Awan (%)",
        help="Persentase tutupan awan terhadap total area langit yang dianalisis",
        format="%.2f%%",
        min_value=0,
        max_value=100
    ),
    "okta_value": st.column_config.NumberColumn(
        "Okta",
        help="Skala tutupan awan (0-8)"
    ),
    "sky_condition": st.column_config.TextColumn(
        "Kondisi Langit",
        help="Deskripsi kondisi langit berdasarkan nilai Okta"
    ),
    "dominant_cloud_type": st.column_config.TextColumn(
        "Awan Dominan",
        help="Jenis awan yang paling dominan terdeteksi"
    ),
    "classification_details": st.column_config.TextColumn(
        "Detail Klasifikasi",
        help="Tingkat keyakinan deteksi untuk setiap jenis awan yang teridentifikasi"
    ),
    # --- Kolom Metadata Tambahan (Bisa disembunyikan) ---
    "file_size_bytes": st.column_config.NumberColumn(
        "Ukuran (Bytes)",
        help="Ukuran file asli dalam satuan bytes",
        format="%d B"
    ),
    "analysis_duration_sec": st.column_config.NumberColumn(
        "Durasi Analisis (s)",
        help="Waktu yang dibutuhkan sistem untuk menyelesaikan analisis dalam detik",
        format="%.2f detik"
    ),
    # --- Kolom Path (Sebaiknya disembunyikan dari pengguna biasa) ---
    "original_path": st.column_config.TextColumn(
        "Path Asli",
        help="Lokasi penyimpanan internal untuk file media asli"
    ),
    "mask_path": st.column_config.TextColumn(
        "Path Mask",
        help="Lokasi penyimpanan internal untuk file gambar mask"
    ),
    "overlay_path": st.column_config.TextColumn(
        "Path Overlay",
        help="Lokasi penyimpanan internal untuk file gambar overlay"
    ),
}

# Tampilkan DataFrame dengan seleksi baris
selection = st.dataframe(
    filtered_df,
    column_config=column_config,
    on_select="rerun", # Memicu rerun saat seleksi berubah
    selection_mode="multi-row",
    use_container_width=True,
    height=500,
    hide_index=False,
    column_order=[
        "preview", "source_filename", "media_type", "analyzed_at",
        "cloud_coverage", "okta_value", "sky_condition", "dominant_cloud_type"
    ]
)

# Ambil baris yang dipilih
selected_rows = selection["selection"]["rows"]
selected_ids = filtered_df.iloc[selected_rows].index.tolist()
selected_data = filtered_df.loc[selected_ids].to_dict('records')

# --- 5. Dasbor Statistik Dinamis ---
# Tentukan data mana yang akan ditampilkan di dasbor
if selected_ids:
    # Jika ada baris yang dipilih, gunakan data yang sudah diseleksi
    df_for_dashboard = filtered_df.loc[selected_ids]
    dashboard_title = f"Dasbor Statistik untuk {len(selected_ids)} Item Terpilih"
else:
    # Jika tidak ada yang dipilih, gunakan semua data yang sudah difilter
    df_for_dashboard = filtered_df
    dashboard_title = "Dasbor Statistik Riwayat Analisis"

# Panggil fungsi dasbor universal dengan data yang sesuai
render_summary_dashboard(df_for_dashboard, title=dashboard_title)

# --- 6. Panel Aksi untuk Data Terpilih ---
section_divider(f"Aksi untuk Data Terpilih ({len(selected_ids)} item)", "⚙️")

if not selected_ids:
    st.info("Pilih satu atau lebih data pada tabel di atas untuk mengunduh atau menghapusnya.")
else:
    col_unduh, col_hapus = st.columns([0.65, 0.35])
    
    with col_unduh:
        # Panggil download controller dengan data yang dipilih
        download_controller(selected_data, context="history")
        
    with col_hapus:
        # Logika penghapusan dengan konfirmasi
        if st.button("Hapus Data Terpilih", type="primary", use_container_width=True):
            st.session_state.confirm_delete = True

        if st.session_state.get("confirm_delete"):
            st.warning(f"Anda akan menghapus **{len(selected_ids)} entri** riwayat beserta semua file terkait secara permanen. Yakin ingin melanjutkan?")
            c1, c2 = st.columns(2)
            if c1.button("✅ Ya, Hapus", use_container_width=True):
                with st.spinner("Menghapus data dan file artefak..."):
                    # 1. Ambil semua path file/folder dari database
                    paths_from_db = get_paths_for_deletion(selected_ids)
                    
                    # 2. Kumpulkan semua DIREKTORI INDUK yang unik untuk dihapus
                    dirs_to_delete = set()
                    for path in paths_from_db:
                        # Path bisa berupa file atau direktori (untuk mask video)
                        # os.path.dirname() akan mengambil folder induknya
                        parent_dir = os.path.dirname(path)
                        # Pastikan kita tidak mencoba menghapus folder arsip utama
                        if parent_dir and "archive" in parent_dir:
                            dirs_to_delete.add(parent_dir)
                    
                    # 3. Hapus setiap direktori unik beserta seluruh isinya
                    for dir_path in dirs_to_delete:
                        try:
                            if os.path.exists(dir_path):
                                shutil.rmtree(dir_path)
                        except Exception as e:
                            st.warning(f"Gagal menghapus folder {os.path.basename(dir_path)}: {e}")
                    
                    # 4. Hapus entri dari database
                    delete_history_entries(selected_ids)
                    
                    st.session_state.confirm_delete = False
                    st.session_state.toast_message = (f"Berhasil menghapus {len(selected_ids)} entri.", "✅")
                    st.rerun()
            
            if c2.button("❌ Batal", use_container_width=True):
                st.session_state.confirm_delete = False
                st.rerun()