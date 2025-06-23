# Lokasi: pages/2_Riwayat_Analisis.py
import streamlit as st
import pandas as pd
from ui_components import render_page_header, apply_global_styles, render_sidebar_footer, render_analysis_card, render_download_controller
from core.io import file_manager

st.set_page_config(page_title="Riwayat Analisis", page_icon="📜", layout="wide")
apply_global_styles()
render_page_header("Riwayat Analisis", "📜")

history_records = file_manager.get_history()

if not history_records:
    st.info("Belum ada riwayat analisis yang tersimpan.")
    st.stop()

history_df = pd.DataFrame(history_records)

# --- SIDEBAR: Kontrol Filter ---
with st.sidebar:
    st.header("🔍 Filter Riwayat")
    # Filter berdasarkan jenis awan
    all_clouds = sorted(history_df['dominant_cloud_type'].unique())
    selected_clouds = st.multiselect("Filter berdasarkan Jenis Awan:", all_clouds, default=all_clouds)
    
    # Filter berdasarkan tanggal
    start_date = pd.to_datetime(history_df['timestamp']).dt.date.min()
    end_date = pd.to_datetime(history_df['timestamp']).dt.date.max()
    date_range = st.date_input("Filter berdasarkan Tanggal:", value=(start_date, end_date), min_value=start_date, max_value=end_date)
    
    render_sidebar_footer()

# --- Terapkan Filter ---
filtered_df = history_df[history_df['dominant_cloud_type'].isin(selected_clouds)]
if len(date_range) == 2:
    start_filter, end_filter = pd.to_datetime(date_range[0]).date(), pd.to_datetime(date_range[1]).date()
    filtered_df = filtered_df[pd.to_datetime(filtered_df['timestamp']).dt.date.between(start_filter, end_filter)]

st.metric("Total Riwayat Tersimpan", len(history_df))
st.metric("Menampilkan Hasil", len(filtered_df))

# --- Tampilan Hasil ---
if filtered_df.empty:
    st.warning("Tidak ada riwayat yang cocok dengan filter Anda.")
else:
    st.dataframe(filtered_df.drop(columns=['original_path', 'mask_path', 'overlay_path', 'roi_path', 'preview_path', 'metrics'], errors='ignore'))
    
    with st.expander("Lihat Detail dan Hapus Entri"):
        for index, row in filtered_df.iterrows():
            with st.container(border=True):
                render_analysis_card(row.to_dict())
                if st.button("Hapus Entri Ini", key=f"del_{row['file_hash']}", type="secondary"):
                    file_manager.delete_history_entry(row['file_hash'])
                    st.success(f"Entri untuk '{row['file_name']}' telah dihapus.")
                    st.rerun()

    # Kontrol Unduhan untuk data yang difilter
    st.markdown("---")
    render_download_controller(filtered_df.to_dict('records'), "riwayat", "Yafi Amri")