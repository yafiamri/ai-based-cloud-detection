# pages/history.py
import streamlit as st
import pandas as pd
import os
import shutil
from PIL import Image

# Impor komponen UI dari file layout.py
from utils.layout import apply_global_styles, render_page_header, render_sidebar_footer, section_divider
from utils.download import download_controller

# --- 1. Konfigurasi Halaman & Gaya ---
st.set_page_config(page_title="Riwayat Analisis", layout="wide")
apply_global_styles()
render_sidebar_footer()

# --- 2. Header Halaman ---
render_page_header("Riwayat Analisis")

# --- 3. Cek Keberadaan Data ---
csv_path = "temps/history/riwayat.csv"
if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
    st.info("Belum ada riwayat analisis yang tersimpan.")
    st.stop()

df = pd.read_csv(csv_path).reset_index(drop=True)
if df.empty:
    st.warning("Riwayat analisis masih kosong.")
    st.stop()

# --- Fungsi Bantu untuk Menampilkan Media ---
def render_media_in_table(path):
    try:
        if not pd.isna(path) and os.path.exists(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.mp4', '.mov', '.avi', '.mpeg4']:
                st.video(path)
            else:
                st.image(path, use_container_width=True)
        else:
            # Tidak menampilkan warning di sini agar UI lebih bersih
            pass
    except Exception:
        st.warning("File rusak")

# --- Inisialisasi State dan Kontrol Filter ---
column_label_map = {
    "Waktu Analisis": "timestamp", "Tutupan Awan": "coverage", "Nilai Oktaf": "oktaf",
    "Kondisi Langit": "kondisi_langit", "Jenis Awan": "jenis_awan", "Prediksi Teratas": "top_preds"
}

if "selected_rows" not in st.session_state: st.session_state["selected_rows"] = set()
if "confirm_delete" not in st.session_state: st.session_state["confirm_delete"] = False

with st.expander("🔎 Filter & Sortir Data"):
    filtered_df = df.copy()
    for label, col in column_label_map.items():
        if col in filtered_df.columns and not pd.api.types.is_numeric_dtype(filtered_df[col]):
            options = sorted(filtered_df[col].dropna().unique())
            selected = st.multiselect(f"{label}:", options, key=f"filter_{col}")
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]
    
    sort_label = st.selectbox("Urutkan Berdasarkan:", list(column_label_map.keys()), key="sort_by")
    sort_col = column_label_map[sort_label]
    sort_asc = st.radio("Urutan:", ["Naik", "Turun"], horizontal=True, key="sort_dir") == "Naik"
    if sort_col in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc).reset_index(drop=True)

# --- Paginasi ---
total_rows = len(filtered_df)
top_col1, top_col2, top_col3 = st.columns([0.7, 0.15, 0.15])
with top_col1:
    section_divider("Data Riwayat", "🗂️")
per_page_options = [10, 20, 50, 100, -1]
per_page_labels = {10: "10", 20: "20", 50: "50", 100: "100", -1: "Semua"}
with top_col2:
    selected_option = st.selectbox("Item per halaman:", options=per_page_options, format_func=lambda x: per_page_labels[x], key="perpage")
per_page = total_rows if selected_option == -1 else selected_option
total_pages = 1 if selected_option == -1 else max(1, (total_rows + per_page - 1) // per_page)
with top_col3:
    current_page = st.number_input("Halaman ke:", min_value=1, max_value=total_pages, value=1, key="curpage")

start = (current_page - 1) * per_page
end = start + per_page
paginated_df = filtered_df.iloc[start:end]

# --- Logika Pemilihan dan Tampilan Tabel ---
def update_selection(row_index, is_selected):
    if is_selected: st.session_state.selected_rows.add(row_index)
    else: st.session_state.selected_rows.discard(row_index)

def toggle_select_all_page():
    current_page_indices = set(paginated_df.index)
    if st.session_state.select_all_key:
        st.session_state.selected_rows.update(current_page_indices)
    else:
        st.session_state.selected_rows.difference_update(current_page_indices)

header_cols = st.columns([0.10, 0.15, 0.11, 0.11, 0.13, 0.10, 0.15, 0.15])
with header_cols[0]:
    is_all_selected = set(paginated_df.index).issubset(st.session_state.selected_rows) if not paginated_df.empty else False
    st.checkbox("**Pilih**", value=is_all_selected, key="select_all_key", on_change=toggle_select_all_page)

headers = ["Waktu Analisis", "Original", "Overlay", "Tutupan Awan", "Nilai Oktaf", "Kondisi Langit", "Jenis Awan"]
for col, title in zip(header_cols[1:], headers):
    col.markdown(f"<div style='text-align: center; font-weight: bold'>{title}</div>", unsafe_allow_html=True)

for idx, row in paginated_df.iterrows():
    cols = st.columns([0.10, 0.15, 0.11, 0.11, 0.13, 0.10, 0.15, 0.15])
    is_selected = idx in st.session_state.selected_rows
    cols[0].checkbox("", value=is_selected, key=f"cb_{idx}", on_change=update_selection, args=(idx, not is_selected))
    cols[1].markdown(f"<div style='text-align: center'>{row['timestamp']}</div>", unsafe_allow_html=True)
    with cols[2]: render_media_in_table(row["original_path"])
    with cols[3]: render_media_in_table(row["overlay_path"])
    cols[4].markdown(f"<div style='text-align: center'>{row['coverage']:.2f}%</div>", unsafe_allow_html=True)
    cols[5].markdown(f"<div style='text-align: center'>{row['oktaf']} oktaf</div>", unsafe_allow_html=True)
    cols[6].markdown(f"<div style='text-align: center'>{row['kondisi_langit']}</div>", unsafe_allow_html=True)
    cols[7].markdown(f"<div style='text-align: center'>{row['jenis_awan']}</div>", unsafe_allow_html=True)

# --- Panel Aksi ---
section_divider("Aksi untuk Data Terpilih", "⚙️")

valid_selected_indices = [idx for idx in st.session_state["selected_rows"] if idx in df.index]
num_selected = len(valid_selected_indices)
has_selection = num_selected > 0

st.markdown(f"📌 **{num_selected} data terpilih**")

if has_selection:
    subset = df.loc[valid_selected_indices]
    report_data = subset.to_dict('records')
    
    col_unduh, col_hapus = st.columns([0.7, 0.3])
    with col_unduh:
        st.markdown("##### 📥 Unduh Hasil")
        download_controller(report_data, context="history")
    with col_hapus:
        st.markdown("##### 🗑️ Hapus Hasil")
        if st.button("Hapus Data Terpilih", type="primary", use_container_width=True):
            st.session_state["confirm_delete"] = True
        
        if st.session_state.get("confirm_delete"):
            st.warning("Yakin ingin menghapus?")
            c1, c2 = st.columns(2)
            if c1.button("✅ Ya, Hapus", use_container_width=True):
                for _, row in subset.iterrows():
                    # PERBAIKAN: Logika penghapusan file dan folder
                    for path_key in ["original_path", "mask_path", "overlay_path"]:
                        path = row.get(path_key)
                        if pd.notna(path) and os.path.exists(path):
                            try:
                                if os.path.isdir(path):
                                    shutil.rmtree(path) # Gunakan ini untuk menghapus folder
                                else:
                                    os.remove(path) # Gunakan ini untuk menghapus file
                            except Exception as e:
                                st.warning(f"Gagal menghapus {path}: {e}")
                
                df.drop(index=valid_selected_indices, inplace=True)
                df.to_csv(csv_path, index=False)
                
                st.session_state.selected_rows.clear()
                st.session_state.confirm_delete = False
                st.success("Data berhasil dihapus.")
                st.rerun()
                
            if c2.button("❌ Batal", use_container_width=True):
                st.session_state.confirm_delete = False
                st.rerun()
else:
    st.info("Pilih satu atau lebih data pada tabel di atas untuk mengunduh atau menghapusnya.")