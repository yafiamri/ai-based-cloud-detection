# pages/history.py
import streamlit as st
import pandas as pd
import os
from PIL import Image

from utils.export import export_csv, export_pdf, export_zip
from utils.download import download_controller
from utils.layout import apply_global_styles, render_page_header, render_sidebar_footer

st.set_page_config(page_title="Riwayat Analisis", layout="wide")
apply_global_styles()
render_page_header("📖 Riwayat Analisis Awan")

# Cek keberadaan data
csv_path = "temps/history/riwayat.csv"
if not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0:
    st.error("Belum ada hasil analisis yang tersimpan.")
    st.stop()

df = pd.read_csv(csv_path)
if df.empty:
    st.warning("Riwayat analisis masih kosong.")
    st.stop()

# Tandai jenis file
def jenis_file(nama):
    return "Video" if str(nama).lower().endswith(".mp4") else "Gambar"
df["jenis_data"] = df["nama_file"].apply(jenis_file)

@st.cache_resource
def load_image(path):
    return Image.open(path)

column_label_map = {
    "Waktu Analisis": "timestamp",
    "Tutupan Awan": "coverage",
    "Nilai Oktaf": "oktaf",
    "Kondisi Langit": "kondisi_langit",
    "Jenis Awan": "jenis_awan",
    "Prediksi Teratas": "top_preds"
}

# Inisialisasi session_state jika belum ada
if "selected_rows" not in st.session_state:
    st.session_state["selected_rows"] = set()
if "select_all" not in st.session_state:
    st.session_state["select_all"] = False
if "confirm_delete" not in st.session_state:
    st.session_state["confirm_delete"] = False

def render_filter_controls(df, column_label_map):
    filtered_df = df.copy()
    filters = {}
    filter_applied = False  # Deteksi apakah ada filter aktif

    with st.expander("🔎 Filter & Sortir Data"):
        # Filter jenis file (Gambar / Video)
        jenis_opsi = ["Semua", "Gambar", "Video"]
        jenis_terpilih = st.selectbox("Jenis data:", jenis_opsi, key="filter_jenis_data")
        if jenis_terpilih != "Semua":
            filtered_df = filtered_df[filtered_df["jenis_data"] == jenis_terpilih]
            filter_applied = True

        # # Search nama file
        # search_query = st.text_input("Cari nama gambar/video:", key="filter_nama")
        # if search_query.strip():
        #     filtered_df = filtered_df[filtered_df["nama_gambar"].str.contains(search_query.strip(), case=False, na=False)]
        #     filter_applied = True
        
        # Filter kolom lainnya
        for label, col in column_label_map.items():
            options = sorted(filtered_df[col].unique())
            default = st.session_state.get(f"filter_{col}", [])
            selected = st.multiselect(f"{label}:", options, default, key=f"filter_{col}")
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]
                filters[col] = selected
                filter_applied = True

        # Sortir
        sort_label = st.selectbox("Urutkan Berdasarkan:", list(column_label_map.keys()), key="sort_by")
        sort_col = column_label_map[sort_label]
        sort_asc = st.radio("Urutan:", ["Naik", "Turun"], horizontal=True, key="sort_dir") == "Naik"
        filtered_df = filtered_df.sort_values(by=sort_col, ascending=sort_asc)

        # Cek apakah sortir diset dari default
        if (
            st.session_state.get("sort_by") != list(column_label_map.keys())[0]
            or st.session_state.get("sort_dir") != "Naik"
        ):
            filter_applied = True

        # Tombol reset muncul jika ada filter/sort aktif
        if filter_applied:
            if st.button("🔄 Reset Filter", key="reset_filters", help="Hapus semua filter dan urutan."):
                for label, col in column_label_map.items():
                    key = f"filter_{col}"
                    if key in st.session_state:
                        del st.session_state[key]
                for key in ["sort_by", "sort_dir"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

    return filtered_df

filtered_df = render_filter_controls(df, column_label_map)

# Hitung jumlah halaman
total_rows = len(filtered_df)
per_page = st.session_state.get("perpage", -1)
total_pages = max(1, (total_rows + per_page - 1) // per_page)
current_page = st.session_state.get("curpage", 1)

# Tampilkan kontrol pagination di kanan atas
top_col1, top_col2, top_col3 = st.columns([0.75, 0.125, 0.125])

# Gunakan -1 untuk mewakili pilihan "Semua"
per_page_options = [5, 10, 20, 50, -1]
per_page_labels = {5: "5", 10: "10", 20: "20", 50: "50", -1: "Semua"}

with top_col2:
    selected_option = st.selectbox(
        "Jumlah per halaman:",
        options=per_page_options,
        format_func=lambda x: per_page_labels[x],  # tampilkan label stringnya
        key="perpage"
    )

if selected_option == -1:
    per_page = total_rows
    total_pages = 1
    current_page = 1
else:
    per_page = selected_option  # sudah int
    total_pages = max(1, (total_rows + per_page - 1) // per_page)

with top_col3:
    current_page = st.number_input(
        "Halaman ke:",
        min_value=1,
        max_value=total_pages,
        step=1,
        value=1,
        key="curpage"
    )

# Subset data sesuai halaman
start = (current_page - 1) * per_page
end = start + per_page
paginated_df = filtered_df.iloc[start:end]

# Header tabel + Select All
header_cols = st.columns([0.10, 0.15, 0.11, 0.11, 0.13, 0.10, 0.15, 0.15])
headers = ["Pilih Semua", "Waktu Analisis", "Original", "Overlay", "Tutupan Awan", "Nilai Oktaf", "Kondisi Langit", "Jenis Awan"]

# Trigger toggle semua (gunakan on_change)
def toggle_select_all():
    current_index = set(paginated_df.index)
    check_all = current_index.issubset(st.session_state["selected_rows"])
    for i in current_index:
        st.session_state["selected_rows"].discard(i) if check_all else st.session_state["selected_rows"].add(i)
        st.session_state[f"row_check_{i}"] = not check_all

def update_selection(idx):
    if st.session_state.get(f"row_check_{idx}", False):
        st.session_state["selected_rows"].add(idx)
    else:
        st.session_state["selected_rows"].discard(idx)

with header_cols[0]:
    current_page_indices = set(paginated_df.index)
    semua_dicentang = current_page_indices.issubset(st.session_state["selected_rows"])
    label_pilih = "❎ Hapus Centang" if semua_dicentang else "✅ Pilih Semua"
    st.markdown(f"**{label_pilih}**")
    st.checkbox("", value=semua_dicentang, key="select_all", on_change=toggle_select_all)

for col, title in zip(header_cols[1:], headers[1:]):
    col.markdown(f"<div style='text-align: center; font-weight: bold'>{title}</div>", unsafe_allow_html=True)

# Baris tabel dengan kontrol per checkbox
new_selected = set()
for i, row in paginated_df.iterrows():
    cols = st.columns([0.10, 0.15, 0.11, 0.11, 0.13, 0.10, 0.15, 0.15])
    with cols[0]:
        def make_callback(idx=i):
            return lambda: update_selection(idx)
        st.checkbox(
            "",
            value=st.session_state.get(f"row_check_{i}", False),
            key=f"row_check_{i}",
            on_change=make_callback(i)
        )
    if st.session_state.get(f"row_check_{i}", False):
        new_selected.add(i)

    with cols[1]:
        st.markdown(f"<div style='text-align: center'>{row['timestamp']}</div>", unsafe_allow_html=True)
    with cols[2]:
        try:
            path_ori = row["original_path"]
            if str(path_ori).lower().endswith(".mp4"):
                st.video(path_ori)
            else:
                st.image(path_ori, width=200)
        except:
            st.warning("Original tidak ditemukan")
    with cols[3]:
        try:
            path_ovl = row["overlay_path"]
            if str(path_ovl).lower().endswith(".mp4"):
                st.video(path_ovl)
            else:
                st.image(path_ovl, width=200)
        except:
            st.warning("Overlay tidak ditemukan")
    with cols[4]:
        st.markdown(f"<div style='text-align: center'>{row['coverage']}%</div>", unsafe_allow_html=True)
    with cols[5]:
        st.markdown(f"<div style='text-align: center'>{row['oktaf']} oktaf</div>", unsafe_allow_html=True)
    with cols[6]:
        st.markdown(f"<div style='text-align: center'>{row['kondisi_langit']}</div>", unsafe_allow_html=True)
    with cols[7]:
        st.markdown(f"<div style='text-align: center'>{row['jenis_awan']}</div>", unsafe_allow_html=True)

# Simpan hasil pilihan
st.session_state["selected_rows"] = new_selected

# Tombol aksi (kanan-kiri)
if st.session_state["selected_rows"]:
    subset = df.loc[list(st.session_state["selected_rows"])].copy()
    report_data = [{
        "timestamp": row["timestamp"],
        "name": row["nama_file"],
        "original_path": row["original_path"],
        "mask_path": row["mask_path"],
        "overlay_path": row["overlay_path"],
        "coverage": row["coverage"],
        "oktaf": row["oktaf"],
        "jenis_awan": row["jenis_awan"],
        "kondisi_langit": row["kondisi_langit"],
        "top_preds": row["top_preds"] if "top_preds" in row else "",
        "hash": row["hash"] if "hash" in row else ""
    } for _, row in subset.iterrows()]

    st.markdown("---")
    st.markdown(f"📌 **{len(st.session_state['selected_rows'])} data terpilih**")

    col_unduh, col_hapus = st.columns([0.7, 0.3])
    with col_unduh:
        st.markdown("### 📥 Unduh Hasil Data Terpilih")
        download_controller(report_data, context="history")

    with col_hapus:
        if st.button("🗑️ Hapus Data Terpilih", type="primary", use_container_width=True):
            st.session_state["confirm_delete"] = True

        if st.session_state.get("confirm_delete", False):
            st.warning("Apakah kamu yakin ingin menghapus semua data yang dipilih?")
            l, col1, col2, r = st.columns([0.2, 0.4, 0.3, 0.1])

            with col1:
                if st.button("✅ Ya, Hapus", key="confirm_hapus"):
                    subset = df.loc[list(st.session_state["selected_rows"])].copy()
                    for _, row in subset.iterrows():
                        all_paths = (
                            str(row.get("original_path", "")).split(";") +
                            str(row.get("overlay_path", "")).split(";") +
                            str(row.get("mask_path", "")).split(";")
                        )
                        for path in all_paths:
                            if path and os.path.exists(path):
                                try:
                                    os.remove(path)
                                except:
                                    pass
                    df = df.drop(index=list(st.session_state["selected_rows"])).reset_index(drop=True)
                    df.to_csv(csv_path, index=False)
                    st.session_state["selected_rows"] = set()
                    st.session_state["confirm_delete"] = False
                    st.rerun()

            with col2:
                if st.button("❌ Batal", key="cancel_hapus"):
                    st.session_state["confirm_delete"] = False
                    st.rerun()

render_sidebar_footer()