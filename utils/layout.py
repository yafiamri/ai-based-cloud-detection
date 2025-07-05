# Lokasi: utils/layout.py
import streamlit as st
import os
import base64
from PIL import Image

# Menentukan path ke folder assets dengan benar
ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')

def apply_global_styles():
    """
    Menerapkan gaya CSS global yang komprehensif.
    """
    st.markdown("""
    <style>
        /* Variabel Warna Utama */
        :root {
            --primary: #1f77b4;
            --secondary: #2ca02c;
            --card-bg: #f8f9fa;
            --card-border: #dee2e6;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --primary: #6fa8dc;
                --secondary: #93c47d;
                --card-bg: #161b22;
                --card-border: #30363d;
            }
        }
        
        /* Gaya Kartu Navigasi */
        .nav-card {
            display: flex; flex-direction: column; justify-content: space-between;
            height: 100%; padding: 1.5rem 1.2rem; border-radius: 10px;
            background-color: var(--card-bg); border: 1px solid var(--card-border);
            border-top: 4px solid var(--primary); transition: transform 0.2s, box-shadow 0.2s;
            text-align: center;
        }
        .nav-card:hover { transform: translateY(-5px); box-shadow: 0 8px 30px rgba(0,0,0,0.1); }
        
        /* Gaya Tombol Navigasi */
        .nav-card a.nav-button-link {
            display: block; text-align: center; font-weight: bold;
            padding: 0.7rem; margin-top: 1rem; background-color: var(--primary);
            color: white !important; border-radius: 5px; text-decoration: none;
            transition: background-color 0.2s;
        }
        .nav-card a.nav-button-link:hover { background-color: var(--secondary); }

        /* Gaya Tabel Hasil */
        table {
            width: 100%;
            border-collapse: collapse;
            table-layout: fixed; /* PERBAIKAN: Ini adalah kuncinya */
        }
        td, th {
            padding: 8px;
            border: 1px solid #ddd;
            text-align: left;
            word-wrap: break-word; /* Memastikan teks panjang tidak merusak layout */
        }
        th {
            background-color: #f2f2f2;
        }
        
        /* Gaya untuk caption rata tengah */
        .centered-caption {
            text-align: center;
            color: grey;
            font-size: 0.9em;
        }
    </style>
    """, unsafe_allow_html=True)

def render_page_header(title: str, logo_name: str = "logo_transparent.png"):
    """
    Menampilkan header halaman yang rapi dengan logo dan judul.
    """
    logo_path = os.path.join(ASSETS_DIR, logo_name)
    try:
        with open(logo_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            logo_src = f"data:image/png;base64,{b64}"
            st.markdown(f"""
            <div style="display:flex; align-items:center; margin-bottom:1.5rem;">
              <img src="{logo_src}" width="60" style="margin-right:15px;"/>
              <h1 style="margin:0; color:var(--primary); font-size: 2.5rem; font-family: 'Segoe UI', sans-serif;">{title}</h1>
            </div>
            """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.header(title)

def render_sidebar_footer():
    """
    Menampilkan footer statis dengan background yang menempel di bagian bawah sidebar.
    """
    st.sidebar.markdown("""
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            width: inherit;
            border-top: 1px solid var(--card-border);
            padding: 1rem;
            text-align: center;
        }
    </style>
    <div class="footer">
      <p style="font-size:0.75rem; color:var(--primary);">
      🧑‍💻 Dikembangkan oleh <b>Yafi Amri</b><br>
      Mahasiswa Meteorologi ITB 2021
      </p>
    </div>
    """, unsafe_allow_html=True)

def section_divider(text: str, emoji: str = "📌"):
    """
    Membuat pembatas antar-section yang stylish.
    """
    st.markdown(f"""
    <div style="display: flex; align-items: center; margin-top: 1.5rem; margin-bottom: 1.5rem;">
        <h2 style="color:var(--primary); margin:0; font-size: 1.75rem; font-family: 'Segoe UI', sans-serif;">{emoji} {text}</h2>
        <hr style="
            flex-grow: 1;
            margin-left: 1rem;
            border: none;
            height: 2px;
            background: linear-gradient(90deg, var(--primary), transparent);
        ">
    </div>
    """, unsafe_allow_html=True)

def render_result(r):
    """
    Menampilkan hasil analisis untuk gambar atau video.
    """
    st.subheader(f"Hasil Analisis: {r.get('nama_file', r.get('name', 'N/A'))}")

    is_video = os.path.splitext(r['original_path'])[1].lower() in ['avi', '.mov', '.mp4', '.mpeg4']

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if os.path.exists(r["original_path"]):
            if is_video: st.video(r["original_path"])
            else: st.image(r["original_path"], use_container_width=True)
            st.markdown('<p class="centered-caption">Citra Asli</p>', unsafe_allow_html=True)
        else: st.warning("❗ File asli tidak ditemukan.")
            
    with col_g2:
        if os.path.exists(r["overlay_path"]):
            if is_video: st.video(r["overlay_path"])
            else: st.image(r["overlay_path"], use_container_width=True)
            st.markdown('<p class="centered-caption"><i>Overlay</i> Segmentasi</p>', unsafe_allow_html=True)
        else: st.warning("❗ File overlay tidak ditemukan.")

    with st.container():
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("#### 🖼️ Metrik Hasil Segmentasi")
            st.markdown(f"""
            <table>
                <thead><tr><th>Parameter</th><th>Nilai</th></tr></thead>
                <tbody>
                    <tr><td>Tutupan Awan (%)</td><td>{r['coverage']:.2f}%</td></tr>
                    <tr><td>Tutupan Awan (oktaf)</td><td>{r['oktaf']} oktaf</td></tr>
                    <tr><td>Kondisi Langit</td><td>{r['kondisi_langit']}</td></tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)
    
        with col_t2:
            st.markdown("#### 🌥️ Jenis Awan Terdeteksi")
            table_md = "<table><thead><tr><th>Klasifikasi Awan</th><th>Kepercayaan</th></tr></thead><tbody>"
            top_preds = r.get("top_preds")
            if top_preds:
                if isinstance(top_preds, str):
                    try:
                        preds = [item.strip().split(' (') for item in top_preds.split(';')]
                        top_preds = [(label, float(prob.replace('%)',''))/100) for label, prob in preds if '%)' in prob]
                    except: top_preds = []
                for label, score in top_preds:
                    table_md += f"<tr><td>{label}</td><td>{score * 100:.2f}%</td></tr>"
            else:
                table_md += "<tr><td>Tidak ada</td><td>-</td></tr>"
            table_md += "</tbody></table>"
            st.markdown(table_md, unsafe_allow_html=True)
    
    st.markdown("---")