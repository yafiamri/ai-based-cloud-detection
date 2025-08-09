# utils/media.py
import os
import io
import re
import shutil
import tempfile
import uuid
import requests
import gdown
import yt_dlp
import zipfile
import mimetypes
import cv2
import base64
import streamlit as st
from urllib.parse import urlparse
from typing import List, Dict, Any, Optional, IO, Tuple
from PIL import Image
from streamlink import Streamlink
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Impor konfigurasi terpusat
from .config import config

# Ambil konfigurasi yang relevan untuk efisiensi
PATHS = config.get('paths', {})
ANALYSIS_CONFIG = config.get('analysis', {})
VIDEO_EXTENSIONS = tuple(ANALYSIS_CONFIG.get('video_extensions', ['.mp4']))

# --- Fungsi Publik ---

def extract_media_from_zip(zip_file: IO[bytes]) -> Tuple[List[IO[bytes]], Optional[str]]:
    """
    Mengekstrak semua file media yang didukung dari sebuah arsip ZIP.

    Args:
        zip_file (IO[bytes]): Objek file ZIP dalam format byte.

    Returns:
        List[IO[bytes]]: Daftar objek file media dalam format BytesIO.
                         Setiap objek memiliki atribut `.name` yang sesuai.
    """
    media_files = []
    supported_ext = tuple(
        ANALYSIS_CONFIG.get('image_extensions', []) + 
        ANALYSIS_CONFIG.get('video_extensions', [])
    )
    
    try:
        with zipfile.ZipFile(io.BytesIO(zip_file.getvalue()), 'r') as zf:
            for member in zf.infolist():
                # Abaikan direktori dan file sistem internal (misal dari macOS)
                if not member.is_dir() and not member.filename.startswith('__MACOSX'):
                    if member.filename.lower().endswith(supported_ext):
                        file_data = zf.read(member.filename)
                        file_buffer = io.BytesIO(file_data)
                        file_buffer.name = os.path.basename(member.filename)
                        media_files.append(file_buffer)
    except zipfile.BadZipFile:
        return [], "File ZIP yang diunggah korup atau formatnya tidak valid."

def load_demo_files() -> List[Tuple[str, Image.Image, str]]:
    """
    Memuat gambar contoh dari direktori demo yang ditentukan di config.

    Returns:
        List[Tuple[str, Image.Image]]: Daftar tuple berisi (nama_file, objek_PIL_Image).
    """
    demo_dir = PATHS.get('demo', 'assets/demo')
    image_ext = tuple(ANALYSIS_CONFIG.get('image_extensions', []))
    video_ext = tuple(ANALYSIS_CONFIG.get('video_extensions', []))

    media_list = []
    if not os.path.isdir(demo_dir):
        return []

    for f in sorted(os.listdir(demo_dir)):
        file_path = os.path.join(demo_dir, f)
        if f.lower().endswith(image_ext):
            try:
                media_list.append((f, Image.open(file_path).convert("RGB"), "image"))
            except Exception:
                continue
        elif f.lower().endswith(video_ext):
            media_list.append((f, None, "video"))
    return media_list

def fetch_media_from_url(url: str) -> Tuple[List[IO[bytes]], Optional[str]]:
    """
    Mengunduh media dari URL. Bertindak sebagai dispatcher yang memilih
    metode yang tepat berdasarkan struktur URL dan tipe konten.

    Alur Logika:
    1. Cek apakah URL adalah dari Google Drive -> _handle_gdrive.
    2. Jika bukan, lakukan permintaan untuk mengecek Content-Type ("KTP" file).
    3. Jika Content-Type adalah media langsung (gambar/video) -> _handle_direct_media.
    4. Jika Content-Type adalah halaman web (text/html) -> _handle_video_page (yt-dlp).
    5. Jika tidak didukung, tampilkan error.

    Args:
        url (str): URL sumber media.

    Returns:
        List[IO[bytes]]: Daftar file media dalam memori, atau daftar kosong jika gagal.
    """
    # Pintu 1: Rute khusus untuk Google Drive
    if "drive.google.com" in url:
        return _handle_gdrive(url)

    # Pintu 2: Rute umum untuk semua URL lain
    try:
        # Filter awal untuk menolak URL pencarian YouTube sebelum request
        parsed_url = urlparse(url)
        if "youtube.com" in parsed_url.netloc and parsed_url.path == "/results":
            return [], "URL adalah halaman hasil pencarian, bukan video tunggal."

        if "youtube.com" in url or "youtu.be" in url:
            return _handle_video_page(url)

        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, stream=True, timeout=15, headers=headers)
        response.raise_for_status()
        content_type = response.headers.get('content-type', '').lower()

        supported_media_types = ['image/', 'video/']
        if any(media_type in content_type for media_type in supported_media_types):
            return _handle_direct_media(response)
        elif 'text/html' in content_type:
            # Panggil helper yang sudah diperbarui
            return _handle_video_page(url)
        else:
            return [], f"Format tidak didukung. Tipe konten: '{content_type}'"
    except requests.exceptions.RequestException as e:
        return [], f"Gagal mengakses URL. Pastikan link benar dan dapat diakses. Detail: {e}"
    except Exception as e:
        return [], f"Terjadi kesalahan saat memproses URL. Detail: {e}"

def fetch_live_stream_source(raw_url: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Memvalidasi dan mengambil URL live stream dengan alur yang logis:
    1. Cek apakah ini URL YouTube.
    2. Jika ya, coba metode non-API (yt-dlp) terlebih dahulu.
    3. Jika gagal, gunakan YouTube API sebagai fallback yang andal.
    4. Jika bukan YouTube, gunakan metode non-API (yt-dlp, streamlink).

    Args:
        raw_url (str): URL input dari pengguna.

    Returns:
        Dict[str, Any] or None: Dictionary berisi info stream jika valid, jika tidak None.
    """
    display_url = raw_url
    # --- Pintu gerbang utama untuk URL RTSP ---
    if raw_url.strip().lower().startswith("rtsp://"):
        return {
            "src": raw_url.strip(), 
            "title": "RTSP Stream",
            "is_live": True, 
            "display_url": raw_url.strip(),
            "type": "rtsp"  # Tambahkan tipe untuk identifikasi
        }, "URL RTSP terdeteksi. Pratinjau akan berupa gambar statis."
    
    # --- ALUR KHUSUS UNTUK YOUTUBE ---
    is_youtube = "youtube.com" in raw_url or "youtu.be" in raw_url
    if is_youtube:
        video_id_match = re.search(r"(?:v=|\/|live\/|embed\/|shorts\/)([0-9A-Za-z_-]{11})", raw_url)
        if video_id_match:
            video_id = video_id_match.group(1)
            # Langsung gunakan API sebagai satu-satunya metode
            return _get_youtube_stream_with_api(video_id)
        else:
            return None, "URL adalah video biasa. Silakan gunakan halaman 'Deteksi Awan'."

    # --- ALUR UNTUK PLATFORM LAIN (NON-YOUTUBE) ---
    try: # Coba dengan yt-dlp
        with yt_dlp.YoutubeDL({"format": "best", "quiet": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(raw_url, download=False)
            if info and info.get('is_live'):
                return {
                    "src": info["url"], "title": info.get("title", "Live Stream"),
                    "is_live": True, "display_url": display_url
                }, None
    except Exception:
        pass

    try: # Coba dengan Streamlink
        session = Streamlink()
        session.set_option("http-headers", {"User-Agent": "Mozilla/5.0"})
        streams = session.streams(raw_url)
        if "best" in streams:
            return {
                "src": streams["best"].url, "title": "Live Stream",
                "is_live": True, "display_url": display_url
            }, None
    except Exception:
        pass

    return None, "Gagal memvalidasi URL sebagai siaran langsung yang didukung."

def get_video_metadata(video_file: IO[bytes]) -> Tuple[Optional[Image.Image], float]:
    """
    Mengekstrak metadata penting dari file video in-memory dalam sekali proses.
    
    Args:
        video_file (IO[bytes]): Objek file video seperti UploadedFile.
        
    Returns:
        Tuple[Optional[Image.Image], float]: 
        - Objek PIL Image dari frame pertama sebagai pratinjau.
        - Durasi video dalam detik.
    """
    preview_image = None
    duration_seconds = 0.0
    
    try:
        # Simpan ke file temporary sekali saja
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp4") as tmp:
            tmp.write(video_file.getvalue())
            tmp.seek(0)
            
            cap = cv2.VideoCapture(tmp.name)
            if cap.isOpened():
                # Ambil durasi
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration_seconds = (frame_count / fps) if fps > 0 else 0.0
                
                # Ambil frame pertama untuk pratinjau
                ret, frame = cap.read()
                if ret:
                    preview_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            cap.release()
        
        return preview_image, duration_seconds

    except Exception:
        # Jika gagal, kembalikan nilai default
        placeholder_path = PATHS.get('placeholder')
        if placeholder_path and os.path.exists(placeholder_path):
            preview_image = Image.open(placeholder_path)
        return preview_image, 60.0 # Default durasi 60 detik

def get_preview_as_pil(source: any, max_size: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
    """
    Fungsi inti terpusat untuk mendapatkan pratinjau sebagai objek PIL Image.
    Menangani berbagai sumber: path file (str), objek file bytes (io.BytesIO),
    direktori, gambar, dan video. Versi ini lebih robust.
    
    Args:
        source (any): Path ke file (str) atau objek file in-memory (io.BytesIO).
        max_size (tuple, optional): Jika diberikan (lebar, tinggi), akan mengubah ukuran gambar.
        
    Returns:
        Optional[Image.Image]: Objek PIL Image atau None jika gagal.
    """
    pil_img = None
    placeholder_path = PATHS.get('placeholder')

    try:
        # --- Kasus 1: Sumber adalah path (string) di disk ---
        if isinstance(source, str):
            if not os.path.exists(source):
                raise FileNotFoundError(f"Path tidak ditemukan: {source}")

            # 1a: Jika path adalah file video
            if os.path.isfile(source) and source.lower().endswith(VIDEO_EXTENSIONS):
                try:
                    cap = cv2.VideoCapture(source)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                except Exception as e:
                    print(f"Gagal membuat thumbnail dari video '{source}': {e}")
                    # Jika gagal, akan jatuh ke blok except luar untuk placeholder
            
            # 1b: Jika path adalah direktori (untuk frame live monitoring)
            elif os.path.isdir(source):
                try:
                    # Cari gambar pertama yang valid di dalam direktori
                    image_files = sorted([f for f in os.listdir(source) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    if image_files:
                        first_image_path = os.path.join(source, image_files[0])
                        pil_img = Image.open(first_image_path)
                except Exception as e:
                    print(f"Gagal membaca gambar dari direktori '{source}': {e}")

            # 1c: Jika path adalah file gambar biasa
            elif os.path.isfile(source):
                try:
                    pil_img = Image.open(source)
                except Exception as e:
                    print(f"Gagal membuka file gambar '{source}': {e}")

        # --- Kasus 2: Sumber adalah objek file in-memory (seperti UploadedFile) ---
        elif hasattr(source, 'read') and hasattr(source, 'name'):
            # Logika untuk file in-memory sudah cukup baik, bisa dipertahankan
            # (tapi untuk kasus Anda, sumbernya adalah path string)
            is_video = source.name.lower().endswith(VIDEO_EXTENSIONS)
            if is_video:
                with tempfile.NamedTemporaryFile(delete=True, suffix=os.path.splitext(source.name)[1]) as tmp:
                    tmp.write(source.getvalue())
                    tmp.seek(0)
                    cap = cv2.VideoCapture(tmp.name)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else: # Jika gambar
                pil_img = Image.open(source)

        # --- Logika untuk mengubah ukuran (jika diminta) ---
        if pil_img:
            if max_size:
                pil_img.thumbnail(max_size, Image.Resampling.LANCZOS)
            return pil_img.convert("RGB") # Selalu pastikan format RGB

    except Exception as e:
        print(f"Error umum di get_preview_as_pil untuk source '{source}': {e}")
    
    # --- Fallback: Jika semua gagal, kembalikan placeholder ---
    if placeholder_path and os.path.exists(placeholder_path):
        return Image.open(placeholder_path)
    
    return None

def get_preview_as_base64(path: str) -> str:
    """
    Pembungkus (wrapper) untuk get_preview_as_pil yang mengembalikan string Base64.
    Digunakan untuk menampilkan gambar di antarmuka web Streamlit.
    """
    pil_img = get_preview_as_pil(path)
    
    if pil_img:
        buffer = io.BytesIO()
        pil_img.convert("RGB").save(buffer, format="PNG") # Pastikan format RGB sebelum save
        encoded_string = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{encoded_string}"
        
    # Jika gagal, kembalikan string kosong atau path ke placeholder jika ada
    placeholder_path = PATHS.get('placeholder')
    if placeholder_path and os.path.exists(placeholder_path):
        with open(placeholder_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
            return f"data:image/png;base64,{encoded_string}"

    return ""

# --- Fungsi Helper Internal (diawali dengan _) ---

def _get_youtube_stream_with_api(video_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Mencoba mendapatkan detail live stream menggunakan YouTube Data API v3.
    Ini adalah metode yang paling andal untuk lingkungan cloud.
    """
    try:
        # 1. Ambil API Key dari Streamlit Secrets
        api_key = st.secrets["YOUTUBE_API_KEY"]
        youtube = build('youtube', 'v3', developerKey=api_key)
        
        # 2. Minta detail video
        request = youtube.videos().list(
            part="snippet,liveStreamingDetails",
            id=video_id
        )
        response = request.execute()

        items = response.get("items", [])
        if not items:
            return None, "Video YouTube dengan ID tersebut tidak ditemukan."
        
        video_data = items[0]
        live_details = video_data.get("liveStreamingDetails")
        
        # 3. Pastikan ini adalah live stream yang sedang aktif
        if not live_details or "actualStartTime" not in live_details or "actualEndTime" in live_details:
             return None, "URL adalah video YouTube biasa atau siaran langsung telah berakhir."

        # 4. Gunakan yt-dlp untuk mendapatkan URL stream mentah (lebih andal daripada hlsManifestUrl)
        with yt_dlp.YoutubeDL({'format': 'best', 'quiet': True}) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            result = {
                "src": info["url"], 
                "title": info.get("title", "YouTube Live Stream"),
                "is_live": True, 
                "display_url": f"https://www.youtube.com/watch?v={video_id}"
            }
            return result, "URL divalidasi menggunakan API resmi YouTube."

    except HttpError:
        return None, "Terjadi kesalahan pada YouTube API. Periksa kuota atau konfigurasi kunci Anda."
    except KeyError:
        return None, "Kunci 'YOUTUBE_API_KEY' tidak ditemukan di Streamlit Secrets. Silakan konfigurasikan."
    except Exception:
        # Jika API berhasil tetapi yt-dlp gagal (jarang terjadi), kembali ke metode non-API
        return None, None

def _handle_gdrive(url: str) -> Tuple[List[IO[bytes]], Optional[str]]:
    """Helper untuk menangani pengunduhan dari URL Google Drive (file/folder)."""
    temp_dir = tempfile.mkdtemp()
    try:
        if '/drive/folders/' in url: # Mengunduh seluruh isi folder
            downloaded_paths = gdown.download_folder(url, output=temp_dir, quiet=True, use_cookies=False)
        else: # Mengunduh sebagai file tunggal
            path = gdown.download(url, output=os.path.join(temp_dir, ''), quiet=True, fuzzy=True)
            downloaded_paths = [path] if path else []
        
        if not downloaded_paths:
            return [], "Gagal mengunduh atau tidak ada media di GDrive. Pastikan tautan publik."

        media_files = []
        for path in downloaded_paths:
            if path and os.path.exists(path): # Jika file adalah ZIP, ekstrak isinya
                if path.lower().endswith(ANALYSIS_CONFIG.get('zip_extension', '.zip')):
                    with open(path, "rb") as f_zip:
                        extracted_files, error = extract_media_from_zip(f_zip)
                        if error: return [], error # Teruskan error jika ada
                        media_files.extend(extracted_files)
                else: # Proses file media biasa
                    with open(path, "rb") as f:
                        file_bytes = f.read()
                    file_buffer = io.BytesIO(file_bytes)
                    file_buffer.name = os.path.basename(path)
                    media_files.append(file_buffer)
        
        if not media_files:
            return [], "Tidak ada media yang valid ditemukan di dalam tautan Google Drive."
        return media_files, None
    finally:
        if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

def _handle_direct_media(response: requests.Response) -> Tuple[List[IO[bytes]], Optional[str]]:
    """Helper untuk memproses respons yang sudah divalidasi sebagai media langsung."""
    file_bytes = response.content
    file_buffer = io.BytesIO(file_bytes)
    
    # Logika penamaan file cerdas
    content_type = response.headers.get('content-type', '').lower()
    ext = mimetypes.guess_extension(content_type) or ''
    path = urlparse(response.url).path
    base_name = os.path.basename(path)

    if base_name and '.' in base_name:
        file_buffer.name = base_name
    else:
        file_buffer.name = f"downloaded_{uuid.uuid4().hex[:8]}{ext}"

    return [file_buffer], None

def _handle_video_page(url: str) -> Tuple[List[IO[bytes]], Optional[str]]:
    """Helper untuk memproses halaman web (text/html) dengan yt-dlp."""
    # --- ALUR KHUSUS UNTUK YOUTUBE ---
    try:
        ydl_opts = {'noplaylist': True, 'quiet': True}
        
        # Jika YouTube, WAJIBKAN penggunaan API Key
        is_youtube = "youtube.com" in url or "youtu.be" in url
        if is_youtube:
            if "YOUTUBE_API_KEY" not in st.secrets:
                return [], "Kunci 'YOUTUBE_API_KEY' tidak ditemukan di Secrets untuk mengunduh video YouTube."
            ydl_opts['youtube_api_key'] = st.secrets["YOUTUBE_API_KEY"]
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if 'entries' in info and not info.get('id'): return [], "URL adalah playlist/channel."
            if info.get('is_live'): return [], "URL adalah siaran langsung."

            # Proses unduh
            temp_dl_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.%(ext)s")
            ydl_opts_dl = {**ydl_opts, 'outtmpl': temp_dl_path}
            with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl_dl:
                ydl_dl.download([url])
                filename = ydl_dl.prepare_filename(info)
            
            if not filename or not os.path.exists(filename):
                return [], "File video tidak ditemukan setelah diunduh."

            with open(filename, "rb") as f: file_bytes = f.read()
            os.unlink(filename)
            file_buffer = io.BytesIO(file_bytes)
            clean_title = re.sub(r'[\\/*?:"<>|]', "", info.get('title', 'video'))
            file_buffer.name = f"{clean_title}.mp4"
            return [file_buffer], "Video diunduh menggunakan API." if is_youtube else None
            
    except yt_dlp.utils.DownloadError as e:
        return [], f"Gagal memproses URL. Detail: {e}"