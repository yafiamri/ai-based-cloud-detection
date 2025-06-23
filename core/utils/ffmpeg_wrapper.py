# Lokasi File: core/utils/ffmpeg_wrapper.py
"""
Pembungkus (Wrapper) untuk menjalankan perintah FFmpeg secara terprogram.
Menyediakan fungsi untuk merangkai gambar menjadi video dan
meng-overlay urutan gambar (masker) ke video utama.
"""

from __future__ import annotations
import subprocess
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)

def _get_ffmpeg_path() -> str:
    """Mencari path FFmpeg di sistem, fallback ke folder proyek."""
    # Prioritaskan ffmpeg yang ada di PATH sistem
    system_path = shutil.which("ffmpeg")
    if system_path:
        return system_path
    
    # Fallback ke folder ffmpeg/ di root proyek
    local_path = Path(__file__).resolve().parents[3] / "ffmpeg" / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if local_path.is_file():
        return str(local_path)
        
    raise FileNotFoundError(
        "ffmpeg tidak ditemukan. Pastikan ia ada di PATH sistem "
        "atau di dalam folder 'ffmpeg' di root proyek."
    )

FFMPEG_PATH = _get_ffmpeg_path()

def stitch_frames_to_video(
    frames_dir: Path,
    output_path: Path,
    fps: int,
    frame_pattern: str = "frame_%06d.jpg"
):
    """
    Menjahit urutan gambar dari sebuah direktori menjadi file video.
    (Metode Video Awal)
    """
    command = [
        FFMPEG_PATH,
        "-y",  # Overwrite output file jika sudah ada
        "-framerate", str(fps),
        "-i", str(frames_dir / frame_pattern),
        "-c:v", "libx264",      # Codec video yang umum
        "-pix_fmt", "yuv420p",  # Format piksel untuk kompatibilitas luas
        "-crf", "23",           # Kualitas video (lower is better)
        "-preset", "veryfast",
        str(output_path),
    ]
    
    log.info(f"Menjalankan FFmpeg untuk menjahit frame: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        log.info(f"Video berhasil dibuat di: {output_path}")
    except subprocess.CalledProcessError as e:
        log.error(f"FFmpeg gagal saat menjahit frame: {e.stderr}")
        raise

def overlay_image_sequence(
    base_video_path: Path,
    image_sequence_dir: Path,
    output_path: Path,
    fps: int,
    frame_pattern: str = "mask_%06d.png"
):
    """
    Meng-overlay urutan gambar (dengan transparansi) ke atas video dasar.
    (Metode Video Baru yang Diinginkan)
    """
    image_sequence_input = image_sequence_dir / frame_pattern
    
    # Perintah FFmpeg ini menggunakan dua input (-i) dan filter kompleks (-filter_complex)
    command = [
        FFMPEG_PATH,
        "-y", # Overwrite output
        "-i", str(base_video_path),          # Input 0: Video dasar
        "-framerate", str(fps),              # Atur framerate untuk urutan gambar
        "-i", str(image_sequence_input),     # Input 1: Urutan gambar masker
        "-filter_complex", "[0:v][1:v]overlay=shortest=1", # Filter: tumpuk input 1 di atas input 0
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "veryfast",
        str(output_path),
    ]

    log.info(f"Menjalankan FFmpeg untuk overlay: {' '.join(command)}")
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        log.info(f"Video overlay berhasil dibuat di: {output_path}")
    except subprocess.CalledProcessError as e:
        log.error(f"FFmpeg gagal saat overlay: {e.stderr}")
        raise