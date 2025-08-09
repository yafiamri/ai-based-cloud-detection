# utils/config.py
import yaml
import streamlit as st
from typing import Dict, Any

@st.cache_resource
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Memuat file konfigurasi YAML dan menyimpannya dalam cache resource Streamlit.
    Cache ini memastikan file YAML hanya dibaca dari disk sekali per sesi.

    Args:
        config_path (str): Path ke file config.yaml.

    Returns:
        Dict[str, Any]: Dictionary yang berisi semua konfigurasi dari file YAML.
    """
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        st.error(f"File konfigurasi tidak ditemukan di path: {config_path}. Pastikan file 'config.yaml' ada di direktori utama.")
        return {}

# Muat konfigurasi sekali untuk digunakan di seluruh modul
config = load_config()