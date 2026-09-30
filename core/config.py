"""
Konfigurasi Sistem Smart Mosque Gate — Monorepo Core
"""
import os
import sys

# Memastikan user site-packages Windows Store terbaca
site_pack = os.path.expanduser(r"~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages")
if os.path.exists(site_pack) and site_pack not in sys.path:
    sys.path.append(site_pack)

# Direktori Dasar Proyek
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

# Path Model
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "aurat_best.pt")
YOLO_POSE_PATH = os.path.join(MODELS_DIR, "yolov8n-pose.pt")

# Palet Warna UI (Tema Masjid Modern)
GOLD        = (  0, 200, 215)   # Kuning Keemasan (BGR)
GOLD_TERANG = (  0, 230, 255)
NAVY        = ( 30,  15,   5)   # Biru Tua
EMERALD     = ( 80, 200,  80)   # Hijau Zamrud
MERAH       = ( 40,  40, 225)   # Merah Peringatan
PUTIH       = (255, 255, 255)
ABU_TUA     = ( 60,  60,  60)
ABU_TERANG  = (180, 180, 180)
CYAN_ACCENT = (230, 216,   0)

# Ambang Batas (Threshold)
YOLO_CONF_THRESHOLD = 0.40
SMOOTH_N_FRAMES     = 7
