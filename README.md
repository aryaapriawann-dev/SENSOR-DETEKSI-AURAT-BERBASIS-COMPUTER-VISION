# 🕌 Smart Mosque Gate — Monorepo Architecture

Sistem Deteksi Aurat Otomatis Berbasis **YOLOv8 Deep Learning** & **MediaPipe Pose Estimation** untuk Gerbang Masjid Pintar.

---

## 📂 Struktur Monorepo

```
AURAT DATA SET/
├── apps/                          # Aplikasi Pengguna
│   └── gate/
│       └── main.py                # Aplikasi Real-Time Smart Mosque Gate
│
├── core/                          # Modul & Detektor Inti
│   ├── __init__.py
│   ├── config.py                  # Konfigurasi, Palet Warna & Path
│   └── detector.py                # Engine AuratDetector (YOLOv8 + MediaPipe)
│
├── ml/                            # Machine Learning & Training Pipeline
│   ├── download_dataset.py        # Pipeline unduh dataset dari Roboflow
│   └── train.py                   # Pipeline training model YOLOv8
│
├── models/                        # Tempat Bobot Model AI (.pt)
│   ├── aurat_best.pt              # Model terlatih deteksi Hijab vs Non-Hijab
│   ├── yolov8n.pt                 # Model dasar YOLOv8 Nano
│   └── yolov8n-pose.pt            # Model dasar YOLOv8 Pose
│
├── Hijab-1/                       # Dataset Roboflow (train, val, test)
├── .vscode/                       # Konfigurasi IDE (Settings & Path)
│   └── settings.json
│
├── run_gate.py                    # Runner Aplikasi Utama
├── AURAT.PY                       # Runner Alternatif (Backward-Compatible)
├── requirements.txt               # Daftar Library Python
└── README.md                      # Dokumentasi Proyek
```

---

## 🚀 Cara Menjalankan

### 1. Jalankan Aplikasi Gerbang (Gate Camera)
```powershell
python run_gate.py
```
*(atau: `python AURAT.PY`)*

#### Kontrol Keyboard:
* **`[P]`** : Beralih ke **Mode Perempuan** (Wajib Hijab).
* **`[L]`** : Beralih ke **Mode Laki-Laki** (Batas Pusar s.d. Lutut).
* **`[Q]`** : Keluar dari aplikasi.

---

### 2. Download Dataset Tambahan dari Roboflow (Opsional)
```powershell
python ml/download_dataset.py
```

### 3. Training Ulang Model YOLO (Opsional)
```powershell
python ml/train.py
```
Model hasil training otomatis akan diekspor langsung ke folder `models/aurat_best.pt`.

# SENSOR-DETEKSI-AURAT-BERBASIS-COMPUTER-VISION
