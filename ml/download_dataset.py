"""
Roboflow Dataset Downloader Pipeline — Monorepo ML Module
"""
import os
import sys

# Memastikan site-packages Windows Store terbaca
site_pack = os.path.expanduser(r"~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages")
if os.path.exists(site_pack) and site_pack not in sys.path:
    sys.path.append(site_pack)

from core.config import DATASETS_DIR  # type: ignore

def download(api_key: str, workspace: str = "musrifatul-arifah-t2onk", project_id: str = "hijab-ztppw", version: int = 1, target_dir: str = None):
    try:
        from roboflow import Roboflow  # type: ignore
    except ImportError:
        print("[*] Menginstall roboflow library...")
        os.system(f'"{sys.executable}" -m pip install roboflow')
        from roboflow import Roboflow  # type: ignore

    if target_dir is None:
        target_dir = os.path.join(DATASETS_DIR, "hijab_dataset")

    os.makedirs(DATASETS_DIR, exist_ok=True)
    print(f"[*] Menghubungkan ke Roboflow ({workspace}/{project_id} v{version})...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_id)
    dataset = project.version(version).download("yolov8", location=target_dir)
    print(f"[+] Dataset berhasil diunduh ke: {dataset.location}")
    return dataset.location

if __name__ == "__main__":
    api_key = os.environ.get("ROBOFLOW_API_KEY", "WVa8kxfb3Zw7tWhoaVi3")
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    if not api_key:
        api_key = input("Masukkan Roboflow API Key Anda: ").strip()

    if api_key:
        download(api_key)
    else:
        print("[!] API Key tidak boleh kosong.")
