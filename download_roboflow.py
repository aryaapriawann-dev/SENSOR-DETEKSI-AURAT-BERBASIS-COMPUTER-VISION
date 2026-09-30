"""
Monorepo Wrapper: Download Dataset Roboflow
Meneruskan perintah ke ml/download_dataset.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.download_dataset import download

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
