"""
Entry Point Peluncur Utama Smart Mosque Gate — Monorepo
Jalankan file ini untuk memulai aplikasi: python run_gate.py
"""
import os
import sys

print("\n" + "=" * 60)
print("  SMART MOSQUE GATE v3.0 (Monorepo)")
print("=" * 60)
print("[*] Menyiapkan lingkungan monorepo...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print("[*] Memuat PyTorch & Model AI YOLOv8 (membutuhkan ~2-3 detik)...")
from apps.gate.main import run_app  # type: ignore

if __name__ == "__main__":
    run_app()
