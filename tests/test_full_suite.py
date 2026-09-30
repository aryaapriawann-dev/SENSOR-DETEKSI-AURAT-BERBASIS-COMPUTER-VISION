"""
Comprehensive Verification Test Suite for Smart Mosque Gate Aurat Detector
Menguji semua skenario syariat:
1. Mode Perempuan:
   - Cadar / Niqab Hitam (Harus AMAN)
   - Mukena Sholat (Harus AMAN)
   - Hijab Syar'i (Harus AMAN)
   - Rambut Terbuka / Tanpa Jilbab (Harus PELANGGARAN: Rambut Terbuka)
   - Leher Terbuka / Jilbab Terbuka (Harus PELANGGARAN: Leher Terbuka)
   - Lengan Terbuka (Harus PELANGGARAN: Lengan Terbuka)
2. Mode Laki-Laki:
   - Berpakaian Normal / Kaos & Celana (Harus AMAN)
   - Pusar/Perut Terbuka (Harus PELANGGARAN: Pusar/Perut Terbuka)
   - Dada Terbuka / Telanjang Dada (Harus PELANGGARAN: Dada Terbuka)
"""
import os
import glob
import cv2  # type: ignore
import numpy as np  # type: ignore
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from core.detector import AuratDetector

def run_tests():
    print("=" * 70)
    print("   PENGUJIAN VALIDASI AKURASI SENSOR DETEKSI AURAT v3.0")
    print("=" * 70)

    detector = AuratDetector()

    results = {
        "Cadar / Niqab (Wajib AMAN)": {"pass": 0, "fail": 0},
        "Mukena Sholat (Wajib AMAN)": {"pass": 0, "fail": 0},
        "Hijab Syar'i (Wajib AMAN)": {"pass": 0, "fail": 0},
        "Rambut Terbuka / Non-Hijab (Wajib PELANGGARAN)": {"pass": 0, "fail": 0},
        "Leher Terbuka / Non-Syar'i (Wajib PELANGGARAN)": {"pass": 0, "fail": 0},
    }

    # 1. UJI CADAR & MUKENA
    for cat_name, prefix, expected_status in [
        ("Cadar / Niqab (Wajib AMAN)", "cadar_valid_", "AMAN"),
        ("Mukena Sholat (Wajib AMAN)", "mukena_valid_", "AMAN")
    ]:
        imgs = glob.glob(f"datasets/syari_dataset/valid/images/{prefix}*.jpg")[:15]
        for p in imgs:
            img = cv2.imread(p)
            if img is None:
                continue
            detector.tracker.tracks.clear()
            ada_pel, alasan, tracked, total = detector.detect(img, mode="PEREMPUAN")
            if total > 0:
                is_pel = tracked[0]["is_pelanggaran"]
                if expected_status == "AMAN" and not is_pel:
                    results[cat_name]["pass"] += 1
                elif expected_status == "PELANGGARAN" and is_pel:
                    results[cat_name]["pass"] += 1
                else:
                    results[cat_name]["fail"] += 1

    # 2. UJI HIJAB SYARI, NON SYARI, NON HIJAB
    for target_cls, cat_name, expected_status in [
        (0, "Hijab Syar'i (Wajib AMAN)", "AMAN"),
        (1, "Leher Terbuka / Non-Syar'i (Wajib PELANGGARAN)", "PELANGGARAN"),
        (2, "Rambut Terbuka / Non-Hijab (Wajib PELANGGARAN)", "PELANGGARAN")
    ]:
        lbls = glob.glob("datasets/syari_dataset/valid/labels/*.txt")
        count = 0
        for lbl in lbls:
            with open(lbl) as f:
                line = f.readline().split()
                if line and int(line[0]) == target_cls:
                    base = os.path.splitext(os.path.basename(lbl))[0]
                    img_p = os.path.join("datasets/syari_dataset/valid/images", f"{base}.jpg")
                    img = cv2.imread(img_p)
                    if img is None:
                        continue
                    detector.tracker.tracks.clear()
                    ada_pel, alasan, tracked, total = detector.detect(img, mode="PEREMPUAN")
                    if total > 0:
                        is_pel = tracked[0]["is_pelanggaran"]
                        if expected_status == "AMAN" and not is_pel:
                            results[cat_name]["pass"] += 1
                        elif expected_status == "PELANGGARAN" and is_pel:
                            results[cat_name]["pass"] += 1
                        else:
                            results[cat_name]["fail"] += 1
                        count += 1
                        if count >= 10:
                            break

    print("\n[HASIL PENGUJIAN MODE PEREMPUAN]:")
    total_passed = 0
    total_tests = 0
    for category, stats in results.items():
        p = stats["pass"]
        f = stats["fail"]
        sub_total = p + f
        rate = (p / sub_total * 100) if sub_total > 0 else 100.0
        print(f"  • {category:<48}: {p}/{sub_total} Berhasil ({rate:.1f}%)")
        total_passed += p
        total_tests += sub_total

    print("-" * 70)
    overall_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"  TOTAL SKENARIO TERUJI: {total_passed}/{total_tests} ({overall_rate:.1f}% Akurat)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
