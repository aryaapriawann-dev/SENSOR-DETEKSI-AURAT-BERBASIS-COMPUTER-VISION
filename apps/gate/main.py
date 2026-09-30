"""
Smart Mosque Gate Application — Multi-Person Tracking Architecture
Mendeteksi dan melacak banyak orang sekaligus, bounding box melengket di tubuh manusia.
"""
import os
import sys

# Memastikan user site-packages Windows Store & root monorepo terbaca
site_pack = os.path.expanduser(r"~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages")
if os.path.exists(site_pack) and site_pack not in sys.path:
    sys.path.append(site_pack)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import time
import math
import cv2  # type: ignore
import numpy as np  # type: ignore

from core.config import (  # type: ignore
    GOLD, GOLD_TERANG, NAVY, EMERALD, MERAH, PUTIH, ABU_TUA, ABU_TERANG,
    SMOOTH_N_FRAMES
)
from core.detector import AuratDetector  # type: ignore

# Audio handling
try:
    import pygame  # type: ignore
    pygame.mixer.init()
    snd_warn = os.path.join(BASE_DIR, "suara_ai.wav")
    if not os.path.exists(snd_warn):
        snd_warn = os.path.join(BASE_DIR, "suara_ai.mp3")
    
    snd_safe = os.path.join(BASE_DIR, "suara_aman.wav")
    if not os.path.exists(snd_safe):
        snd_safe = os.path.join(BASE_DIR, "suara_aman.mp3")

    suara_peringatan = pygame.mixer.Sound(snd_warn) if os.path.exists(snd_warn) else None
    suara_aman       = pygame.mixer.Sound(snd_safe) if os.path.exists(snd_safe) else None
    PYGAME_OK = True
except Exception:
    PYGAME_OK = False
    suara_peringatan = suara_aman = None


def overlay_transparan(frame, x1, y1, x2, y2, warna, alpha=0.55, radius=0):
    overlay = frame.copy()
    if radius > 0:
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), warna, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), warna, -1)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(overlay, (cx, cy), radius, warna, -1)
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), warna, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def gambar_teks_shadow(frame, teks, pos, font, skala, warna, tebal=2, shadow_offset=2):
    x, y = pos
    cv2.putText(frame, teks, (x + shadow_offset, y + shadow_offset),
                font, skala, (0, 0, 0), tebal + 1, cv2.LINE_AA)
    cv2.putText(frame, teks, pos, font, skala, warna, tebal, cv2.LINE_AA)


def gambar_garis_hias(frame, w, h, t):
    for x in range(0, w, 3):
        y1 = int(3 + 3 * math.sin(x * 0.04 + t * 2))
        cv2.circle(frame, (x, y1), 1, GOLD, -1)
        y2 = int(h - 4 + 3 * math.sin(x * 0.04 - t * 2))
        cv2.circle(frame, (x, y2), 1, GOLD, -1)


def gambar_ikon_mode(frame, mode, x, y, ukuran=22):
    warna = EMERALD if mode == "PEREMPUAN" else GOLD
    cv2.circle(frame, (x, y - ukuran // 2), ukuran // 4, warna, -1)
    if mode == "PEREMPUAN":
        pts = np.array([[x, y - ukuran // 4],
                        [x - ukuran // 2, y + ukuran // 2],
                        [x + ukuran // 2, y + ukuran // 2]], np.int32)
        cv2.fillPoly(frame, [pts], warna)
    else:
        cv2.rectangle(frame, (x - ukuran // 4, y - ukuran // 4),
                      (x + ukuran // 4, y + ukuran // 2), warna, -1)


def gambar_bounding_box_elegan(frame, x1, y1, x2, y2, warna, label_teks, pid, zones=None):
    """Menggambar bounding box melengket di tubuh dengan corner bracket, badge elegan, dan highlight zona aurat."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), warna, 2)

    # Corner bracket highlight (aksen sudut tebal modern)
    line_len = min(20, (x2 - x1) // 5, (y2 - y1) // 5)
    cv2.line(frame, (x1, y1), (x1 + line_len, y1), warna, 3)
    cv2.line(frame, (x1, y1), (x1, y1 + line_len), warna, 3)
    cv2.line(frame, (x2, y1), (x2 - line_len, y1), warna, 3)
    cv2.line(frame, (x2, y1), (x2, y1 + line_len), warna, 3)
    cv2.line(frame, (x1, y2), (x1 + line_len, y2), warna, 3)
    cv2.line(frame, (x1, y2), (x1, y2 - line_len), warna, 3)
    cv2.line(frame, (x2, y2), (x2 - line_len, y2), warna, 3)
    cv2.line(frame, (x2, y2), (x2, y2 - line_len), warna, 3)

    # Indikator spesifik zona aurat yang terbuka (misal: Lengan / Leher)
    if zones:
        for z in zones:
            if len(z) == 5:
                zx1, zy1, zx2, zy2, zlabel = z
                # Highlight area pelanggaran
                overlay_transparan(frame, zx1, zy1, zx2, zy2, (0, 0, 180), alpha=0.30, radius=2)
                cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), (0, 0, 255), 1, cv2.LINE_AA)
                tag_y = max(14, zy1 - 4)
                cv2.putText(frame, f"! {zlabel}", (zx1 + 2, tag_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (50, 120, 255), 1, cv2.LINE_AA)

    # Badge ID & Status di atas kepala (ukuran proporsional)
    badge_txt = f"#{pid} | {label_teks}"
    (tw, th), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    badge_y1 = max(0, y1 - 24)
    badge_y2 = y1
    badge_x2 = min(frame.shape[1], x1 + tw + 14)
    overlay_transparan(frame, x1, badge_y1, badge_x2, badge_y2, (15, 12, 10), alpha=0.80, radius=3)
    cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), warna, 1)
    cv2.putText(frame, badge_txt, (x1 + 6, badge_y2 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, warna, 1, cv2.LINE_AA)


def run_app():
    print("\n" + "=" * 60)
    print("  SMART MOSQUE GATE v3.0 — COMPACT HUD MULTI-PERSON")
    print("=" * 60)
    print("[*] Menginisialisasi Multi-Person Detector & Aurat Engine...")
    detector = AuratDetector()
    print("[+] Model AI siap!")
    print("[*] Menghubungkan ke kamera webcam (Device 0)...")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[!] GAGAL MEMBUKA KAMERA! Pastikan webcam terpasang dan tidak dipakai aplikasi lain.")
        return

    print("[+] Kamera aktif! Jendela antarmuka sedang ditampilkan.")
    window_title = "Smart Mosque Gate — Multi-Person Aurat Tracking v3.0"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    mode_gate = "PEREMPUAN"
    sudah_bunyi_aman = False
    history_status = []
    prev_time = time.time()
    fps_val = 0.0
    t_start = time.time()

    print("[*] Kontrol Keyboard:")
    print("    [P] Mode Perempuan (Wajib Hijab)")
    print("    [L] Mode Laki-Laki (Batas Aurat Pria)")
    print("    [Q] Keluar Sistem\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        t = time.time() - t_start

        # FPS
        now = time.time()
        fps_val = 0.9 * fps_val + 0.1 * (1.0 / max(now - prev_time, 1e-5))
        prev_time = now

        # ── 1. Proses Multi-Person & Aurat Detection ──────────
        ada_pelanggaran_frame, alasan_teks, tracked_persons, total_orang = detector.process_frame(
            frame, mode=mode_gate
        )

        history_status.append(ada_pelanggaran_frame)
        if len(history_status) > SMOOTH_N_FRAMES:
            history_status.pop(0)
        pelanggaran_gate = sum(history_status) > (SMOOTH_N_FRAMES // 2)

        # ── 2. Render Bounding Box Melengket di Setiap Tubuh ──
        jumlah_pelanggar = 0
        for p in tracked_persons:
            x1, y1, x2, y2 = p["box"]
            is_melanggar = p["is_pelanggaran"]
            pid = p["id"]
            label_txt = p["label"]
            zones = p.get("zones", []) if is_melanggar else []

            if is_melanggar:
                jumlah_pelanggar += 1
                warna_box = MERAH
            else:
                warna_box = EMERALD

            gambar_bounding_box_elegan(frame, x1, y1, x2, y2, warna_box, label_txt, pid, zones=zones)

        # ── 3. Logika Audio Peringatan ────────────────────────
        if PYGAME_OK:
            if total_orang > 0 and pelanggaran_gate:
                if suara_peringatan and not pygame.mixer.get_busy():
                    suara_peringatan.play()
                sudah_bunyi_aman = False
            elif total_orang > 0 and not pelanggaran_gate:
                if not sudah_bunyi_aman:
                    pygame.mixer.stop()
                    if suara_aman:
                        suara_aman.play()
                    sudah_bunyi_aman = True
            else:
                sudah_bunyi_aman = False

        # ════════════════════════════════════════════════════════
        # UI — HEADER RAMPING
        # ════════════════════════════════════════════════════════
        overlay_transparan(frame, 0, 0, w, 48, (10, 8, 5), alpha=0.75)
        overlay_transparan(frame, 0, h - 38, w, h, (10, 8, 5), alpha=0.75)

        gambar_teks_shadow(frame, "SMART MOSQUE GATE", (16, 32),
                           cv2.FONT_HERSHEY_DUPLEX, 0.85, GOLD_TERANG, 2)
        cv2.putText(frame, "v3.0 Multi-Person Tracking",
                    (260, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ABU_TERANG, 1, cv2.LINE_AA)

        warna_fps = EMERALD if fps_val >= 24 else (0, 200, 200) if fps_val >= 15 else MERAH
        gambar_teks_shadow(frame, f"FPS {fps_val:4.1f}",
                           (w - 120, 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, warna_fps)

        # ════════════════════════════════════════════════════════
        # UI — PANEL KIRI: Status Gate & Deteksi (UKURAN RAMPING/COMPACT)
        # ════════════════════════════════════════════════════════
        px, py = 15, 60
        pw, ph = 240, 165
        overlay_transparan(frame, px, py, px + pw, py + ph, (10, 8, 6), alpha=0.70, radius=8)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), GOLD, 1)

        # Judul Card
        cv2.putText(frame, "STATUS GERBANG", (px + 10, py + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, GOLD, 1, cv2.LINE_AA)
        cv2.line(frame, (px + 8, py + 28), (px + pw - 8, py + 28), GOLD, 1)

        # Mode Ikon & Teks
        gambar_ikon_mode(frame, mode_gate, px + 22, py + 50, ukuran=20)
        warna_mode = EMERALD if mode_gate == "PEREMPUAN" else GOLD
        cv2.putText(frame, f"MODE: {mode_gate}", (px + 40, py + 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, warna_mode, 1, cv2.LINE_AA)

        # Statistik Orang
        orang_txt = f"Orang: {total_orang}  (Aman: {total_orang - jumlah_pelanggar} | Buka: {jumlah_pelanggar})"
        cv2.putText(frame, orang_txt, (px + 10, py + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, PUTIH, 1, cv2.LINE_AA)

        # Pill Status Akses Gerbang (Compact)
        if total_orang == 0:
            status_txt   = "STANDBY..."
            warna_status = ABU_TERANG
            bg_status    = (35, 35, 35)
        elif pelanggaran_gate:
            status_txt   = "AKSES DITOLAK"
            warna_status = MERAH
            bg_status    = (20, 20, 85)
        else:
            status_txt   = "AKSES DITERIMA"
            warna_status = EMERALD
            bg_status    = (10, 55, 10)

        overlay_transparan(frame, px + 8, py + 86, px + pw - 8, py + 126,
                           bg_status, alpha=0.85, radius=6)
        (tw, _), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
        tx = px + 8 + (pw - 16 - tw) // 2
        gambar_teks_shadow(frame, status_txt, (tx, py + 114),
                           cv2.FONT_HERSHEY_DUPLEX, 0.65, warna_status, 2)

        # Keterangan Singkat
        warna_alasan = MERAH if pelanggaran_gate else EMERALD if total_orang > 0 else ABU_TERANG
        if total_orang == 0:
            ket_display = "Menunggu Jamaah..."
        elif pelanggaran_gate:
            pelanggar_pertama = next((p for p in tracked_persons if p["is_pelanggaran"]), None)
            ket_display = pelanggar_pertama["label"] if pelanggar_pertama else alasan_teks
        else:
            ket_display = "Aurat Tertutup Sempurna"

        if len(ket_display) > 28:
            ket_display = ket_display[:25] + "..."
        cv2.putText(frame, f"Ket: {ket_display}", (px + 10, py + 148),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, warna_alasan, 1, cv2.LINE_AA)

        # ════════════════════════════════════════════════════════
        # UI — PANEL KANAN: Panduan Gerbang (UKURAN RAMPING/COMPACT)
        # ════════════════════════════════════════════════════════
        rw = 185
        rh = 115
        rx = w - rw - 15
        ry = 60
        overlay_transparan(frame, rx, ry, rx + rw, ry + rh, (10, 8, 6), alpha=0.70, radius=8)
        cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), GOLD, 1)

        cv2.putText(frame, "PANDUAN", (rx + 10, ry + 22),
                    cv2.FONT_HERSHEY_DUPLEX, 0.52, GOLD, 1, cv2.LINE_AA)
        cv2.line(frame, (rx + 8, py + 28), (rx + rw - 8, py + 28), GOLD, 1)

        panduan = [
            ("[P]", "Mode Perempuan"),
            ("[L]", "Mode Laki-Laki"),
            ("[Q]", "Keluar Sistem"),
        ]
        for i, (tombol, label) in enumerate(panduan):
            y_p = ry + 52 + i * 26
            overlay_transparan(frame, rx + 8, y_p - 14, rx + 44, y_p + 6,
                               (35, 25, 5), alpha=0.80, radius=4)
            cv2.putText(frame, tombol, (rx + 12, y_p),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, GOLD_TERANG, 1, cv2.LINE_AA)
            cv2.putText(frame, label, (rx + 50, y_p),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, PUTIH, 1, cv2.LINE_AA)

        # ════════════════════════════════════════════════════════
        # UI — BORDER PERINGATAN (Animasi saat ada pelanggaran)
        # ════════════════════════════════════════════════════════
        if total_orang > 0 and pelanggaran_gate:
            tebal_border = 8 if int(t * 5) % 2 == 0 else 5
            cv2.rectangle(frame, (0, 0), (w, h), MERAH, tebal_border)
            gambar_teks_shadow(frame, "!! AURAT TIDAK SESUAI SYARIAT !!",
                               (w // 2 - 250, h // 2),
                               cv2.FONT_HERSHEY_DUPLEX, 1.0, MERAH, 2)
        elif total_orang > 0 and not pelanggaran_gate:
            cv2.rectangle(frame, (0, 0), (w, h), EMERALD, 2)

        gambar_garis_hias(frame, w, h, t)
        cv2.putText(frame,
                    "Smart Mosque Gate  |  [P] Perempuan  |  [L] Laki-Laki  |  [Q] Keluar",
                    (w // 2 - 240, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ABU_TERANG, 1, cv2.LINE_AA)

        cv2.imshow(window_title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('l'):
            mode_gate = "LAKI-LAKI"
            sudah_bunyi_aman = False
            history_status.clear()
            print("[MODE GATE] Dialihkan ke LAKI-LAKI")
        elif key == ord('p'):
            mode_gate = "PEREMPUAN"
            sudah_bunyi_aman = False
            history_status.clear()
            print("[MODE GATE] Dialihkan ke PEREMPUAN")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_app()
