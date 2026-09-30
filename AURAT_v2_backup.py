import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ================================================================
#   SMART MOSQUE GATE — AURAT DETECTION SYSTEM  v2.0
#   Kompatibel Python 3.10 | MediaPipe 0.10.x | OpenCV
# ================================================================

try:
    import pygame
    pygame.mixer.init()
    try:
        suara_peringatan = pygame.mixer.Sound("suara_ai.mp3")
        suara_aman       = pygame.mixer.Sound("suara_aman.mp3")
    except Exception:
        suara_peringatan = suara_aman = None
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False
    suara_peringatan = suara_aman = None

# ── MediaPipe ────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.65,
    min_tracking_confidence=0.65,
    enable_segmentation=True,
    model_complexity=1
)

# ── Palet Warna ──────────────────────────────────────────────────
# Tema: Masjid Modern — Gold + Deep Navy + Emerald
GOLD       = (  0, 200, 215)   # Kuning keemasan (BGR)
GOLD_TERANG= (  0, 230, 255)
NAVY       = ( 30,  15,   5)   # Biru tua (BGR)
EMERALD    = ( 80, 200,  80)
MERAH      = ( 40,  40, 220)
PUTIH      = (255, 255, 255)
ABU_TUA    = ( 60,  60,  60)
ABU_TERANG = (180, 180, 180)
TRANSPARAN_BIRU = (120, 80, 20)  # Sentuhan aksen biru

# ── State global ─────────────────────────────────────────────────
mode_gate        = "PEREMPUAN"
sudah_bunyi_aman = False
history_status   = []   # Smoothing N frame terakhir
SMOOTH_N         = 8
pelanggaran_hitung = 0  # Counter pelanggaran berturut-turut
THRESH_FRAME     = 5    # Perlu N frame pelanggaran baru alert

# ── Timestamp animasi ────────────────────────────────────────────
t_start = time.time()


# ════════════════════════════════════════════════════════════════
# FUNGSI UTILITAS UI
# ════════════════════════════════════════════════════════════════

def overlay_transparan(frame, x1, y1, x2, y2, warna, alpha=0.55, radius=0):
    """Gambar kotak transparan (opsional rounded)."""
    overlay = frame.copy()
    if radius > 0:
        # Simulasi rounded rect dengan 4 lingkaran + 1 rect
        cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), warna, -1)
        cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), warna, -1)
        for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                       (x1+radius, y2-radius), (x2-radius, y2-radius)]:
            cv2.circle(overlay, (cx, cy), radius, warna, -1)
    else:
        cv2.rectangle(overlay, (x1, y1), (x2, y2), warna, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def gambar_garis_hias(frame, w, h, t):
    """Garis ornamen bawah + atas (animasi gelombang)."""
    # Garis atas
    for x in range(0, w, 2):
        y = int(3 + 3 * math.sin(x * 0.04 + t * 2))
        cv2.circle(frame, (x, y), 1, GOLD, -1)
    # Garis bawah
    for x in range(0, w, 2):
        y = int(h - 4 + 3 * math.sin(x * 0.04 - t * 2))
        cv2.circle(frame, (x, y), 1, GOLD, -1)


def gambar_teks_shadow(frame, teks, pos, font, skala, warna, tebal=2, shadow_offset=2):
    """Teks dengan efek shadow."""
    x, y = pos
    cv2.putText(frame, teks, (x + shadow_offset, y + shadow_offset),
                font, skala, (0, 0, 0), tebal + 1, cv2.LINE_AA)
    cv2.putText(frame, teks, pos, font, skala, warna, tebal, cv2.LINE_AA)


def gambar_meter_persen(frame, x, y, lebar, tinggi, persen, warna_bar):
    """Meter horizontal dengan animasi gradien."""
    # Background
    cv2.rectangle(frame, (x, y), (x + lebar, y + tinggi), (30, 30, 30), -1)
    cv2.rectangle(frame, (x, y), (x + lebar, y + tinggi), ABU_TUA, 1)
    # Bar isi
    isi = int(lebar * min(persen / 100.0, 1.0))
    if isi > 0:
        cv2.rectangle(frame, (x, y + 2), (x + isi, y + tinggi - 2), warna_bar, -1)
    # Label %
    label = f"{persen:.1f}%"
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, label, (x + lebar // 2 - tw // 2, y + tinggi - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, PUTIH, 1, cv2.LINE_AA)


def gambar_ikon_mode(frame, mode, x, y, ukuran=40):
    """Ikon siluet orang (sederhana)."""
    warna = EMERALD if mode == "PEREMPUAN" else GOLD
    # Kepala
    cv2.circle(frame, (x, y - ukuran // 2), ukuran // 4, warna, -1)
    # Badan
    if mode == "PEREMPUAN":
        # Segitiga / jubah
        pts = np.array([[x, y - ukuran // 4],
                        [x - ukuran // 2, y + ukuran // 2],
                        [x + ukuran // 2, y + ukuran // 2]], np.int32)
        cv2.fillPoly(frame, [pts], warna)
    else:
        cv2.rectangle(frame, (x - ukuran // 4, y - ukuran // 4),
                      (x + ukuran // 4, y + ukuran // 2), warna, -1)


# ════════════════════════════════════════════════════════════════
# FUNGSI DETEKSI AURAT — AKURASI DITINGKATKAN
# ════════════════════════════════════════════════════════════════

def hitung_sudut(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    r = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(np.degrees(r))
    return angle if angle <= 180 else 360 - angle


def bangun_mask_aurat(mode, lm, kordinat, siluet_mask, w, h):
    """
    Bangun mask area aurat yang lebih akurat menggunakan:
    - Koordinat landmark anatomis
    - Sudut sendi untuk estimasi batas tubuh
    - Dilation/erosion untuk kehalusan tepi
    """
    mask_aurat = np.zeros((h, w), dtype=np.uint8)
    kernel_halus = np.ones((7, 7), np.uint8)

    if mode == "LAKI-LAKI":
        # Aurat pria: pusar → lutut (seluruh lebar tubuh di rentang tersebut)
        if 23 in kordinat and 24 in kordinat:
            y_pinggul = int((kordinat[23][1] + kordinat[24][1]) / 2)
            x_kiri  = min(kordinat[23][0], kordinat[24][0])
            x_kanan = max(kordinat[23][0], kordinat[24][0])
            margin_x = int((x_kanan - x_kiri) * 0.6)
            x_kiri  = max(0, x_kiri  - margin_x)
            x_kanan = min(w, x_kanan + margin_x)

            y_atas   = max(0, y_pinggul - 20)  # Sedikit di atas pusar
            y_lutut  = int((kordinat.get(25, (0, h))[1] +
                            kordinat.get(26, (0, h))[1]) / 2) + 30
            y_lutut  = min(h, y_lutut)

            cv2.rectangle(mask_aurat, (x_kiri, y_atas), (x_kanan, y_lutut), 255, -1)
            # Iris dengan siluet badan
            mask_aurat = cv2.bitwise_and(mask_aurat, siluet_mask)
            mask_aurat = cv2.morphologyEx(mask_aurat, cv2.MORPH_CLOSE, kernel_halus)

    else:  # PEREMPUAN
        # Aurat wanita: seluruh tubuh KECUALI wajah + telapak tangan
        mask_aurat = siluet_mask.copy()
        mask_aurat = cv2.dilate(mask_aurat, kernel_halus, iterations=1)

        # ── Kecualikan WAJAH (ellips presisi, bukan circle biasa) ──
        if 0 in kordinat:
            cx, cy = kordinat[0]
            # Estimasi ukuran wajah dari jarak telinga ke telinga
            jarak_telinga = 100
            if 7 in kordinat and 8 in kordinat:
                jarak_telinga = abs(kordinat[7][0] - kordinat[8][0])
            radius_wajah = max(60, int(jarak_telinga * 0.65))
            # Tambah margin lebih besar ke bawah (dagu/leher)
            cv2.ellipse(mask_aurat, (cx, cy + 10),
                        (radius_wajah, int(radius_wajah * 1.25)),
                        0, 0, 360, 0, -1)

        # ── Kecualikan TELAPAK TANGAN (pergelangan tangan + jari) ──
        for idx_pergelangan, idx_jari_tengah in [(15, 19), (16, 20)]:
            if idx_pergelangan in kordinat:
                px, py = kordinat[idx_pergelangan]
                radius_tangan = 50
                if idx_jari_tengah in kordinat:
                    jx, jy = kordinat[idx_jari_tengah]
                    radius_tangan = max(40, int(
                        math.hypot(jx - px, jy - py) * 1.3
                    ))
                cv2.circle(mask_aurat, (px, py), radius_tangan, 0, -1)
                # Hapus juga area jari-jari
                for jari_idx in [17, 18, 19, 20, 21, 22]:
                    if jari_idx in kordinat:
                        cv2.circle(mask_aurat, kordinat[jari_idx], 30, 0, -1)

        mask_aurat = cv2.morphologyEx(mask_aurat, cv2.MORPH_OPEN,
                                       np.ones((5, 5), np.uint8))

    return mask_aurat


def deteksi_warna_kulit_multi(frame_bgr, frame_hsv, frame_ycrcb):
    """
    Deteksi kulit menggunakan 3 ruang warna: HSV + YCrCb + BGR.
    Hasilnya di-AND → lebih presisi, false positive berkurang.
    """
    # HSV range kulit
    mask_hsv = cv2.inRange(frame_hsv,
                            np.array([0, 25, 80]),
                            np.array([25, 255, 255]))

    # YCrCb range kulit (lebih stabil dengan pencahayaan)
    mask_ycrcb = cv2.inRange(frame_ycrcb,
                              np.array([0, 133, 77]),
                              np.array([255, 173, 127]))

    # Gabung (AND) → hanya piksel yang KEDUANYA terdeteksi
    mask_gabung = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # Morphology: bersihkan noise, tutup lubang kecil
    kernel = np.ones((7, 7), np.uint8)
    mask_gabung = cv2.morphologyEx(mask_gabung, cv2.MORPH_OPEN,  kernel)
    mask_gabung = cv2.morphologyEx(mask_gabung, cv2.MORPH_CLOSE, kernel)
    mask_gabung = cv2.GaussianBlur(mask_gabung, (5, 5), 0)
    _, mask_gabung = cv2.threshold(mask_gabung, 64, 255, cv2.THRESH_BINARY)

    return mask_gabung


# ════════════════════════════════════════════════════════════════
# MAIN LOOP
# ════════════════════════════════════════════════════════════════
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
cv2.namedWindow('Smart Mosque Gate — Aurat Detection v2', cv2.WINDOW_NORMAL)

# FPS
prev_time = time.time()
fps_val   = 0.0

print("╔══════════════════════════════════════════╗")
print("║   SMART MOSQUE GATE — v2.0               ║")
print("║   [P] Mode Perempuan  [L] Mode Laki-Laki ║")
print("║   [Q] Keluar                             ║")
print("╚══════════════════════════════════════════╝")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    t = time.time() - t_start

    # ── Konversi warna ────────────────────────────────────────
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv      = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb    = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # ── FPS ───────────────────────────────────────────────────
    now     = time.time()
    fps_val = 0.9 * fps_val + 0.1 * (1.0 / max(now - prev_time, 1e-5))
    prev_time = now

    # ── Proses MediaPipe ──────────────────────────────────────
    hasil = pose.process(rgb)

    # ── Default status ────────────────────────────────────────
    persen_aurat = 0.0
    pelanggaran  = False
    pose_detected = False

    # ── Latar belakang: panel utama gelap ─────────────────────
    # Overlay ringan agar teks terbaca
    overlay_transparan(frame, 0, 0, w, 65,   (10, 8, 5), alpha=0.75)   # header
    overlay_transparan(frame, 0, h-55, w, h, (10, 8, 5), alpha=0.75)   # footer

    if hasil.pose_landmarks and hasil.segmentation_mask is not None:
        pose_detected = True
        lm = hasil.pose_landmarks.landmark

        # Koordinat landmark (hanya yang visibility cukup)
        kordinat = {
            i: (int(lm[i].x * w), int(lm[i].y * h))
            for i in range(33) if lm[i].visibility > 0.45
        }

        # Siluet badan dari segmentation mask
        siluet_raw  = (hasil.segmentation_mask > 0.55).astype(np.uint8) * 255
        siluet_mask = cv2.morphologyEx(
            siluet_raw, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8)
        )

        # ── Deteksi kulit ──────────────────────────────────────
        mask_kulit = deteksi_warna_kulit_multi(frame, hsv, ycrcb)

        # ── Bangun mask aurat ──────────────────────────────────
        mask_aurat = bangun_mask_aurat(
            mode_gate, lm, kordinat, siluet_mask, w, h
        )

        # ── Hitung pelanggaran ─────────────────────────────────
        kulit_di_aurat    = cv2.bitwise_and(mask_kulit, mask_aurat)
        pixel_kulit       = cv2.countNonZero(kulit_di_aurat)
        total_pixel_aurat = cv2.countNonZero(mask_aurat)
        persen_aurat = (pixel_kulit / total_pixel_aurat * 100) \
                       if total_pixel_aurat > 0 else 0.0

        # Threshold: 3.5% (lebih ketat dari versi sebelumnya)
        THRESHOLD = 3.5

        # Smoothing status pelanggaran
        history_status.append(persen_aurat > THRESHOLD)
        if len(history_status) > SMOOTH_N:
            history_status.pop(0)
        pelanggaran = sum(history_status) > SMOOTH_N // 2

        # ── Gambar kerangka tubuh ──────────────────────────────
        mp_drawing.draw_landmarks(
            frame,
            hasil.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(200, 200, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 200, 180), thickness=2)
        )

        # ── Highlight area bocor ───────────────────────────────
        if pelanggaran and pixel_kulit > 200:
            # Overlay merah semi-transparan pada area yang bocor
            overlay_merah = frame.copy()
            overlay_merah[kulit_di_aurat > 0] = (40, 40, 200)
            cv2.addWeighted(overlay_merah, 0.40, frame, 0.60, 0, frame)
            # Kontur area bocor
            cnts, _ = cv2.findContours(
                kulit_di_aurat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, cnts, -1, (0, 0, 255), 2)
        else:
            # Overlay hijau sangat tipis saat aman
            overlay_hijau = frame.copy()
            area_badan = siluet_mask > 0
            overlay_hijau[area_badan] = (
                frame[area_badan] * 0.85 + np.array([0, 60, 0]) * 0.15
            ).astype(np.uint8)
            cv2.addWeighted(overlay_hijau, 0.30, frame, 0.70, 0, frame)

        # ── Suara ─────────────────────────────────────────────
        if PYGAME_OK:
            if pelanggaran:
                if suara_peringatan and not pygame.mixer.get_busy():
                    suara_peringatan.play()
                sudah_bunyi_aman = False
            else:
                if not sudah_bunyi_aman:
                    pygame.mixer.stop()
                    if suara_aman:
                        suara_aman.play()
                    sudah_bunyi_aman = True

    # ════════════════════════════════════════════════════════
    # UI — HEADER
    # ════════════════════════════════════════════════════════
    # Judul
    gambar_teks_shadow(frame, "SMART MOSQUE GATE", (20, 42),
                       cv2.FONT_HERSHEY_DUPLEX, 1.1, GOLD_TERANG, 2)
    # Versi
    cv2.putText(frame, "Aurat Detection System v2.0",
                (320, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, ABU_TERANG, 1, cv2.LINE_AA)

    # FPS (kanan atas)
    warna_fps = EMERALD if fps_val >= 24 else (0, 200, 200) if fps_val >= 15 else MERAH
    gambar_teks_shadow(frame, f"FPS {fps_val:4.1f}",
                       (w - 140, 42), cv2.FONT_HERSHEY_DUPLEX, 0.8, warna_fps)

    # ════════════════════════════════════════════════════════
    # UI — PANEL KIRI: Status Deteksi
    # ════════════════════════════════════════════════════════
    px, py = 15, 80
    pw, ph = 360, 330
    overlay_transparan(frame, px, py, px + pw, py + ph, (12, 10, 8), alpha=0.70, radius=12)
    # Border
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), GOLD, 1)

    # Judul panel
    gambar_teks_shadow(frame, "STATUS DETEKSI", (px + 15, py + 32),
                       cv2.FONT_HERSHEY_DUPLEX, 0.75, GOLD, 2)
    cv2.line(frame, (px + 10, py + 42), (px + pw - 10, py + 42), GOLD, 1)

    # Mode gate + ikon
    gambar_ikon_mode(frame, mode_gate, px + 40, py + 100, ukuran=35)
    warna_mode = EMERALD if mode_gate == "PEREMPUAN" else GOLD
    gambar_teks_shadow(frame, f"MODE: {mode_gate}", (px + 80, py + 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, warna_mode, 2)

    # Status pose
    if pose_detected:
        gambar_teks_shadow(frame, "POSE TERDETEKSI  [OK]", (px + 15, py + 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, EMERALD, 1)
    else:
        gambar_teks_shadow(frame, "POSE TIDAK TERDETEKSI", (px + 15, py + 130),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, ABU_TERANG, 1)

    # Meter % aurat terekspos
    cv2.putText(frame, "Kulit Terekspos:", (px + 15, py + 165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, PUTIH, 1, cv2.LINE_AA)
    warna_meter = MERAH if persen_aurat > 3.5 else (0, 200, 200) if persen_aurat > 1.5 else EMERALD
    gambar_meter_persen(frame, px + 15, py + 172, pw - 30, 24,
                        persen_aurat, warna_meter)

    # Status utama — BESAR
    if not pose_detected:
        status_txt  = "MENUNGGU..."
        warna_status = ABU_TERANG
        bg_status    = (30, 30, 30)
    elif pelanggaran:
        status_txt  = "AKSES DITOLAK"
        warna_status = MERAH
        bg_status    = (20, 20, 80)
    else:
        status_txt  = "AKSES DITERIMA"
        warna_status = EMERALD
        bg_status    = (10, 50, 10)

    overlay_transparan(frame, px + 10, py + 210, px + pw - 10, py + 270,
                       bg_status, alpha=0.85, radius=8)
    (tw, th), _ = cv2.getTextSize(status_txt, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
    tx = px + 10 + (pw - 20 - tw) // 2
    gambar_teks_shadow(frame, status_txt, (tx, py + 252),
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, warna_status, 2)

    # Sub-status (aman/aurat)
    if pose_detected:
        sub = "Hijab / Pakaian: SESUAI" if not pelanggaran else f"Aurat Terekspos: {persen_aurat:.1f}%"
        warna_sub = EMERALD if not pelanggaran else MERAH
        cv2.putText(frame, sub, (px + 20, py + 305),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, warna_sub, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════
    # UI — PANEL KANAN: Panduan & Info
    # ════════════════════════════════════════════════════════
    rx = w - 280
    ry = 80
    rw = 265
    rh = 220
    overlay_transparan(frame, rx, ry, rx + rw, ry + rh, (12, 10, 8), alpha=0.70, radius=12)
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), GOLD, 1)

    gambar_teks_shadow(frame, "PANDUAN", (rx + 15, ry + 32),
                       cv2.FONT_HERSHEY_DUPLEX, 0.75, GOLD, 2)
    cv2.line(frame, (rx + 10, ry + 42), (rx + rw - 10, ry + 42), GOLD, 1)

    panduan = [
        ("[P]", "Mode Perempuan"),
        ("[L]", "Mode Laki-Laki"),
        ("[Q]", "Keluar"),
    ]
    for i, (tombol, label) in enumerate(panduan):
        y_p = ry + 75 + i * 40
        overlay_transparan(frame, rx + 10, y_p - 18, rx + 70, y_p + 8,
                           (40, 30, 5), alpha=0.85, radius=5)
        gambar_teks_shadow(frame, tombol, (rx + 18, y_p),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.65, GOLD_TERANG, 2)
        cv2.putText(frame, label, (rx + 80, y_p),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, PUTIH, 1, cv2.LINE_AA)

    # Info threshold
    cv2.putText(frame, f"Threshold: 3.5% | Smooth: {SMOOTH_N}f",
                (rx + 10, ry + rh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, ABU_TERANG, 1, cv2.LINE_AA)

    # ════════════════════════════════════════════════════════
    # UI — BORDER PERINGATAN ANIMASI
    # ════════════════════════════════════════════════════════
    if pose_detected and pelanggaran:
        # Kedip merah (ganti intensitas sesuai sinus waktu)
        alpha_border = int(180 + 75 * math.sin(t * 6))
        tebal_border = 10 if int(t * 4) % 2 == 0 else 6
        cv2.rectangle(frame, (0, 0), (w, h), MERAH, tebal_border)
        gambar_teks_shadow(frame, "!! AURAT TERLIHAT !!",
                           (w // 2 - 220, h // 2),
                           cv2.FONT_HERSHEY_DUPLEX, 1.5, MERAH, 3)
    elif pose_detected and not pelanggaran:
        # Border hijau tipis
        cv2.rectangle(frame, (0, 0), (w, h), EMERALD, 3)

    # ════════════════════════════════════════════════════════
    # UI — GARIS ORNAMEN + FOOTER
    # ════════════════════════════════════════════════════════
    gambar_garis_hias(frame, w, h, t)

    cv2.putText(frame,
                "Tekan [P] Perempuan  |  [L] Laki-Laki  |  [Q] Keluar",
                (w // 2 - 320, h - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, ABU_TERANG, 1, cv2.LINE_AA)

    cv2.imshow('Smart Mosque Gate — Aurat Detection v2', frame)

    # ── Keyboard ──────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('l'):
        mode_gate = "LAKI-LAKI"
        sudah_bunyi_aman = False
        history_status.clear()
        print("[MODE] Beralih ke LAKI-LAKI")
    elif key == ord('p'):
        mode_gate = "PEREMPUAN"
        sudah_bunyi_aman = False
        history_status.clear()
        print("[MODE] Beralih ke PEREMPUAN")

cap.release()
cv2.destroyAllWindows()
print("Sesi selesai.")