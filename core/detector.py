"""
Aurat Detector Engine — Multi-Person Tracking & YOLOv8 Deep Learning
Mendeteksi banyak orang sekaligus secara real-time dan melacak posisi tubuh (melengket tanpa kedip).
"""
import os
import sys
from typing import Any
import cv2  # type: ignore
import numpy as np  # type: ignore

site_pack = os.path.expanduser(r"~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages")
if os.path.exists(site_pack) and site_pack not in sys.path:
    sys.path.append(site_pack)

from ultralytics import YOLO  # type: ignore
from .config import YOLO_MODEL_PATH, BASE_DIR
from .tracker import PersonTracker


def hitung_persen_kulit(crop_bgr, mask_exclude=None):
    """
    Menghitung persentase piksel kulit asli manusia dengan filter presisi tinggi:
    - HSV Hue ketat 0..15 & 172..180 (membuang kain seragam krem, cokelat, khaki yang H=18..35).
    - Kaidah biologis hemoglobin darah manusia: R > G > B dan (R - G) >= 14.
    - YCrCb chrominance spesifik: Cr in [135, 175], Cb in [85, 127].
    - Mendukung mask_exclude untuk mengecualikan tangan/lengan yang bersedekap/menutupi perut.
    """
    pct, _ = hitung_persen_kulit_detail(crop_bgr, mask_exclude=mask_exclude)
    return pct


def hitung_persen_kulit_detail(crop_bgr, mask_exclude=None):
    """Mengembalikan (persentase_kulit, rasio_blob_kulit_terbesar_kontinyu)."""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 6 or crop_bgr.shape[1] < 6:
        return 0.0, 0.0
    
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)

    # 1. Filter HSV Ketat (Kulit manusia asli: Hue 0..15 & 172..180, bukan kain seragam krem/cokelat H=18..35)
    mask_hsv1 = cv2.inRange(hsv, np.array([0, 32, 60]), np.array([15, 255, 255]))
    mask_hsv2 = cv2.inRange(hsv, np.array([172, 32, 60]), np.array([180, 255, 255]))
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    # 2. Filter YCrCb (Karakteristik melanin & hemoglobin)
    mask_ycrcb = cv2.inRange(ycrcb, np.array([50, 135, 85]), np.array([255, 175, 127]))
    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    # 3. Kaidah Biologis Hemoglobin (Membedakan kulit vs kain cokelat/krem)
    b = crop_bgr[:, :, 0].astype(np.int32)
    g = crop_bgr[:, :, 1].astype(np.int32)
    r = crop_bgr[:, :, 2].astype(np.int32)
    mask_bio = ((r > g) & (g > b) & ((r - g) >= 14) & (r >= 85)).astype(np.uint8) * 255
    skin_mask = cv2.bitwise_and(skin_mask, mask_bio)

    # 4. Eksklusi Area Lengan & Tangan jika ada
    if mask_exclude is not None and mask_exclude.shape[:2] == skin_mask.shape[:2]:
        skin_mask[mask_exclude > 0] = 0
        valid_pixels = int(np.count_nonzero(mask_exclude == 0))
    else:
        valid_pixels = crop_bgr.shape[0] * crop_bgr.shape[1]

    if valid_pixels <= 0:
        return 0.0, 0.0

    kernel = np.ones((3, 3), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    pixel_kulit = cv2.countNonZero(skin_mask)
    pct = (pixel_kulit / float(valid_pixels)) * 100.0

    # 5. Hitung blob kulit kontinyu terbesar (perut terbuka membentuk 1 bidang kulit yang luas)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skin_mask)
    largest_blob = max(stats[1:, cv2.CC_STAT_AREA]) if num_labels > 1 else 0
    blob_ratio = largest_blob / float(valid_pixels)

    return pct, blob_ratio


class AuratDetector:
    def __init__(self, model_aurat_path: str = YOLO_MODEL_PATH):
        # 1. Model Deteksi Manusia & Pose (YOLOv8-Pose)
        person_model_path = os.path.join(BASE_DIR, "models", "yolov8n-pose.pt")
        if not os.path.exists(person_model_path):
            person_model_path = os.path.join(BASE_DIR, "models", "yolov8n.pt")
        if not os.path.exists(person_model_path):
            person_model_path = "yolov8n-pose.pt"
        
        print(f"[AuratDetector] Memuat Person & Pose Detector dari: {person_model_path}")
        self.model_person = YOLO(person_model_path)

        # 2. Model Klasifikasi Aurat / Hijab
        if not os.path.exists(model_aurat_path):
            model_aurat_path = os.path.join(BASE_DIR, "models", "aurat_best.pt")
        
        print(f"[AuratDetector] Memuat Model Aurat dari: {model_aurat_path}")
        self.model_aurat = YOLO(model_aurat_path)

        # 3. Multi-Person Tracker (Menjaga bounding box melengket & tidak hilang-hilang)
        self.tracker = PersonTracker(max_missing_frames=5, smooth_factor=0.60)
        print("[AuratDetector] Multi-Person Tracking Engine siap!")

    def process_frame(self, frame, mode: str = "PEREMPUAN"):
        """
        Memproses frame untuk BANYAK ORANG sekaligus dengan KAIDAH LENGKAP:
        - Mode PEREMPUAN: Memeriksa Rambut, Leher, dan Lengan Tangan.
        - Mode LAKI-LAKI: Memeriksa aurat antara pusar hingga lutut.
        """
        h, w, _ = frame.shape
        raw_boxes = []
        raw_statuses = []

        # 1. Deteksi Semua Manusia (Multi-Person) menggunakan YOLOv8 Nano Pose
        pred_person: Any = self.model_person.predict(
            source=frame,
            classes=[0],     # Class 0 = Person (COCO dataset)
            conf=0.38,       # Sensitivitas optimal: mendeteksi manusia penuh maupun close-up setengah badan
            iou=0.45,        # Filter Non-Maximum Suppression antar kotak
            verbose=False,
            imgsz=320
        )
        res_person: Any = pred_person[0] if pred_person else None

        if res_person is not None and getattr(res_person, "boxes", None) is not None:
            has_kpts = hasattr(res_person, "keypoints") and res_person.keypoints is not None
            for p_idx, pbox in enumerate(res_person.boxes):
                px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                # Batasi koordinat ke dimensi frame
                px1 = max(0, px1)
                py1 = max(0, py1)
                px2 = min(w, px2)
                py2 = min(h, py2)

                bw = px2 - px1
                bh = py2 - py1

                # Abaikan kotak yang terlalu kecil (objek parsial / noise jauh)
                if bw < 60 or bh < 80:
                    continue

                aspect_ratio = bh / float(max(1, bw))

                # ── EKSTRAKSI KEYPOINTS POSE ANATOMI TUBUH ──
                # 5: Bahu Kiri, 6: Bahu Kanan
                # 11: Pinggul Kiri, 12: Pinggul Kanan
                # 13: Lutut Kiri, 14: Lutut Kanan
                sh_y = None
                sh_x1 = px1 + int(bw * 0.20)
                sh_x2 = px2 - int(bw * 0.20)
                hip_y = None
                knee_y = None
                kpts: Any = None
                kconf: Any = None

                if has_kpts and len(res_person.keypoints) > p_idx:
                    kpts = res_person.keypoints.xy[p_idx].cpu().numpy()
                    kconf = res_person.keypoints.conf[p_idx].cpu().numpy() if res_person.keypoints.conf is not None else None

                    if kpts is not None and kconf is not None and len(kpts) >= 17:
                        valid_sh = [j for j in [5, 6] if kconf[j] > 0.35]
                        if valid_sh:
                            sh_y = int(np.mean([kpts[j][1] for j in valid_sh]))
                            if len(valid_sh) == 2:
                                sh_x1 = max(0, int(min(kpts[5][0], kpts[6][0])))
                                sh_x2 = min(w, int(max(kpts[5][0], kpts[6][0])))

                        valid_hip = [j for j in [11, 12] if kconf[j] > 0.35]
                        if valid_hip:
                            hip_y = int(np.mean([kpts[j][1] for j in valid_hip]))

                        valid_knee = [j for j in [13, 14] if kconf[j] > 0.35]
                        if valid_knee:
                            knee_y = int(np.mean([kpts[j][1] for j in valid_knee]))

                # Default status
                status = "AMAN"
                label_txt = "Sesuai Syariat"
                conf_score = float(pbox.conf[0])
                detail_pelanggaran = []
                zones_pelanggaran = []

                if mode == "PEREMPUAN":
                    # ── A. PEMBAGIAN ZONA ADAPTIF DENGAN BANTUAN POSE ──
                    if sh_y is not None:
                        head_bottom = min(h, max(py1 + 20, sh_y))
                        neck_y1 = max(0, sh_y - int((sh_y - py1) * 0.38))
                        neck_y2 = min(h, sh_y + 15)
                        arm_y1 = sh_y
                        arm_y2 = min(h, sh_y + int(bh * 0.50))
                    elif aspect_ratio >= 2.0:
                        head_bottom = py1 + int(bh * 0.24)
                        neck_y1 = py1 + int(bh * 0.16)
                        neck_y2 = py1 + int(bh * 0.28)
                        arm_y1 = py1 + int(bh * 0.25)
                        arm_y2 = py1 + int(bh * 0.65)
                    else:
                        head_bottom = py1 + int(bh * 0.38)
                        neck_y1 = py1 + int(bh * 0.26)
                        neck_y2 = py1 + int(bh * 0.44)
                        arm_y1 = py1 + int(bh * 0.36)
                        arm_y2 = py1 + int(bh * 0.85)

                    arm_w = int(bw * 0.28)
                    head_bottom = min(h, max(py1 + 20, head_bottom))
                    neck_y1 = max(0, min(h, neck_y1))
                    neck_y2 = max(neck_y1 + 5, min(h, neck_y2))
                    neck_x1 = max(0, min(w, px1 + int(bw * 0.28)))
                    neck_x2 = max(neck_x1 + 5, min(w, px1 + int(bw * 0.72)))

                    arm_y1 = max(0, min(h, arm_y1))
                    arm_y2 = max(arm_y1 + 10, min(h, arm_y2))
                    arm_l_x1 = max(0, px1)
                    arm_l_x2 = min(w, px1 + arm_w)
                    arm_r_x1 = max(0, px2 - arm_w)
                    arm_r_x2 = min(w, px2)

                    # ── B. PEMERIKSAAN KEPALA & BUSANA (YOLO Model 5-Kelas: Cadar, Mukena, Hijab Syar'i, Non Syar-i, Non Hijab) ──
                    # Menggunakan area tubuh bagian atas (kepala hingga dada) agar model dapat mengenali mukena dan jilbab syar'i yang menutup dada
                    upper_bottom = min(h, py1 + int(bh * 0.70)) if sh_y is None else min(h, max(sh_y + 40, py1 + int(bh * 0.55)))
                    upper_crop = frame[py1:upper_bottom, px1:px2]
                    head_crop = frame[py1:head_bottom, px1:px2]
                    cadar_found = False
                    mukena_found = False
                    hijab_syari_found = False
                    non_syari_found = False
                    non_hijab_found = False
                    best_conf = 0.0

                    if upper_crop.size > 0:
                        pred_aurat: Any = self.model_aurat.predict(
                            source=upper_crop,
                            conf=0.25,
                            verbose=False,
                            imgsz=224
                        )
                        res_aurat: Any = pred_aurat[0] if pred_aurat else None

                        if res_aurat is not None and getattr(res_aurat, "boxes", None) is not None:
                            for abox in res_aurat.boxes:
                                a_cls = int(abox.cls[0])
                                a_conf = float(abox.conf[0])
                                a_label = self.model_aurat.names.get(a_cls, str(a_cls))
                                l_str = a_label.lower()

                                # Deteksi 5 Kelas Model YOLO
                                if "cadar" in l_str or "niqab" in l_str:
                                    cadar_found = True
                                    best_conf = max(best_conf, a_conf)
                                elif "mukena" in l_str:
                                    mukena_found = True
                                    best_conf = max(best_conf, a_conf)
                                elif ("hijabsyar" in l_str or "syar-i" in l_str or "syari" in l_str) and "non" not in l_str:
                                    hijab_syari_found = True
                                    best_conf = max(best_conf, a_conf)
                                elif "non syar" in l_str and a_conf > 0.30:
                                    non_syari_found = True
                                    best_conf = max(best_conf, a_conf)
                                elif "non hijab" in l_str and a_conf > 0.30:
                                    non_hijab_found = True
                                    best_conf = max(best_conf, a_conf)

                        # ── DETEKSI BIOMETRIK CADAR / NIQAB (Penutup Wajah Sesuai Syariat) ──
                        # Jika seseorang memakai cadar hitam/gelap, bagian bawah wajah (hidung, mulut, dagu)
                        # tertutup kain secara rapat sehingga persentase kulit mendekati 0%.
                        hh, hw = head_crop.shape[:2]
                        if hh >= 20 and hw >= 20:
                            # Titik acuan mata jika tersedia dari pose
                            eye_y = None
                            head_cx = px1 + int(bw * 0.5)
                            if kpts is not None and kconf is not None and len(kpts) >= 3:
                                if float(kconf[1]) > 0.25 and float(kconf[2]) > 0.25:
                                    eye_y = int((float(kpts[1][1]) + float(kpts[2][1])) / 2.0)
                                    head_cx = int((float(kpts[1][0]) + float(kpts[2][0])) / 2.0)

                            if eye_y is not None and eye_y > py1 and eye_y < head_bottom:
                                # Area cadar: tepat di bawah mata s.d. batas bahu/dagu
                                cz_y1 = eye_y + 6
                                cz_y2 = head_bottom
                            else:
                                cz_y1 = py1 + int(hh * 0.42)
                                cz_y2 = head_bottom

                            # Lebar area cadar di tengah wajah
                            cz_w_half = max(15, int(bw * 0.18))
                            cz_x1 = max(0, min(w, head_cx - cz_w_half))
                            cz_x2 = max(cz_x1 + 10, min(w, head_cx + cz_w_half))

                            cadar_crop = frame[cz_y1:cz_y2, cz_x1:cz_x2]
                            kulit_cadar, _ = hitung_persen_kulit_detail(cadar_crop)

                            # Verifikasi bagian atas kepala (dahi & penutup rambut)
                            top_y1 = py1
                            top_y2 = cz_y1
                            top_crop = frame[top_y1:top_y2, cz_x1:cz_x2]
                            kulit_top, _ = hitung_persen_kulit_detail(top_crop)

                            # Jika wajah bawah tertutup rapat (< 7% kulit) dan dahi/kepala tertutup (< 8% kulit)
                            if kulit_cadar < 7.0 and kulit_top < 8.0:
                                cadar_found = True
                                # Cadar 100% syar'i: batalkan salah deteksi YOLO jika sempat mengira rambut terbuka
                                non_hijab_found = False
                                non_syari_found = False

                    # ── C. PEMERIKSAAN LEHER & DADA ──
                    neck_crop = frame[neck_y1:neck_y2, neck_x1:neck_x2]
                    kulit_leher = 0.0 if (cadar_found or mukena_found) else hitung_persen_kulit(neck_crop)

                    # ── D. PEMERIKSAAN LENGAN TANGAN (Kiri & Kanan) ──
                    # Kaidah Syariat: Telapak tangan & pergelangan tangan BUKAN aurat wanita.
                    # Kita mengecualikan pergelangan tangan (Keypoint 9 & 10) agar tidak terjadi false alarm.
                    arm_l_crop = frame[arm_y1:arm_y2, arm_l_x1:arm_l_x2]
                    arm_r_crop = frame[arm_y1:arm_y2, arm_r_x1:arm_r_x2]

                    mask_exclude_l = np.zeros(arm_l_crop.shape[:2], dtype=np.uint8)
                    mask_exclude_r = np.zeros(arm_r_crop.shape[:2], dtype=np.uint8)

                    if kpts is not None and kconf is not None and len(kpts) >= 17:
                        try:
                            # Wrist Kiri (9)
                            if float(kconf[9]) > 0.20:
                                wx_l = int(float(kpts[9][0])) - arm_l_x1
                                wy_l = int(float(kpts[9][1])) - arm_y1
                                cv2.circle(mask_exclude_l, (wx_l, wy_l), 38, (255, 255, 255), -1)
                                cv2.circle(mask_exclude_l, (wx_l, wy_l + 25), 32, (255, 255, 255), -1)
                            # Wrist Kanan (10)
                            if float(kconf[10]) > 0.20:
                                wx_r = int(float(kpts[10][0])) - arm_r_x1
                                wy_r = int(float(kpts[10][1])) - arm_y1
                                cv2.circle(mask_exclude_r, (wx_r, wy_r), 38, (255, 255, 255), -1)
                                cv2.circle(mask_exclude_r, (wx_r, wy_r + 25), 32, (255, 255, 255), -1)
                        except Exception:
                            pass

                    kulit_lengan_l, blob_l = hitung_persen_kulit_detail(arm_l_crop, mask_exclude=mask_exclude_l)
                    kulit_lengan_r, blob_r = hitung_persen_kulit_detail(arm_r_crop, mask_exclude=mask_exclude_r)

                    # ── E. EVALUASI KAIDAH SYARIAT PEREMPUAN ──
                    if non_hijab_found and not cadar_found and not mukena_found:
                        detail_pelanggaran.append("Rambut Terbuka")
                        zones_pelanggaran.append((px1, py1, px2, head_bottom, "RAMBUT"))

                    if non_syari_found and not cadar_found and not mukena_found:
                        detail_pelanggaran.append("Leher Terbuka")
                        zones_pelanggaran.append((neck_x1, neck_y1, neck_x2, neck_y2, "LEHER"))
                    elif kulit_leher > 14.0 and not hijab_syari_found and not cadar_found and not mukena_found:
                        detail_pelanggaran.append("Leher Terbuka")
                        zones_pelanggaran.append((neck_x1, neck_y1, neck_x2, neck_y2, "LEHER"))

                    # Lengan harus membentuk bidang kulit terbuka yang signifikan (di atas pergelangan tangan)
                    if kulit_lengan_l > 22.0 and blob_l > 0.16:
                        detail_pelanggaran.append("Lengan Kiri")
                        zones_pelanggaran.append((arm_l_x1, arm_y1, arm_l_x2, arm_y2, "LENGAN"))
                    if kulit_lengan_r > 22.0 and blob_r > 0.16:
                        detail_pelanggaran.append("Lengan Kanan")
                        zones_pelanggaran.append((arm_r_x1, arm_y1, arm_r_x2, arm_y2, "LENGAN"))

                    if len(detail_pelanggaran) > 0:
                        status = "PELANGGARAN"
                        has_rambut = "Rambut Terbuka" in detail_pelanggaran
                        has_leher = "Leher Terbuka" in detail_pelanggaran
                        has_lengan = any("Lengan" in d for d in detail_pelanggaran)

                        if has_rambut and has_lengan:
                            label_txt = "Aurat: Rambut & Lengan Terbuka"
                        elif has_leher and has_lengan:
                            label_txt = "Aurat: Leher & Lengan Terbuka"
                        elif has_rambut:
                            label_txt = f"Aurat: Rambut Terbuka ({best_conf*100:.0f}%)" if best_conf > 0 else "Aurat: Rambut Terbuka"
                        elif has_leher:
                            label_txt = "Aurat: Leher Terbuka"
                        elif has_lengan:
                            label_txt = "Aurat: Lengan Terbuka"
                        else:
                            label_txt = f"Aurat: {detail_pelanggaran[0]}"
                    elif cadar_found:
                        status = "AMAN"
                        label_txt = "Bercadar / Niqab Syar'i (Tertutup Sempurna)"
                    elif mukena_found:
                        status = "AMAN"
                        label_txt = "Mukena Sholat (Tertutup Sempurna)"
                    elif hijab_syari_found:
                        status = "AMAN"
                        label_txt = f"Mukena/Hijab Syar'i ({best_conf*100:.0f}%)"
                    elif not non_hijab_found and kulit_leher < 12.0:
                        status = "AMAN"
                        label_txt = "Hijab Sesuai Syariat"
                    else:
                        status = "PELANGGARAN"
                        label_txt = "Aurat: Belum Berhijab"
                        zones_pelanggaran.append((px1, py1, px2, head_bottom, "RAMBUT"))

                else:  # LAKI-LAKI (Kaidah Syariat Islam: Batas aurat dari pusar s.d. lutut)
                    if sh_y is not None:
                        # Wajah & Kepala berada di ATAS sh_y sehingga WAJAH TIDAK AKAN PERNAH disangka dada/perut
                        torso_w1 = max(px1, sh_x1)
                        torso_w2 = min(w, sh_x2)
                        if torso_w2 - torso_w1 < 25:
                            torso_w1 = px1 + int(bw * 0.20)
                            torso_w2 = px2 - int(bw * 0.20)

                        if hip_y is not None and hip_y > sh_y + 30:
                            # Kasus: Torso bawah dan pusar masuk frame kamera (badan setengah/penuh)
                            torso_h = hip_y - sh_y
                            belly_y1 = max(0, min(h, sh_y + int(torso_h * 0.42)))
                            belly_y2 = max(belly_y1 + 10, min(h, hip_y + int(torso_h * 0.10)))

                            belly_crop = frame[belly_y1:belly_y2, torso_w1:torso_w2]

                            # ── EKSKLUSI TANGAN / LENGAN BAWAH DI DEPAN PERUT ──
                            # Tangan bukan aurat pria. Jika bersedekap/menaruh tangan di perut,
                            # area tangan dieksklusi agar tidak disangka perut terbuka.
                            mask_exclude = np.zeros(belly_crop.shape[:2], dtype=np.uint8)
                            if kpts is not None and kconf is not None and len(kpts) >= 17:
                                for elbow_idx, wrist_idx in [(7, 9), (8, 10)]:
                                    try:
                                        if float(kconf[wrist_idx]) > 0.25:
                                            wx = int(float(kpts[wrist_idx][0])) - torso_w1
                                            wy = int(float(kpts[wrist_idx][1])) - belly_y1
                                            cv2.circle(mask_exclude, (wx, wy), 36, (255, 255, 255), -1)
                                            cv2.circle(mask_exclude, (wx, wy + 20), 30, (255, 255, 255), -1)
                                            if float(kconf[elbow_idx]) > 0.25:
                                                ex = int(float(kpts[elbow_idx][0])) - torso_w1
                                                ey = int(float(kpts[elbow_idx][1])) - belly_y1
                                                cv2.line(mask_exclude, (ex, ey), (wx, wy), (255, 255, 255), 32)
                                    except Exception:
                                        pass

                            kulit_perut, blob_ratio = hitung_persen_kulit_detail(belly_crop, mask_exclude=mask_exclude)

                            # Verifikasi perut telanjang asli:
                            # 1. Persentase kulit di luar tangan harus tinggi (> 24%)
                            # 2. Harus membentuk 1 bidang kulit kontinyu yang luas (blob_ratio > 0.16)
                            if kulit_perut > 24.0 and blob_ratio > 0.16:
                                detail_pelanggaran.append("Pusar/Perut Terbuka")
                                zones_pelanggaran.append((torso_w1, belly_y1, torso_w2, belly_y2, "PUSAR/PERUT"))

                            # Periksa area dada atas (hanya jika tanpa baju)
                            chest_y1 = max(0, min(h, sh_y + 10))
                            chest_y2 = max(chest_y1 + 10, min(h, sh_y + int(torso_h * 0.40)))
                            chest_crop = frame[chest_y1:chest_y2, torso_w1:torso_w2]
                            kulit_dada, blob_dada = hitung_persen_kulit_detail(chest_crop)
                            if kulit_dada > 35.0 and blob_dada > 0.20:
                                detail_pelanggaran.append("Dada Terbuka")
                                zones_pelanggaran.append((torso_w1, chest_y1, torso_w2, chest_y2, "DADA"))

                        else:
                            # Kasus: Close-up webcam (Hanya kepala & bahu/dada atas).
                            # Pinggul/Pusar berada di bawah jangkauan kamera, jadi pusar TIDAK BISA dideteksi terbuka.
                            # Hanya deteksi jika BERTELANJANG DADA sama sekali (tanpa kaos):
                            chest_y1 = max(0, min(h, sh_y + 15))
                            chest_y2 = max(chest_y1 + 10, min(h, py2 - 5))
                            if (chest_y2 - chest_y1) > 25 and (torso_w2 - torso_w1) > 30:
                                chest_crop = frame[chest_y1:chest_y2, torso_w1:torso_w2]
                                kulit_dada, blob_dada = hitung_persen_kulit_detail(chest_crop)
                                if kulit_dada > 40.0 and blob_dada > 0.25:
                                    detail_pelanggaran.append("Dada Terbuka")
                                    zones_pelanggaran.append((torso_w1, chest_y1, torso_w2, chest_y2, "DADA"))

                        # ── Pemeriksaan Paha / Lutut (Jika tampak) ──
                        if hip_y is not None and knee_y is not None and knee_y > hip_y + 20:
                            thigh_y1 = hip_y
                            thigh_y2 = min(h, knee_y)
                            thigh_w1 = px1 + int(bw * 0.16)
                            thigh_w2 = px2 - int(bw * 0.16)
                            thigh_crop = frame[thigh_y1:thigh_y2, thigh_w1:thigh_w2]
                            kulit_paha, blob_paha = hitung_persen_kulit_detail(thigh_crop)
                            if kulit_paha > 25.0 and blob_paha > 0.18:
                                detail_pelanggaran.append("Paha/Lutut Terbuka")
                                zones_pelanggaran.append((thigh_w1, thigh_y1, thigh_w2, thigh_y2, "PAHA"))

                    elif aspect_ratio >= 1.6:
                        # Fallback jika keypoints tidak terbaca tapi badan berdiri penuh
                        belly_y1 = max(0, min(h, py1 + int(bh * 0.40)))
                        belly_y2 = max(belly_y1 + 10, min(h, py1 + int(bh * 0.60)))
                        belly_crop = frame[belly_y1:belly_y2, px1 + int(bw * 0.20): px2 - int(bw * 0.20)]
                        if hitung_persen_kulit(belly_crop) > 22.0:
                            detail_pelanggaran.append("Pusar/Perut Terbuka")
                            zones_pelanggaran.append((px1 + int(bw * 0.20), belly_y1, px2 - int(bw * 0.20), belly_y2, "PUSAR/PERUT"))

                    # ── KEPUTUSAN STATUS AKHIR PRIA ──
                    if len(detail_pelanggaran) > 0:
                        status = "PELANGGARAN"
                        has_pusar = "Pusar/Perut Terbuka" in detail_pelanggaran
                        has_dada = "Dada Terbuka" in detail_pelanggaran
                        has_paha = "Paha/Lutut Terbuka" in detail_pelanggaran

                        if (has_pusar or has_dada) and has_paha:
                            label_txt = "Aurat Pria: Perut & Paha Terbuka"
                        elif has_pusar:
                            label_txt = "Aurat Pria: Pusar/Perut Terbuka"
                        elif has_dada:
                            label_txt = "Aurat Pria: Dada Terbuka"
                        elif has_paha:
                            label_txt = "Aurat Pria: Paha Terbuka"
                        else:
                            label_txt = f"Aurat Pria: {detail_pelanggaran[0]}"
                    else:
                        status = "AMAN"
                        label_txt = f"Pria Sesuai Syariat ({conf_score*100:.0f}%)"

                raw_boxes.append([px1, py1, px2, py2])
                raw_statuses.append((status, label_txt, conf_score, detail_pelanggaran, zones_pelanggaran))

        # 3. Update Multi-Person Tracker (Smoothing & Persistence)
        tracked_persons = self.tracker.update(raw_boxes, raw_statuses)

        # 4. Hitung Statistik Gate Keseluruhan
        total_orang = len(tracked_persons)
        jumlah_pelanggar = sum(1 for p in tracked_persons if p["is_pelanggaran"])
        ada_pelanggaran = jumlah_pelanggar > 0

        if total_orang == 0:
            alasan_gate = "Menunggu Orang di Depan Gerbang..."
        elif ada_pelanggaran:
            alasan_gate = f"{jumlah_pelanggar} dari {total_orang} Orang Belum Menutup Aurat"
        else:
            alasan_gate = f"Semua ({total_orang} Orang) Sesuai Kaidah Syariat"

        return ada_pelanggaran, alasan_gate, tracked_persons, total_orang

    detect = process_frame
