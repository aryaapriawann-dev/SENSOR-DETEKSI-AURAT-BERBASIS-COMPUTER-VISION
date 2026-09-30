"""
Aurat Detector Engine — Multi-Person Tracking & YOLOv8 Deep Learning
Mendeteksi banyak orang sekaligus secara real-time dan melacak posisi tubuh (melengket tanpa kedip).
"""
import os
import sys
import cv2  # type: ignore
import numpy as np  # type: ignore

site_pack = os.path.expanduser(r"~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages")
if os.path.exists(site_pack) and site_pack not in sys.path:
    sys.path.append(site_pack)

from ultralytics import YOLO  # type: ignore
from .config import YOLO_MODEL_PATH, BASE_DIR
from .tracker import PersonTracker


def hitung_persen_kulit(crop_bgr):
    """Menghitung persentase piksel kulit menggunakan segmentasi ganda HSV + YCrCb yang adaptif."""
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 6 or crop_bgr.shape[1] < 6:
        return 0.0
    
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)

    # Rentang warna kulit presisi (toleran terhadap ragam pencahayaan & skin tone)
    mask_hsv1 = cv2.inRange(hsv, np.array([0, 20, 50]), np.array([25, 240, 255]))
    mask_hsv2 = cv2.inRange(hsv, np.array([170, 20, 50]), np.array([180, 240, 255]))
    mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)

    mask_ycrcb = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))
    skin_mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)

    kernel = np.ones((3, 3), np.uint8)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)

    pixel_kulit = cv2.countNonZero(skin_mask)
    total_pixel = crop_bgr.shape[0] * crop_bgr.shape[1]
    return (pixel_kulit / float(total_pixel)) * 100.0 if total_pixel > 0 else 0.0


class AuratDetector:
    def __init__(self, model_aurat_path: str = YOLO_MODEL_PATH):
        # 1. Model Deteksi Manusia (Multi-Person Detector)
        person_model_path = os.path.join(BASE_DIR, "models", "yolov8n.pt")
        if not os.path.exists(person_model_path):
            person_model_path = "yolov8n.pt"
        
        print(f"[AuratDetector] Memuat Person Detector dari: {person_model_path}")
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

        # 1. Deteksi Semua Manusia (Multi-Person) menggunakan YOLOv8 Nano
        res_person = self.model_person.predict(
            source=frame,
            classes=[0],     # Class 0 = Person (COCO dataset)
            conf=0.50,       # Naikkan threshold agar bagian tubuh (misal tangan) tidak dideteksi orang terpisah
            iou=0.45,        # Filter Non-Maximum Suppression antar kotak
            verbose=False,
            imgsz=320
        )[0]

        if res_person and res_person.boxes:
            for pbox in res_person.boxes:
                px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                # Batasi koordinat ke dimensi frame
                px1 = max(0, px1)
                py1 = max(0, py1)
                px2 = min(w, px2)
                py2 = min(h, py2)

                bw = px2 - px1
                bh = py2 - py1

                # Abaikan kotak yang terlalu kecil (potongan tangan, objek parsial, noise)
                if bw < 75 or bh < 120:
                    continue

                aspect_ratio = bh / float(max(1, bw))

                # Default status
                status = "AMAN"
                label_txt = "Sesuai Syariat"
                conf_score = float(pbox.conf[0])
                detail_pelanggaran = []
                zones_pelanggaran = []

                if mode == "PEREMPUAN":
                    # ── A. PEMBAGIAN ZONA ADAPTIF (Full Body vs Upper Body/Webcam) ──
                    if aspect_ratio >= 2.0:
                        # Badan Penuh (Full body berdiri)
                        head_bottom = py1 + int(bh * 0.24)
                        neck_y1 = py1 + int(bh * 0.16)
                        neck_y2 = py1 + int(bh * 0.28)
                        arm_y1 = py1 + int(bh * 0.25)
                        arm_y2 = py1 + int(bh * 0.65)
                        arm_w = int(bw * 0.28)
                    else:
                        # Setengah Badan / Duduk di Webcam
                        head_bottom = py1 + int(bh * 0.38)
                        neck_y1 = py1 + int(bh * 0.26)
                        neck_y2 = py1 + int(bh * 0.44)
                        arm_y1 = py1 + int(bh * 0.36)
                        arm_y2 = py1 + int(bh * 0.85)
                        arm_w = int(bw * 0.30)

                    # Batasi koordinat zona ke dimensi frame
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

                    # ── B. PEMERIKSAAN KEPALA (Rambut & Hijab Syar'i via YOLO Model) ──
                    head_crop = frame[py1:head_bottom, px1:px2]
                    non_hijab_found = False
                    hijab_syari_found = False
                    non_syari_found = False
                    best_conf = 0.0

                    if head_crop.size > 0:
                        res_aurat = self.model_aurat.predict(
                            source=head_crop,
                            conf=0.28,
                            verbose=False,
                            imgsz=224
                        )[0]

                        if res_aurat and res_aurat.boxes:
                            for abox in res_aurat.boxes:
                                a_cls = int(abox.cls[0])
                                a_conf = float(abox.conf[0])
                                a_label = self.model_aurat.names[a_cls]

                                if a_label == "Non hijab" and a_conf > 0.32:
                                    non_hijab_found = True
                                    best_conf = max(best_conf, a_conf)
                                elif "HijabSyar" in a_label and a_conf > 0.30:
                                    hijab_syari_found = True
                                    best_conf = max(best_conf, a_conf)
                                elif "Non Syar" in a_label and a_conf > 0.30:
                                    non_syari_found = True
                                    best_conf = max(best_conf, a_conf)

                    # ── C. PEMERIKSAAN LEHER & DADA ──
                    neck_crop = frame[neck_y1:neck_y2, neck_x1:neck_x2]
                    kulit_leher = hitung_persen_kulit(neck_crop)

                    # ── D. PEMERIKSAAN LENGAN TANGAN (Kiri & Kanan) ──
                    arm_l_crop = frame[arm_y1:arm_y2, arm_l_x1:arm_l_x2]
                    arm_r_crop = frame[arm_y1:arm_y2, arm_r_x1:arm_r_x2]
                    kulit_lengan_l = hitung_persen_kulit(arm_l_crop)
                    kulit_lengan_r = hitung_persen_kulit(arm_r_crop)

                    # ── E. EVALUASI KAIDAH SYARIAT PEREMPUAN ──
                    # 1. Cek Rambut
                    if non_hijab_found:
                        detail_pelanggaran.append("Rambut Terbuka")
                        zones_pelanggaran.append((px1, py1, px2, head_bottom, "RAMBUT"))

                    # 2. Cek Leher (Mukena/hijab wajib menutup leher sempurna)
                    if non_syari_found:
                        detail_pelanggaran.append("Leher Terbuka")
                        zones_pelanggaran.append((neck_x1, neck_y1, neck_x2, neck_y2, "LEHER"))
                    elif kulit_leher > 14.0 and not hijab_syari_found:
                        detail_pelanggaran.append("Leher Terbuka")
                        zones_pelanggaran.append((neck_x1, neck_y1, neck_x2, neck_y2, "LEHER"))

                    # 3. Cek Lengan Tangan (Mukena/baju panjang wajib menutup pergelangan tangan)
                    if kulit_lengan_l > 16.0:
                        detail_pelanggaran.append("Lengan Kiri")
                        zones_pelanggaran.append((arm_l_x1, arm_y1, arm_l_x2, arm_y2, "LENGAN"))
                    if kulit_lengan_r > 16.0:
                        detail_pelanggaran.append("Lengan Kanan")
                        zones_pelanggaran.append((arm_r_x1, arm_y1, arm_r_x2, arm_y2, "LENGAN"))

                    # 4. Keputusan Akhir Status & Label
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
                    elif hijab_syari_found:
                        status = "AMAN"
                        label_txt = f"Mukena/Hijab Syar'i ({best_conf*100:.0f}%)"
                    elif not hijab_syari_found and not non_syari_found:
                        status = "PELANGGARAN"
                        label_txt = "Aurat: Belum Berhijab"
                        zones_pelanggaran.append((px1, py1, px2, head_bottom, "RAMBUT"))
                    else:
                        status = "AMAN"
                        label_txt = "Pakaian Sesuai Syariat"

                else:  # LAKI-LAKI
                    # Batas Aurat Laki-Laki: Antara pusar dan lutut
                    if aspect_ratio >= 1.8:
                        thigh_y1 = max(0, min(h, py1 + int(bh * 0.48)))
                        thigh_y2 = max(thigh_y1 + 5, min(h, py1 + int(bh * 0.72)))
                        thigh_x1 = max(0, min(w, px1 + int(bw * 0.20)))
                        thigh_x2 = max(thigh_x1 + 5, min(w, px2 - int(bw * 0.20)))
                        thigh_crop = frame[thigh_y1:thigh_y2, thigh_x1:thigh_x2]
                        kulit_paha = hitung_persen_kulit(thigh_crop)

                        if kulit_paha > 25.0:
                            status = "PELANGGARAN"
                            label_txt = "Aurat: Paha/Lutut Terbuka"
                            detail_pelanggaran.append("Paha Terbuka")
                            zones_pelanggaran.append((thigh_x1, thigh_y1, thigh_x2, thigh_y2, "PAHA"))
                        else:
                            status = "AMAN"
                            label_txt = f"Pria Sesuai Kaidah ({conf_score*100:.0f}%)"
                    else:
                        status = "AMAN"
                        label_txt = f"Pria Sesuai Kaidah ({conf_score*100:.0f}%)"

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
