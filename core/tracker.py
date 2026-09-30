"""
Lightweight Multi-Person Tracker with Bounding Box Smoothing & Duplicate Suppression
Membuat kotak deteksi melengket mulus di tubuh manusia tanpa jitter, tanpa duplikat, dan akurat.
"""
import os
import sys

site_pack = os.path.expanduser(r"~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages")
if os.path.exists(site_pack) and site_pack not in sys.path:
    sys.path.append(site_pack)

import numpy as np  # type: ignore

def hitung_iou(boxA, boxB):
    """Menghitung Intersection over Union (IoU) antara dua box."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def saring_kotak_duplikat(boxes, statuses):
    """
    Menyaring kotak ganda / tumpang-tindih pada orang yang sama.
    Jika satu kotak (misal tangan/badan atas) berada di dalam kotak tubuh orang yang sama,
    kotak kecil tersebut akan dibuang.
    """
    if len(boxes) <= 1:
        return boxes, statuses

    # Urutkan berdasarkan luas area kotak (terbesar ke terkecil)
    indices = sorted(
        range(len(boxes)),
        key=lambda i: (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1]),
        reverse=True
    )
    kept_indices = []

    for i in indices:
        boxA = boxes[i]
        areaA = max(1, (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
        is_duplicate = False

        for k in kept_indices:
            boxB = boxes[k]
            areaB = max(1, (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)

            # Jika 40% dari boxA berada di dalam boxB atau IoU > 0.30
            overlap_ratio = interArea / float(areaA)
            iou = interArea / float(areaA + areaB - interArea)

            if overlap_ratio > 0.40 or iou > 0.30:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_indices.append(i)

    filtered_boxes = [boxes[i] for i in kept_indices]
    filtered_statuses = [statuses[i] for i in kept_indices]
    return filtered_boxes, filtered_statuses


class PersonTracker:
    def __init__(self, max_missing_frames=5, smooth_factor=0.60):
        self.next_id = 1
        self.tracks = {}
        self.max_missing = max_missing_frames
        self.alpha = smooth_factor

    def update(self, detected_boxes, detected_statuses):
        """
        detected_boxes: list of [x1, y1, x2, y2]
        detected_statuses: list of (status, label, conf)
        """
        # 1. Bersihkan duplikat / potongan tubuh sebelum pelacakan
        clean_boxes, clean_statuses = saring_kotak_duplikat(detected_boxes, detected_statuses)

        matched_track_ids = set()
        updated_results = []

        # 2. Cocokkan deteksi ke track yang sudah ada menggunakan IoU
        for box, stat_data in zip(clean_boxes, clean_statuses):
            if len(stat_data) == 5:
                status, label, conf, detail_pel, zones_pel = stat_data
            else:
                status, label, conf = stat_data[:3]
                detail_pel, zones_pel = [], []

            box_arr = np.array(box, dtype=np.float32)
            best_iou = 0.25
            best_id = None

            for tid, tdata in self.tracks.items():
                if tid in matched_track_ids:
                    continue
                iou = hitung_iou(box_arr, tdata["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_id = tid

            if best_id is not None:
                # Update existing track dengan smoothing (melengket stabil)
                self.tracks[best_id]["box"] = self.alpha * box_arr + (1 - self.alpha) * self.tracks[best_id]["box"]
                self.tracks[best_id]["missing"] = 0
                self.tracks[best_id]["status_history"].append(status)
                if len(self.tracks[best_id]["status_history"]) > 8:
                    self.tracks[best_id]["status_history"].pop(0)
                self.tracks[best_id]["label"] = label
                self.tracks[best_id]["conf"] = conf
                self.tracks[best_id]["detail"] = detail_pel
                self.tracks[best_id]["zones"] = zones_pel
                matched_track_ids.add(best_id)
            else:
                # Track baru (beri ID baru atau daur ulang nomor kecil jika kosong)
                assigned_id = self.next_id
                self.next_id += 1
                if self.next_id > 99:
                    self.next_id = 1  # Reset ID jika terlalu besar
                
                self.tracks[assigned_id] = {
                    "box": box_arr,
                    "missing": 0,
                    "status_history": [status],
                    "label": label,
                    "conf": conf,
                    "detail": detail_pel,
                    "zones": zones_pel
                }
                matched_track_ids.add(assigned_id)

        # 3. Tangani track yang hilang
        to_delete = []
        for tid, tdata in self.tracks.items():
            if tid not in matched_track_ids:
                tdata["missing"] += 1
                if tdata["missing"] > self.max_missing:
                    to_delete.append(tid)

        for tid in to_delete:
            del self.tracks[tid]

        # 4. HANYA tampilkan & hitung orang yang BENAR-BENAR AKTIF di frame
        # Urutkan berdasarkan posisi x kiri-ke-kanan agar penomoran rapi
        active_tracks = [
            (tid, tdata) for tid, tdata in self.tracks.items() if tdata["missing"] == 0
        ]
        active_tracks.sort(key=lambda item: item[1]["box"][0])

        for tid, tdata in active_tracks:
            history = tdata["status_history"]
            is_pelanggaran = sum(1 for s in history if s == "PELANGGARAN") > (len(history) // 3)
            box_int = [int(v) for v in tdata["box"]]
            updated_results.append({
                "id": tid,
                "box": box_int,
                "is_pelanggaran": is_pelanggaran,
                "label": tdata["label"],
                "conf": tdata["conf"],
                "detail": tdata.get("detail", []),
                "zones": tdata.get("zones", [])
            })

        return updated_results
