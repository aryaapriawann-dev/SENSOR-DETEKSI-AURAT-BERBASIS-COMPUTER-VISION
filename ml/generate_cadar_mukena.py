"""
Dataset Builder & Augmentor for Cadar (Niqab) and Mukena
Menyiapkan dataset lengkap 5 kelas:
  0: HijabSyar-i
  1: Non Syar-i
  2: Non hijab
  3: Cadar_Niqab
  4: Mukena
"""
import os
import glob
import shutil
import random
import cv2  # type: ignore
import numpy as np  # type: ignore

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "Hijab-1")
OUT_DIR = os.path.join(BASE_DIR, "datasets", "syari_dataset")


def create_cadar_image(img_bgr, bbox_norm):
    """
    Mengaplikasikan cadar/niqab hitam atau gelap yang realistis pada wajah.
    bbox_norm: [cls, x_c, y_c, w, h] atau poligon
    """
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    # Perkirakan area kepala/wajah dari bounding box
    xc, yc, bw, bh = bbox_norm
    x1 = max(0, int((xc - bw / 2.0) * w))
    y1 = max(0, int((yc - bh / 2.0) * h))
    x2 = min(w, int((xc + bw / 2.0) * w))
    y2 = min(h, int((yc + bh / 2.0) * h))

    fw = x2 - x1
    fh = y2 - y1
    if fw < 20 or fh < 20:
        return None, None

    # Cadar menutupi dari bawah mata (sekitar 40% dari tinggi kepala) sampai leher dan dada
    cadar_y1 = y1 + int(fh * 0.40)
    cadar_y2 = min(h, y2 + int(fh * 0.20))
    cadar_x1 = max(0, x1 - int(fw * 0.05))
    cadar_x2 = min(w, x2 + int(fw * 0.05))

    # Warna cadar: variasi hitam pekat, hitam abu, navy gelap
    base_color = random.choice([
        (18, 18, 18),
        (25, 25, 28),
        (35, 30, 30),
        (30, 25, 35),
        (12, 12, 15)
    ])

    # Gambar kain cadar dengan tekstur lipatan alami (folds/shadows)
    cadar_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Poligon cadar: melengkung di hidung dan jatuh menutupi leher/dada
    pts = np.array([
        [cadar_x1, cadar_y1 + int(fh * 0.05)],
        [int((cadar_x1 + cadar_x2) / 2), cadar_y1],
        [cadar_x2, cadar_y1 + int(fh * 0.05)],
        [cadar_x2 + int(fw * 0.08), cadar_y2],
        [cadar_x1 - int(fw * 0.08), cadar_y2]
    ], np.int32)
    
    cv2.fillPoly(cadar_mask, [pts], 255)
    cadar_mask = cv2.GaussianBlur(cadar_mask, (7, 7), 0)

    # Buat tekstur kain dengan sedikit noise & gradien lipatan
    noise = np.random.randint(-8, 8, (h, w, 3), dtype=np.int16)
    cloth = np.full((h, w, 3), base_color, dtype=np.int16) + noise
    cloth = np.clip(cloth, 0, 255).astype(np.uint8)

    # Tambahkan lipatan kain lembut (vertical subtle shadow)
    fold_x = int((cadar_x1 + cadar_x2) / 2)
    cv2.line(cloth, (fold_x, cadar_y1 + 10), (fold_x, cadar_y2), (max(0, base_color[0]-10), max(0, base_color[1]-10), max(0, base_color[2]-10)), 3)

    # Blending kain cadar ke gambar
    alpha = (cadar_mask / 255.0)[:, :, np.newaxis]
    out = (out * (1.0 - alpha) + cloth * alpha).astype(np.uint8)

    # Hitung bounding box baru untuk Cadar_Niqab (kelas 3)
    c_xc = ((cadar_x1 + cadar_x2) / 2.0) / w
    c_yc = ((y1 + cadar_y2) / 2.0) / h
    c_w = (cadar_x2 - cadar_x1 + int(fw * 0.16)) / w
    c_h = (cadar_y2 - y1) / h

    return out, (3, c_xc, c_yc, c_w, c_h)


def create_mukena_image(img_bgr, bbox_norm):
    """
    Mengaplikasikan mukena sholat putih/pastel yang membungkus kepala dan dada.
    """
    h, w = img_bgr.shape[:2]
    out = img_bgr.copy()

    xc, yc, bw, bh = bbox_norm
    x1 = max(0, int((xc - bw / 2.0) * w))
    y1 = max(0, int((yc - bh / 2.0) * h))
    x2 = min(w, int((xc + bw / 2.0) * w))
    y2 = min(h, int((yc + bh / 2.0) * h))

    fw = x2 - x1
    fh = y2 - y1
    if fw < 20 or fh < 20:
        return None, None

    # Warna mukena: putih bersih, broken white, krem muda, pastel
    mukena_color = random.choice([
        (240, 242, 245),
        (230, 235, 238),
        (225, 230, 230),
        (245, 240, 240),
        (235, 238, 245)
    ])

    # Mukena menutupi seluruh kepala, leher, dada, kecuali oval wajah
    mukena_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Outer hood mukena
    hood_pts = np.array([
        [x1 - int(fw * 0.15), y1 + int(fh * 0.20)],
        [int((x1 + x2) / 2), y1 - int(fh * 0.10)],
        [x2 + int(fw * 0.15), y1 + int(fh * 0.20)],
        [x2 + int(fw * 0.30), y2 + int(fh * 0.35)],
        [x1 - int(fw * 0.30), y2 + int(fh * 0.35)]
    ], np.int32)
    cv2.fillPoly(mukena_mask, [hood_pts], 255)

    # Face cutout (oval wajah yang terbuka: dahi s.d. dagu)
    face_cx = int((x1 + x2) / 2)
    face_cy = int(y1 + fh * 0.45)
    face_ax1 = int(fw * 0.32)
    face_ax2 = int(fh * 0.36)
    cv2.ellipse(mukena_mask, (face_cx, face_cy), (face_ax1, face_ax2), 0, 0, 360, 0, -1)
    
    mukena_mask = cv2.GaussianBlur(mukena_mask, (9, 9), 0)

    # Tekstur kain mukena dengan renda/lipatan
    noise = np.random.randint(-10, 10, (h, w, 3), dtype=np.int16)
    cloth = np.full((h, w, 3), mukena_color, dtype=np.int16) + noise
    cloth = np.clip(cloth, 0, 255).astype(np.uint8)

    # Blending mukena ke gambar
    alpha = (mukena_mask / 255.0)[:, :, np.newaxis]
    out = (out * (1.0 - alpha) + cloth * alpha).astype(np.uint8)

    # Hitung bounding box baru untuk Mukena (kelas 4)
    m_xc = ((x1 + x2) / 2.0) / w
    m_yc = ((y1 - int(fh * 0.10) + y2 + int(fh * 0.35)) / 2.0) / h
    m_w = min(1.0, (fw * 1.6) / w)
    m_h = min(1.0, (fh * 1.45) / h)

    return out, (4, m_xc, m_yc, m_w, m_h)


def parse_label_line(line):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = int(parts[0])
    # Jika polygon, ambil bounding box dari polygon
    if len(parts) > 5:
        coords = [float(p) for p in parts[1:]]
        xs = coords[0::2]
        ys = coords[1::2]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        xc = (min_x + max_x) / 2.0
        yc = (min_y + max_y) / 2.0
        bw = max_x - min_x
        bh = max_y - min_y
        return cls_id, xc, yc, bw, bh
    else:
        return cls_id, float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])


def build_dataset():
    print("[*] Membangun dataset lengkap: Hijab Syar-i, Mukena, Cadar, Non Syar-i, Non Hijab...")
    
    for split in ["train", "valid"]:
        img_out = os.path.join(OUT_DIR, split, "images")
        lbl_out = os.path.join(OUT_DIR, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        src_imgs = glob.glob(os.path.join(SRC_DIR, split, "images", "*.*"))
        print(f"[*] Memproses {len(src_imgs)} gambar dasar pada split '{split}'...")

        cadar_count = 0
        mukena_count = 0

        for img_path in src_imgs:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(SRC_DIR, split, "labels", f"{base_name}.txt")
            
            img = cv2.imread(img_path)
            if img is None:
                continue

            # 1. Salin gambar asli dan label asli
            shutil.copyfile(img_path, os.path.join(img_out, os.path.basename(img_path)))
            if os.path.exists(lbl_path):
                shutil.copyfile(lbl_path, os.path.join(lbl_out, f"{base_name}.txt"))

            # Baca label untuk augmentasi cadar & mukena
            if not os.path.exists(lbl_path):
                continue

            with open(lbl_path, "r") as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                parsed = parse_label_line(line)
                if parsed is None:
                    continue
                cls_id, xc, yc, bw, bh = parsed

                # Buat Cadar (Kelas 3)
                cadar_img, cadar_box = create_cadar_image(img, (xc, yc, bw, bh))
                if cadar_img is not None and cadar_box is not None:
                    c_name = f"cadar_{split}_{cadar_count:04d}"
                    cv2.imwrite(os.path.join(img_out, f"{c_name}.jpg"), cadar_img)
                    with open(os.path.join(lbl_out, f"{c_name}.txt"), "w") as cf:
                        cf.write(f"{cadar_box[0]} {cadar_box[1]:.6f} {cadar_box[2]:.6f} {cadar_box[3]:.6f} {cadar_box[4]:.6f}\n")
                    cadar_count += 1

                # Buat Mukena (Kelas 4)
                mukena_img, mukena_box = create_mukena_image(img, (xc, yc, bw, bh))
                if mukena_img is not None and mukena_box is not None:
                    m_name = f"mukena_{split}_{mukena_count:04d}"
                    cv2.imwrite(os.path.join(img_out, f"{m_name}.jpg"), mukena_img)
                    with open(os.path.join(lbl_out, f"{m_name}.txt"), "w") as mf:
                        mf.write(f"{mukena_box[0]} {mukena_box[1]:.6f} {mukena_box[2]:.6f} {mukena_box[3]:.6f} {mukena_box[4]:.6f}\n")
                    mukena_count += 1

        print(f"[+] Split '{split}' selesai! Ditambahkan {cadar_count} Cadar dan {mukena_count} Mukena.")

    # 2. Tulis data.yaml untuk Ultralytics YOLOv8
    yaml_content = f"""path: {OUT_DIR.replace('\\', '/')}
train: train/images
val: valid/images
test: valid/images

nc: 5
names:
  0: HijabSyar-i
  1: Non Syar-i
  2: Non hijab
  3: Cadar_Niqab
  4: Mukena
"""
    yaml_path = os.path.join(OUT_DIR, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n[+] Dataset berhasil disiapkan di: {OUT_DIR}")
    print(f"[+] File konfigurasi: {yaml_path}")
    return yaml_path


if __name__ == "__main__":
    build_dataset()
