"""
YOLO Model Training Pipeline — Monorepo ML Module
"""
import os
import sys
import glob
import shutil

# Memastikan site-packages Windows Store terbaca
site_pack = os.path.expanduser(r"~\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages")
if os.path.exists(site_pack) and site_pack not in sys.path:
    sys.path.append(site_pack)

from ultralytics import YOLO  # type: ignore
from core.config import BASE_DIR, MODELS_DIR, YOLO_MODEL_PATH  # type: ignore

def find_yaml():
    for pattern in ["datasets/**/data.yaml", "Hijab-1/data.yaml", "**/data.yaml"]:
        found = glob.glob(os.path.join(BASE_DIR, pattern), recursive=True)
        if found:
            return found[0]
    return "data.yaml"

def train(data_yaml: str | None = None, epochs: int = 15, imgsz: int = 224, pretrained: str = "models/yolov8n.pt"):
    if data_yaml is None or not os.path.exists(data_yaml):
        data_yaml = find_yaml()

    if not os.path.exists(data_yaml):
        print(f"[!] File dataset '{data_yaml}' belum ditemukan.")
        return

    # Fallback pretrained
    if not os.path.exists(pretrained):
        pretrained = "yolov8n.pt"

    print(f"[*] Training dataset: {data_yaml}")
    print(f"[*] Base model: {pretrained} | Epochs: {epochs} | Imgsz: {imgsz}")
    
    model = YOLO(pretrained)
    project_run = os.path.join(BASE_DIR, "runs", "aurat_model")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        project=project_run,
        name="training_result",
        device="cpu",
        workers=2
    )

    # Copy output weights to models/aurat_best.pt
    best_candidate = os.path.join(project_run, "training_result", "weights", "best.pt")
    if not os.path.exists(best_candidate):
        best_candidate = os.path.join(BASE_DIR, "runs", "detect", "runs", "aurat_model", "training_result", "weights", "best.pt")

    if os.path.exists(best_candidate):
        os.makedirs(MODELS_DIR, exist_ok=True)
        shutil.copyfile(best_candidate, YOLO_MODEL_PATH)
        print("\n" + "=" * 60)
        print("TRAINING MONOREPO SELESAI!")
        print(f"Model .pt tersimpan di: {YOLO_MODEL_PATH}")
        print("=" * 60)

if __name__ == "__main__":
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else None
    train(yaml_path)
