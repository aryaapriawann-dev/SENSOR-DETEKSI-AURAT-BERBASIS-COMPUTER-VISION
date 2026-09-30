"""
Monorepo Wrapper: Melatih Model YOLO
Meneruskan perintah ke ml/train.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.train import train

if __name__ == "__main__":
    yaml_arg = sys.argv[1] if len(sys.argv) > 1 else None
    train(yaml_arg)
