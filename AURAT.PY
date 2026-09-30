"""
SMART MOSQUE GATE — AURAT DETECTION SYSTEM v3.0 (Monorepo Entry)
File ini meneruskan eksekusi langsung ke arsitektur monorepo apps/gate/main.py.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from apps.gate.main import run_app  # type: ignore

if __name__ == "__main__":
    run_app()