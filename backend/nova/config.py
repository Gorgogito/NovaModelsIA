from __future__ import annotations

import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))  # True cuando corre como ejecutable empaquetado (PyInstaller)
ROOT = Path(__file__).resolve().parents[2]


def _default_data_dir() -> Path:
    if FROZEN:  # app instalada: los datos no pueden vivir junto al código
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "NovaModelsIA"
    return ROOT / "data"  # desarrollo: carpeta data/ del repositorio


# NOVA_DATA permite elegir otro disco (los entornos y modelos ocupan decenas de GB).
DATA_DIR = Path(os.environ.get("NOVA_DATA") or _default_data_dir())
ENVS_DIR = DATA_DIR / "envs"
PROJECTS_DIR = DATA_DIR / "projects"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "nova.db"
ENGINES_DIR = Path(__file__).resolve().parent / "engines"  # empaquetado: se incluye como datos en nova/engines

for _d in (DATA_DIR, ENVS_DIR, PROJECTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
