"""Gestor de entornos virtuales: uno por motor, creado con uv."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from nova import config

# Índice de PyTorch con CUDA (RTX 30xx/40xx). Cambiar aquí si se actualiza CUDA.
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_PACKAGES = {"vision": ["torch", "torchvision"], "image": ["torch", "torchvision"]}


def env_dir(engine: str) -> Path:
    return config.ENVS_DIR / engine


def python_exe(engine: str) -> Path:
    return env_dir(engine) / "Scripts" / "python.exe"


def is_ready(engine: str) -> bool:
    return (env_dir(engine) / ".ready").exists()


def uv_command() -> list[str]:
    """Comando para ejecutar uv, sin depender del Python del sistema.

    Orden: variable NOVA_UV > uv.exe junto al ejecutable (app empaquetada) > uv.exe del entorno del
    backend (Scripts/) > uv en el PATH > `python -m uv` (solo con Python normal, no empaquetado).
    """
    candidates = []
    if os.environ.get("NOVA_UV"):
        candidates.append(Path(os.environ["NOVA_UV"]))
    exe_dir = Path(sys.executable).parent
    candidates += [exe_dir / "uv.exe", exe_dir / "Scripts" / "uv.exe"]
    for c in candidates:
        if c.is_file():
            return [str(c)]
    found = shutil.which("uv")
    if found:
        return [found]
    if not getattr(sys, "frozen", False):
        probe = subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True)
        if probe.returncode == 0:
            return [sys.executable, "-m", "uv"]
    raise RuntimeError("No se encontró uv. Reinstala la aplicación, o instálalo con `pip install uv` y define NOVA_UV si hace falta.")


def _uv(*args: str) -> list[str]:
    return [*uv_command(), *args]


def install(engine: str, log=print) -> None:
    """Crea el venv e instala requirements.txt del motor."""
    req = config.ENGINES_DIR / engine / "requirements.txt"
    d = env_dir(engine)
    if not python_exe(engine).exists():
        subprocess.run(_uv("venv", str(d), "--python", "3.12"), check=True)
    cmds = [
        _uv("pip", "install", "--python", str(python_exe(engine)), *TORCH_PACKAGES.get(engine, ["torch"]),
            "--index-url", TORCH_INDEX),
        _uv("pip", "install", "--python", str(python_exe(engine)), "-r", str(req)),
    ]
    for cmd in cmds:
        log(" ".join(cmd))
        subprocess.run(cmd, check=True)
    (d / ".ready").write_text("ok")
