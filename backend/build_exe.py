"""Empaqueta el backend como programa independiente (no necesita Python ni `uv run` en la máquina).

Uso (desde backend/):   python -m uv run python build_exe.py
Resultado: dist/nova-backend/nova-backend.exe  (+ uv.exe al lado, para crear los entornos de los motores)

Lo que NO se empaqueta, a propósito: PyTorch y las librerías de IA. Cada motor se instala la primera vez
que se usa, en un entorno aislado, con el uv que va incluido (necesita internet esa primera vez).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
dist = root / "dist" / "nova-backend"

cmd = [
    sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onedir", "--name", "nova-backend",
    "--distpath", str(root / "dist"), "--workpath", str(root / "build"), "--specpath", str(root / "build"),
    # módulos que se cargan por nombre en tiempo de ejecución
    "--collect-submodules", "uvicorn", "--collect-submodules", "anyio", "--collect-submodules", "nova",
    "--hidden-import", "python_multipart", "--hidden-import", "multipart", "--hidden-import", "huggingface_hub",
    "--collect-all", "soundfile", "--collect-all", "_soundfile_data",
    # los scripts de los motores los ejecuta el Python de cada entorno: van como archivos, no como módulos
    "--add-data", f"{root / 'nova' / 'engines'}{os.pathsep}nova/engines",
    "--exclude-module", "tkinter", "--exclude-module", "pytest",
    str(root / "run_nova.py"),
]
print(" ".join(cmd))
subprocess.run(cmd, check=True, cwd=root)

uv = Path(sys.executable).parent / "uv.exe"
if not uv.is_file():
    sys.exit(f"No se encontró {uv}: instala uv en el entorno (uv sync).")
shutil.copy2(uv, dist / "uv.exe")
size = sum(f.stat().st_size for f in dist.rglob("*") if f.is_file()) / 2**20
print(f"\nListo: {dist / 'nova-backend.exe'}  ({size:.0f} MB en total)")
