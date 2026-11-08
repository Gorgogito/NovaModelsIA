"""Generación de imágenes: proceso persistente con Stable Diffusion cargado."""
from __future__ import annotations

import json
import os
import subprocess
import threading

from nova import config, envs

_proc: subprocess.Popen | None = None
_lock = threading.Lock()


def _send(msg: dict) -> dict:
    assert _proc and _proc.stdin and _proc.stdout
    _proc.stdin.write(json.dumps(msg) + "\n")
    _proc.stdin.flush()
    while True:
        line = _proc.stdout.readline()
        if not line:
            raise RuntimeError("El proceso de predicción terminó inesperadamente.")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue


def load(base_model: str, lora_path: str | None, lora_scale: float) -> dict:
    global _proc
    with _lock:
        _kill()
        if not envs.is_ready("image"):
            raise RuntimeError("El entorno del motor de visión no está instalado.")
        script = config.ENGINES_DIR / "image" / "gen_worker.py"
        _proc = subprocess.Popen(
            [str(envs.python_exe("image")), "-u", str(script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HOME": str(config.MODELS_DIR / "hf")},
        )
        r = _send({"cmd": "load", "base_model": base_model, "lora_path": lora_path, "lora_scale": lora_scale})
        if not r.get("ok"):
            _kill()
            raise RuntimeError(r.get("error", "No se pudo cargar el modelo."))
        return r


def generate(params: dict) -> dict:
    with _lock:
        if _proc is None or _proc.poll() is not None:
            raise RuntimeError("No hay ningún modelo cargado.")
        r = _send({"cmd": "generate", **params})
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "Error de generación."))
        return r


def _kill() -> None:
    global _proc
    if _proc and _proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(_proc.pid), "/T", "/F"], capture_output=True)
    _proc = None


def unload() -> None:
    with _lock:
        _kill()


def is_loaded() -> bool:
    return _proc is not None and _proc.poll() is None
