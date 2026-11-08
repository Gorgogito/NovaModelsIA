"""Inferencia interactiva: un proceso persistente del motor LLM con el modelo cargado."""
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
    _proc.stdin.write(json.dumps(msg, ensure_ascii=False) + "\n")
    _proc.stdin.flush()
    while True:
        line = _proc.stdout.readline()
        if not line:
            raise RuntimeError("El proceso de inferencia terminó inesperadamente.")
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue  # ruido de librerías (barras de progreso, avisos)


def load(base_model: str, adapter_path: str | None, quant_4bit: bool) -> None:
    global _proc
    with _lock:
        unload_locked()
        if not envs.is_ready("llm"):
            raise RuntimeError("El entorno del motor LLM no está instalado.")
        script = config.ENGINES_DIR / "llm" / "chat_worker.py"
        _proc = subprocess.Popen(
            [str(envs.python_exe("llm")), "-u", str(script)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "HF_HOME": str(config.MODELS_DIR / "hf")},
        )
        r = _send({"cmd": "load", "base_model": base_model, "adapter_path": adapter_path, "quant_4bit": quant_4bit})
        if not r.get("ok"):
            unload_locked()
            raise RuntimeError(r.get("error", "No se pudo cargar el modelo."))


def generate(messages: list[dict], max_new_tokens: int, temperature: float) -> str:
    with _lock:
        if _proc is None or _proc.poll() is not None:
            raise RuntimeError("No hay ningún modelo cargado.")
        r = _send({"cmd": "chat", "messages": messages, "max_new_tokens": max_new_tokens, "temperature": temperature})
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "Error de generación."))
        return r["reply"]


def generate_stream(messages: list[dict], max_new_tokens: int, temperature: float, top_p: float = 0.9):
    """Generador: produce los fragmentos de texto y termina devolviendo el uso de tokens.

    Uso: `usage = yield from generate_stream(...)`. Mantiene el bloqueo mientras dura la generación.
    Si el consumidor cierra el generador a mitad, se lee el resto de la salida del proceso para que el
    protocolo no quede desincronizado para la siguiente petición.
    """
    with _lock:
        if _proc is None or _proc.poll() is not None:
            raise RuntimeError("No hay ningún modelo cargado.")
        assert _proc.stdin and _proc.stdout
        _proc.stdin.write(json.dumps({"cmd": "chat_stream", "messages": messages, "max_new_tokens": max_new_tokens,
                                      "temperature": temperature, "top_p": top_p}, ensure_ascii=False) + "\n")
        _proc.stdin.flush()
        finished = False
        try:
            while True:
                line = _proc.stdout.readline()
                if not line:
                    finished = True
                    raise RuntimeError("El proceso de inferencia terminó inesperadamente.")
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue  # ruido de librerías
                if "delta" in ev:
                    yield ev["delta"]
                elif ev.get("done"):
                    finished = True
                    if not ev.get("ok"):
                        raise RuntimeError(ev.get("error", "Error de generación."))
                    return {"prompt_tokens": ev.get("prompt_tokens", 0), "completion_tokens": ev.get("completion_tokens", 0)}
        finally:
            if not finished:  # el consumidor se fue: vaciar lo que queda de esta respuesta
                for line in _proc.stdout:
                    try:
                        if json.loads(line).get("done"):
                            break
                    except json.JSONDecodeError:
                        continue


def is_loaded() -> bool:
    return _proc is not None and _proc.poll() is None


def unload_locked() -> None:
    global _proc
    if _proc and _proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(_proc.pid), "/T", "/F"], capture_output=True)
    _proc = None


def unload() -> None:
    with _lock:
        unload_locked()
