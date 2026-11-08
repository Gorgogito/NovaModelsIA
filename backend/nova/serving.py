"""Registro de modelos servibles y estado de la GPU para la API compatible con OpenAI.

Política: con una sola GPU de 8 GB solo hay un modelo cargado a la vez (de cualquier tipo).
Una petición para otro modelo descarga el actual y carga el pedido. Un candado cubre carga y
generación, de modo que una petición nunca cambia el modelo a mitad de otra.
"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path

from nova import audio_infer, chat, config, db, envs, image_infer


class OpenAIError(Exception):
    """Error con el formato de la API de OpenAI: {"error": {"message", "type", "code", "param"}}."""

    def __init__(self, status: int, message: str, type_: str = "invalid_request_error",
                 code: str | None = None, param: str | None = None):
        super().__init__(message)
        self.status, self.message, self.type, self.code, self.param = status, message, type_, code, param

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "param": self.param, "code": self.code}}


@dataclass
class Entry:
    id: str
    modality: str          # llm | audio | image
    kind: str              # trained | base
    base_model: str
    path: str | None = None  # adaptador LLM / carpeta del Whisper ajustado / carpeta del LoRA
    created: int = 0

    def public(self) -> dict:
        return {"id": self.id, "object": "model", "created": self.created, "owned_by": "nova-local",
                "nova": {"modality": self.modality, "kind": self.kind, "base_model": self.base_model}}


ENGINE_OF = {"llm": "llm", "audio": "audio", "image": "image"}
HF_HUB = config.MODELS_DIR / "hf" / "hub"


def _classify(snapshot: Path) -> str | None:
    """Tipo de un modelo descargado, mirando sus archivos de configuración (sin red)."""
    try:
        if (snapshot / "model_index.json").exists():
            cls = json.loads((snapshot / "model_index.json").read_text(encoding="utf-8")).get("_class_name", "")
            return "image" if cls == "StableDiffusionPipeline" else None  # SDXL no se sirve: no cabe en 8 GB
        if (snapshot / "config.json").exists():
            archs = json.loads((snapshot / "config.json").read_text(encoding="utf-8")).get("architectures") or []
            if any("Whisper" in a for a in archs):
                return "audio"
            if any(a.endswith("ForCausalLM") for a in archs):
                return "llm"
    except (OSError, ValueError):
        pass
    return None


def list_entries() -> list[Entry]:
    out: list[Entry] = []
    for j in db.rows("SELECT * FROM jobs WHERE kind='train' AND status='completed' AND engine IN ('llm','audio','image') ORDER BY id"):
        cfg = json.loads(j["config"])
        d = Path(j["output_dir"])
        created = int(j["finished_at"] or j["created_at"])
        if j["engine"] == "llm" and (d / "adapter").is_dir():
            out.append(Entry(f"nova-llm-{j['id']}", "llm", "trained", cfg["base_model"], str(d / "adapter"), created))
        elif j["engine"] == "audio" and (d / "model").is_dir():
            out.append(Entry(f"nova-whisper-{j['id']}", "audio", "trained", cfg["base_model"], str(d / "model"), created))
        elif j["engine"] == "image" and (d / "lora").is_dir():
            out.append(Entry(f"nova-sd-lora-{j['id']}", "image", "trained", cfg["base_model"], str(d / "lora"), created))
    if HF_HUB.is_dir():
        for repo in sorted(HF_HUB.glob("models--*")):
            snaps = sorted((repo / "snapshots").glob("*")) if (repo / "snapshots").is_dir() else []
            modality = _classify(snaps[-1]) if snaps else None
            if modality:
                rid = repo.name.removeprefix("models--").replace("--", "/")
                out.append(Entry(rid, modality, "base", rid, None, int(repo.stat().st_mtime)))
    return out


def find(model_id: str, modality: str) -> Entry:
    for e in list_entries():
        if e.id == model_id:
            if e.modality != modality:
                raise OpenAIError(400, f"El modelo `{model_id}` es de tipo '{e.modality}' y este endpoint necesita '{modality}'.",
                                  code="model_not_supported", param="model")
            return e
    raise OpenAIError(404, f"El modelo `{model_id}` no existe o no está descargado. Consulta GET /v1/models.",
                      code="model_not_found", param="model")


# ---------- Estado de la GPU ----------
_lock = threading.RLock()
_current: tuple[str, str] | None = None


def _alive(modality: str) -> bool:
    return {"llm": chat.is_loaded, "audio": audio_infer.is_loaded, "image": image_infer.is_loaded}[modality]()


def loaded() -> dict | None:
    cur = _current
    return {"modality": cur[0], "id": cur[1]} if cur and _alive(cur[0]) else None


def invalidate() -> None:
    """Las cargas manuales desde la interfaz sustituyen lo que había: el estado de la API deja de ser fiable."""
    global _current
    _current = None


def unload_all() -> None:
    with _lock:
        chat.unload()
        audio_infer.unload()
        image_infer.unload()
        invalidate()


def _ensure(e: Entry) -> None:
    global _current
    if db.one("SELECT 1 FROM jobs WHERE status='running'"):
        raise OpenAIError(503, "Hay un entrenamiento o evaluación en curso; la GPU está ocupada. Inténtalo cuando termine.",
                          "server_error", "gpu_busy")
    if not envs.is_ready(ENGINE_OF[e.modality]):
        raise OpenAIError(503, f"El motor '{ENGINE_OF[e.modality]}' no está instalado. Instálalo desde la aplicación.",
                          "server_error", "engine_not_installed")
    if _current == (e.modality, e.id) and _alive(e.modality):
        return
    unload_all()
    try:
        if e.modality == "llm":
            chat.load(e.base_model, e.path, True)
        elif e.modality == "audio":
            audio_infer.load(e.path or e.base_model)
        else:
            image_infer.load(e.base_model, e.path, 1.0)
    except Exception as ex:  # noqa: BLE001
        unload_all()
        raise OpenAIError(500, f"No se pudo cargar el modelo `{e.id}`: {ex}", "server_error", "model_load_failed") from ex
    _current = (e.modality, e.id)


@contextmanager
def use(e: Entry):
    """Carga el modelo si hace falta y mantiene el candado mientras se usa."""
    with _lock:
        _ensure(e)
        yield


def preload(e: Entry) -> None:
    """Carga el modelo y suelta el candado: sirve para detectar errores antes de empezar a transmitir."""
    with use(e):
        pass


def entries_public() -> list[dict]:
    return [asdict(e) for e in list_entries()]
