from __future__ import annotations

import json
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from nova import audio_infer, chat, config, datasets, db, envs, image_infer, jobs, serving, vision_infer
from nova.api.openai_compat import router as openai_router
from nova.hardware import HardwareStatus, read_status
from nova.serving import OpenAIError


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init()
    jobs.start()
    yield
    chat.unload()
    vision_infer.unload()
    audio_infer.unload()
    image_infer.unload()


app = FastAPI(title="NovaModelsIA", version="0.2.0", lifespan=lifespan)

# Solo la interfaz local (Tauri/Vite) puede llamar a la API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:1420", "tauri://localhost", "http://tauri.localhost", "https://tauri.localhost"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(openai_router)


@app.exception_handler(OpenAIError)
async def _openai_error(_: Request, e: OpenAIError):
    return JSONResponse(status_code=e.status, content=e.body())


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, e: RequestValidationError):
    if request.url.path.startswith("/v1/"):  # los clientes de OpenAI esperan {"error": {...}}
        first = e.errors()[0] if e.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        return JSONResponse(status_code=400, content=OpenAIError(
            400, f"{loc}: {first.get('msg', 'petición inválida')}" if loc else first.get("msg", "petición inválida"),
            code="invalid_request", param=loc or None).body())
    return await request_validation_exception_handler(request, e)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/hardware", response_model=HardwareStatus)
def hardware() -> HardwareStatus:
    return read_status()


# ---------- Proyectos ----------
class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)


@app.get("/projects")
def list_projects():
    return db.rows("SELECT * FROM projects ORDER BY id DESC")


@app.post("/projects")
def create_project(p: ProjectIn):
    pid = db.execute("INSERT INTO projects(name,created_at) VALUES(?,?)", (p.name.strip(), db.now()))
    (config.PROJECTS_DIR / str(pid) / "datasets").mkdir(parents=True, exist_ok=True)
    return db.one("SELECT * FROM projects WHERE id=?", (pid,))


@app.delete("/projects/{pid}")
def delete_project(pid: int):
    if db.one("SELECT 1 FROM jobs WHERE project_id=? AND status IN ('queued','running')", (pid,)):
        raise HTTPException(409, "Hay entrenamientos activos en este proyecto.")
    for t in ("metrics", "logs"):  # primero los hijos de los trabajos, para no dejar filas huérfanas
        db.execute(f"DELETE FROM {t} WHERE job_id IN (SELECT id FROM jobs WHERE project_id=?)", (pid,))
    for t in ("datasets", "jobs"):
        db.execute(f"DELETE FROM {t} WHERE project_id=?", (pid,))
    db.execute("DELETE FROM projects WHERE id=?", (pid,))
    shutil.rmtree(config.PROJECTS_DIR / str(pid), ignore_errors=True)
    return {"ok": True}


# ---------- Datasets ----------
class DatasetIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    source_path: str
    license: str = ""


def _project_or_404(pid: int) -> dict:
    p = db.one("SELECT * FROM projects WHERE id=?", (pid,))
    if not p:
        raise HTTPException(404, "Proyecto no encontrado.")
    return p


@app.get("/projects/{pid}/datasets")
def list_datasets(pid: int, kind: str | None = None):
    sql, args = "SELECT * FROM datasets WHERE project_id=?", (pid,)
    if kind:
        sql, args = sql + " AND kind=?", (pid, kind)
    out = db.rows(sql + " ORDER BY id DESC", args)
    for d in out:
        d["report"] = json.loads(d["report"]) if d["report"] else None
    return out


@app.post("/projects/{pid}/datasets")
def import_dataset(pid: int, d: DatasetIn):
    _project_or_404(pid)
    src = Path(d.source_path)
    if not src.is_file() or src.suffix.lower() not in {".jsonl", ".json"}:
        raise HTTPException(400, "Indica la ruta de un archivo .jsonl existente.")
    report = datasets.validate_file(src)
    if report["valid"] == 0:
        raise HTTPException(400, {"msg": "No hay registros válidos.", "report": report})
    dst = config.PROJECTS_DIR / str(pid) / "datasets" / f"{src.stem}_{uuid.uuid4().hex[:10]}.jsonl"
    n = datasets.clean_file(src, dst)  # copia normalizada, sin inválidos ni duplicados
    did = db.execute(
        "INSERT INTO datasets(project_id,name,path,rows,license,report,created_at) VALUES(?,?,?,?,?,?,?)",
        (pid, d.name.strip(), str(dst), n, d.license, db.dumps(report), db.now()),
    )
    return {"id": did, "rows": n, "report": report}


@app.get("/datasets/{did}/preview")
def preview_dataset(did: int, limit: int = 5):
    d = db.one("SELECT path FROM datasets WHERE id=?", (did,))
    if not d:
        raise HTTPException(404, "Dataset no encontrado.")
    out = []
    with open(d["path"], encoding="utf-8") as f:
        for _, line in zip(range(max(1, min(limit, 50))), f):
            out.append(json.loads(line))
    return out


# ---------- Entorno del motor ----------
_install_state: dict[str, str] = {}


@app.get("/engines/{engine}/env")
def env_status(engine: str):
    return {"ready": envs.is_ready(engine), "state": _install_state.get(engine, "idle")}


@app.post("/engines/{engine}/env/install")
def env_install(engine: str):
    if engine not in {"llm", "vision", "audio", "image"}:
        raise HTTPException(404, "Motor desconocido.")
    if _install_state.get(engine) == "installing":
        return {"state": "installing"}
    _install_state[engine] = "installing"

    def run():
        try:
            envs.install(engine)
            _install_state[engine] = "done"
        except Exception as e:  # noqa: BLE001
            _install_state[engine] = f"error: {e}"

    threading.Thread(target=run, daemon=True).start()
    return {"state": "installing"}


# ---------- Entrenamiento ----------
class TrainIn(BaseModel):
    dataset_id: int
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    epochs: float = Field(1, gt=0, le=100)
    batch_size: int = Field(1, ge=1, le=64)
    grad_accum: int = Field(8, ge=1, le=256)
    learning_rate: float = Field(2e-4, gt=0, lt=1)
    max_seq_len: int = Field(512, ge=32, le=8192)
    lora_r: int = Field(16, ge=1, le=256)
    lora_alpha: int = Field(32, ge=1, le=512)
    lora_dropout: float = Field(0.05, ge=0, lt=1)
    optimizer: str = "paged_adamw_8bit"
    scheduler: str = "cosine"
    warmup_ratio: float = Field(0.03, ge=0, lt=1)
    quant_4bit: bool = True
    gradient_checkpointing: bool = True
    save_steps: int = Field(50, ge=1)
    resume_job_id: int | None = None


@app.post("/projects/{pid}/train")
def start_training(pid: int, t: TrainIn):
    _project_or_404(pid)
    ds = db.one("SELECT * FROM datasets WHERE id=? AND project_id=? AND kind='text'", (t.dataset_id, pid))
    if not ds:
        raise HTTPException(404, "Dataset no encontrado en este proyecto.")
    if not envs.is_ready("llm"):
        raise HTTPException(409, "Instala primero el entorno del motor LLM.")
    cfg = t.model_dump()
    cfg["dataset_path"] = ds["path"]
    out = config.PROJECTS_DIR / str(pid) / "runs" / f"run_{uuid.uuid4().hex[:10]}"
    if t.resume_job_id:
        prev = db.one("SELECT output_dir FROM jobs WHERE id=? AND project_id=?", (t.resume_job_id, pid))
        if not prev:
            raise HTTPException(404, "Entrenamiento a reanudar no encontrado.")
        out = Path(prev["output_dir"])
    jid = jobs.submit(pid, "llm", "train", cfg, out)
    return {"job_id": jid}


@app.get("/projects/{pid}/jobs")
def list_jobs(pid: int):
    out = db.rows("SELECT * FROM jobs WHERE project_id=? ORDER BY id DESC", (pid,))
    for j in out:
        j["config"] = json.loads(j["config"])
    return out


@app.get("/jobs/{jid}")
def get_job(jid: int):
    j = db.one("SELECT * FROM jobs WHERE id=?", (jid,))
    if not j:
        raise HTTPException(404, "Job no encontrado.")
    j["config"] = json.loads(j["config"])
    return j


@app.get("/jobs/{jid}/metrics")
def job_metrics(jid: int, after: int = 0):
    r = db.rows("SELECT id, ts, data FROM metrics WHERE job_id=? AND id>? ORDER BY id LIMIT 2000", (jid, after))
    return [{"id": x["id"], "ts": x["ts"], **json.loads(x["data"])} for x in r]


@app.get("/jobs/{jid}/logs")
def job_logs(jid: int, after: int = 0):
    return db.rows("SELECT id, ts, line FROM logs WHERE job_id=? AND id>? ORDER BY id LIMIT 500", (jid, after))


@app.post("/jobs/{jid}/stop")
def stop_job(jid: int):
    return {"stopped": jobs.stop(jid)}


# ---------- Modelos: licencia ----------
@app.get("/models/license")
def model_license(repo_id: str):
    """Consulta la licencia declarada en el Hub (requiere red)."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise HTTPException(501, "huggingface_hub no está en el entorno del backend.")
    info = HfApi().model_info(repo_id)
    lic = next((t.split(":", 1)[1] for t in info.tags if t.startswith("license:")), "desconocida")
    return {"repo_id": repo_id, "license": lic}


# ---------- Inferencia ----------
class LoadIn(BaseModel):
    base_model: str
    adapter_path: str | None = None
    quant_4bit: bool = True


class ChatIn(BaseModel):
    messages: list[dict]
    max_new_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0, le=2)


@app.post("/llm/load")
def llm_load(b: LoadIn):
    serving.invalidate()
    if db.one("SELECT 1 FROM jobs WHERE status='running'"):
        raise HTTPException(409, "Hay un entrenamiento en curso; la GPU está ocupada.")
    try:
        chat.load(b.base_model, b.adapter_path, b.quant_4bit)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))
    return {"loaded": True}


@app.post("/llm/chat")
def llm_chat(c: ChatIn):
    try:
        return {"reply": chat.generate(c.messages, c.max_new_tokens, c.temperature)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@app.post("/llm/unload")
def llm_unload():
    serving.invalidate()
    chat.unload()
    return {"loaded": False}


# ---------- Exportación ----------
class ExportIn(BaseModel):
    job_id: int
    gguf: bool = False
    gguf_quant: str = "q8_0"  # f16, bf16 o q8_0 (Q4 requiere llama-quantize)


@app.post("/projects/{pid}/export")
def export_model(pid: int, e: ExportIn):
    src = db.one("SELECT * FROM jobs WHERE id=? AND project_id=? AND kind='train'", (e.job_id, pid))
    if not src or src["status"] != "completed":
        raise HTTPException(400, "Solo se pueden exportar entrenamientos completados.")
    cfg = json.loads(src["config"])
    out = Path(src["output_dir"]) / "export"
    jid = jobs.submit(pid, "llm", "export", {
        "base_model": cfg["base_model"], "adapter_path": str(Path(src["output_dir"]) / "adapter"),
        "out_dir": str(out), "gguf": e.gguf, "gguf_quant": e.gguf_quant,
        "llama_cpp_dir": str(config.DATA_DIR / "llama.cpp"),
    }, out)
    return {"job_id": jid}


# ---------- Evaluación ----------
class EvalIn(BaseModel):
    dataset_id: int
    job_id: int | None = None  # None = modelo base sin adaptador
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_rows: int = Field(50, ge=1, le=500)
    max_new_tokens: int = Field(128, ge=1, le=1024)
    quant_4bit: bool = True


@app.post("/projects/{pid}/eval")
def start_eval(pid: int, e: EvalIn):
    _project_or_404(pid)
    ds = db.one("SELECT * FROM datasets WHERE id=? AND project_id=? AND kind='text'", (e.dataset_id, pid))
    if not ds:
        raise HTTPException(404, "Dataset no encontrado en este proyecto.")
    if not envs.is_ready("llm"):
        raise HTTPException(409, "Instala primero el entorno del motor LLM.")
    cfg = {"dataset_path": ds["path"], "max_rows": e.max_rows, "max_new_tokens": e.max_new_tokens,
           "quant_4bit": e.quant_4bit, "base_model": e.base_model, "adapter_path": None,
           "dataset_id": e.dataset_id, "dataset_name": ds["name"], "target_job_id": e.job_id,
           "label": "Modelo base"}
    trained_on = None
    if e.job_id is not None:
        src = db.one("SELECT * FROM jobs WHERE id=? AND project_id=? AND kind='train' AND status='completed'",
                     (e.job_id, pid))
        if not src:
            raise HTTPException(400, "El entrenamiento indicado no existe o no está completado.")
        scfg = json.loads(src["config"])
        cfg.update(base_model=scfg["base_model"], adapter_path=str(Path(src["output_dir"]) / "adapter"),
                   label=f"Entrenamiento #{src['id']}")
        trained_on = scfg.get("dataset_path")
    cfg["contaminated"] = trained_on == ds["path"]  # evaluar con los mismos datos de entrenamiento infla el resultado
    out = config.PROJECTS_DIR / str(pid) / "evals" / f"eval_{uuid.uuid4().hex[:10]}"
    return {"job_id": jobs.submit(pid, "llm", "eval", cfg, out), "contaminated": cfg["contaminated"]}


@app.get("/projects/{pid}/evals")
def list_evals(pid: int, engine: str = "llm"):
    out = []
    for j in db.rows("SELECT * FROM jobs WHERE project_id=? AND kind='eval' AND engine=? ORDER BY id DESC", (pid, engine)):
        cfg = json.loads(j["config"])
        m = db.one("SELECT data FROM metrics WHERE job_id=? ORDER BY id DESC LIMIT 1", (j["id"],))
        result = json.loads(m["data"]).get("result") if m else None
        out.append({"id": j["id"], "status": j["status"], "error": j["error"], "label": cfg.get("label"),
                    "base_model": cfg.get("base_model"), "dataset_id": cfg.get("dataset_id"),
                    "dataset_name": cfg.get("dataset_name"), "contaminated": cfg.get("contaminated", False),
                    "result": result})
    return out


# =====================================================================
# Visión: clasificación de imágenes
# =====================================================================
VISION_MODELS = [
    {"id": "resnet18.a1_in1k", "name": "ResNet-18", "params_m": 11.7, "note": "Rápido y ligero; buen punto de partida"},
    {"id": "mobilenetv3_large_100.ra_in1k", "name": "MobileNetV3 Large", "params_m": 5.5, "note": "Muy ligero, ideal para CPU"},
    {"id": "efficientnet_b0.ra_in1k", "name": "EfficientNet-B0", "params_m": 5.3, "note": "Buen equilibrio precisión/tamaño"},
    {"id": "resnet50.a1_in1k", "name": "ResNet-50", "params_m": 25.6, "note": "Más preciso, más lento"},
]


@app.get("/vision/models")
def vision_models():
    return VISION_MODELS


class ImageDatasetIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    source_dir: str
    license: str = ""


@app.post("/projects/{pid}/datasets/images")
def import_image_dataset(pid: int, d: ImageDatasetIn):
    _project_or_404(pid)
    src = Path(d.source_dir)
    if not src.is_dir():
        raise HTTPException(400, "Indica una carpeta existente con una subcarpeta por clase.")
    dst = config.PROJECTS_DIR / str(pid) / "datasets" / f"img_{uuid.uuid4().hex[:10]}"
    try:
        report = datasets.import_imagefolder(src, dst)
    except ValueError as e:
        shutil.rmtree(dst, ignore_errors=True)
        raise HTTPException(400, str(e))
    did = db.execute(
        "INSERT INTO datasets(project_id,name,path,rows,license,report,created_at,kind) VALUES(?,?,?,?,?,?,?,'images')",
        (pid, d.name.strip(), str(dst), report["total"], d.license, db.dumps(report), db.now()),
    )
    return {"id": did, "rows": report["total"], "report": report}


class VisionTrainIn(BaseModel):
    dataset_id: int
    model_name: str = "resnet18.a1_in1k"
    epochs: int = Field(10, ge=1, le=200)
    batch_size: int = Field(32, ge=1, le=512)
    learning_rate: float = Field(1e-3, gt=0, lt=1)
    img_size: int = Field(160, ge=32, le=640)
    val_split: float = Field(0.2, gt=0, lt=0.9)
    freeze_backbone: bool = False
    augment: bool = True
    resume_job_id: int | None = None


def _image_dataset(pid: int, did: int) -> dict:
    ds = db.one("SELECT * FROM datasets WHERE id=? AND project_id=? AND kind='images'", (did, pid))
    if not ds:
        raise HTTPException(404, "Dataset de imágenes no encontrado en este proyecto.")
    return ds


def _vision_ready() -> None:
    if not envs.is_ready("vision"):
        raise HTTPException(409, "Instala primero el entorno del motor de visión.")


@app.post("/projects/{pid}/vision/train")
def vision_train(pid: int, t: VisionTrainIn):
    _project_or_404(pid)
    ds = _image_dataset(pid, t.dataset_id)
    _vision_ready()
    cfg = t.model_dump()
    cfg["dataset_dir"] = ds["path"]
    out = config.PROJECTS_DIR / str(pid) / "runs" / f"vision_{uuid.uuid4().hex[:10]}"
    if t.resume_job_id:
        prev = db.one("SELECT output_dir FROM jobs WHERE id=? AND project_id=? AND engine='vision'", (t.resume_job_id, pid))
        if not prev:
            raise HTTPException(404, "Entrenamiento a reanudar no encontrado.")
        out = Path(prev["output_dir"])
    return {"job_id": jobs.submit(pid, "vision", "train", cfg, out)}


class VisionEvalIn(BaseModel):
    dataset_id: int
    job_id: int
    max_per_class: int = Field(200, ge=1, le=5000)


@app.post("/projects/{pid}/vision/eval")
def vision_eval(pid: int, e: VisionEvalIn):
    _project_or_404(pid)
    ds = _image_dataset(pid, e.dataset_id)
    _vision_ready()
    src = db.one("SELECT * FROM jobs WHERE id=? AND project_id=? AND engine='vision' AND kind='train' AND status='completed'",
                 (e.job_id, pid))
    if not src:
        raise HTTPException(400, "El entrenamiento indicado no existe o no está completado.")
    scfg = json.loads(src["config"])
    cfg = {"dataset_dir": ds["path"], "model_dir": src["output_dir"], "max_per_class": e.max_per_class,
           "dataset_id": e.dataset_id, "dataset_name": ds["name"], "target_job_id": e.job_id,
           "label": f"Entrenamiento #{src['id']} · {scfg['model_name']}",
           "contaminated": scfg["dataset_dir"] == ds["path"]}
    out = config.PROJECTS_DIR / str(pid) / "evals" / f"vision_eval_{uuid.uuid4().hex[:10]}"
    return {"job_id": jobs.submit(pid, "vision", "eval", cfg, out), "contaminated": cfg["contaminated"]}


class VisionExportIn(BaseModel):
    job_id: int


@app.post("/projects/{pid}/vision/export")
def vision_export(pid: int, e: VisionExportIn):
    src = db.one("SELECT * FROM jobs WHERE id=? AND project_id=? AND engine='vision' AND kind='train'", (e.job_id, pid))
    if not src or src["status"] != "completed":
        raise HTTPException(400, "Solo se pueden exportar entrenamientos completados.")
    _vision_ready()
    out = Path(src["output_dir"]) / "export"
    return {"job_id": jobs.submit(pid, "vision", "export", {"model_dir": src["output_dir"], "out_dir": str(out)}, out)}


class VisionLoadIn(BaseModel):
    job_id: int


class VisionPredictIn(BaseModel):
    image_b64: str = Field(max_length=30_000_000)


@app.post("/vision/load")
def vision_load(b: VisionLoadIn):
    j = db.one("SELECT * FROM jobs WHERE id=? AND engine='vision' AND kind='train' AND status='completed'", (b.job_id,))
    if not j:
        raise HTTPException(400, "Entrenamiento no encontrado o no completado.")
    try:
        return vision_infer.load(j["output_dir"])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@app.post("/vision/predict")
def vision_predict(b: VisionPredictIn):
    try:
        return {"top": vision_infer.predict(b.image_b64)}
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@app.post("/vision/unload")
def vision_unload():
    vision_infer.unload()
    return {"loaded": False}


# =====================================================================
# Registro de modelos
# =====================================================================
def _dir_size_mb(p: Path) -> float:
    if not p.exists():
        return 0.0
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**20, 1)


def _artifacts(job: dict) -> list[dict]:
    """Archivos útiles producidos por un entrenamiento (adaptador, modelo fusionado, GGUF, ONNX...)."""
    base = Path(job["output_dir"])
    out = []
    candidates = ["adapter", "lora", "model.pt", "model", "export/merged", "export/model.onnx"]
    candidates += [str(f.relative_to(base)) for f in base.glob("export/*.gguf")]
    for rel in candidates:
        p = base / rel
        if p.exists():
            size = _dir_size_mb(p) if p.is_dir() else round(p.stat().st_size / 2**20, 1)
            out.append({"name": rel.replace("\\", "/"), "path": str(p), "size_mb": size})
    return out


@app.get("/projects/{pid}/models")
def list_models(pid: int):
    out = []
    for j in db.rows("SELECT * FROM jobs WHERE project_id=? AND kind='train' ORDER BY id DESC", (pid,)):
        cfg = json.loads(j["config"])
        # El último evento de Trainer es un resumen sin 'loss'; se busca la última métrica de entrenamiento real.
        last = db.one("SELECT data FROM metrics WHERE job_id=? AND data LIKE '%\"loss\"%' ORDER BY id DESC LIMIT 1", (j["id"],))
        m = json.loads(last["data"]) if last else {}
        base = cfg.get("base_model") or cfg.get("model_name")
        out.append({
            "id": j["id"], "engine": j["engine"], "status": j["status"], "base": base,
            "created_at": j["created_at"], "output_dir": j["output_dir"],
            "size_mb": _dir_size_mb(Path(j["output_dir"])),
            "artifacts": _artifacts(j) if j["status"] == "completed" else [],
            "final": {k: m[k] for k in ("step", "loss", "eval_loss", "val_acc") if k in m},
            "hub_id": f"timm/{base}" if j["engine"] == "vision" else base,
        })
    return out


@app.delete("/jobs/{jid}")
def delete_job(jid: int):
    """Borra un trabajo, sus métricas y su carpeta de salida (irreversible)."""
    j = db.one("SELECT * FROM jobs WHERE id=?", (jid,))
    if not j:
        raise HTTPException(404, "Job no encontrado.")
    if j["status"] in ("queued", "running"):
        raise HTTPException(409, "Detén el trabajo antes de borrarlo.")
    out = Path(j["output_dir"]).resolve()
    if config.PROJECTS_DIR.resolve() not in out.parents:  # nunca borrar fuera de data/projects
        raise HTTPException(400, "Ruta de salida fuera del área de proyectos.")
    # Al reanudar, otro trabajo comparte la misma carpeta: en ese caso solo se borra el registro.
    shared = db.one("SELECT 1 FROM jobs WHERE id!=? AND output_dir=?", (jid, j["output_dir"]))
    ids = [jid]
    if not shared:
        # Los trabajos hijos (exportaciones) viven dentro de la carpeta y desaparecen con ella.
        children = [r["id"] for r in db.rows("SELECT id, output_dir FROM jobs WHERE id!=?", (jid,))
                    if out in Path(r["output_dir"]).resolve().parents]
        ids += children
        shutil.rmtree(out, ignore_errors=True)
    for i in ids:
        for t in ("metrics", "logs"):
            db.execute(f"DELETE FROM {t} WHERE job_id=?", (i,))
        db.execute("DELETE FROM jobs WHERE id=?", (i,))
    return {"deleted": ids, "files_removed": not shared}


# =====================================================================
# Audio: transcripción y fine-tuning de Whisper
# =====================================================================
AUDIO_MODELS = [
    {"id": "openai/whisper-tiny", "name": "Whisper tiny", "params_m": 39, "note": "Muy rápido; precisión limitada"},
    {"id": "openai/whisper-base", "name": "Whisper base", "params_m": 74, "note": "Buen punto de partida"},
    {"id": "openai/whisper-small", "name": "Whisper small", "params_m": 244, "note": "Más preciso; ajustado para 8 GB de VRAM"},
]
AUDIO_LANGS = {"spanish": "Español", "english": "Inglés", "french": "Francés", "german": "Alemán",
               "portuguese": "Portugués", "italian": "Italiano"}


@app.get("/audio/models")
def audio_models():
    return {"models": AUDIO_MODELS, "languages": AUDIO_LANGS}


class AudioDatasetIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    source_dir: str
    license: str = ""


@app.post("/projects/{pid}/datasets/audio")
def import_audio_dataset(pid: int, d: AudioDatasetIn):
    _project_or_404(pid)
    src = Path(d.source_dir)
    if not src.is_dir():
        raise HTTPException(400, "Indica una carpeta existente con los clips y metadata.csv.")
    dst = config.PROJECTS_DIR / str(pid) / "datasets" / f"audio_{uuid.uuid4().hex[:10]}"
    try:
        report = datasets.import_audiofolder(src, dst)
    except ValueError as e:
        shutil.rmtree(dst, ignore_errors=True)
        raise HTTPException(400, str(e))
    did = db.execute(
        "INSERT INTO datasets(project_id,name,path,rows,license,report,created_at,kind) VALUES(?,?,?,?,?,?,?,'audio')",
        (pid, d.name.strip(), str(dst), report["clips"], d.license, db.dumps(report), db.now()),
    )
    return {"id": did, "rows": report["clips"], "report": report}


def _audio_dataset(pid: int, did: int) -> dict:
    ds = db.one("SELECT * FROM datasets WHERE id=? AND project_id=? AND kind='audio'", (did, pid))
    if not ds:
        raise HTTPException(404, "Dataset de audio no encontrado en este proyecto.")
    return ds


def _audio_ready() -> None:
    if not envs.is_ready("audio"):
        raise HTTPException(409, "Instala primero el entorno del motor de audio.")


def _check_lang(lang: str) -> None:
    if lang not in AUDIO_LANGS:
        raise HTTPException(400, f"Idioma no soportado: {lang}")


class AudioTrainIn(BaseModel):
    dataset_id: int
    base_model: str = "openai/whisper-tiny"
    language: str = "spanish"
    epochs: int = Field(5, ge=1, le=100)
    batch_size: int = Field(8, ge=1, le=64)
    grad_accum: int = Field(1, ge=1, le=64)
    learning_rate: float = Field(1e-5, gt=0, lt=1)
    val_split: float = Field(0.15, gt=0, lt=0.9)
    freeze_encoder: bool = False
    gradient_checkpointing: bool = False
    max_val_clips: int = Field(100, ge=1, le=1000)
    resume_job_id: int | None = None


@app.post("/projects/{pid}/audio/train")
def audio_train(pid: int, t: AudioTrainIn):
    _project_or_404(pid)
    ds = _audio_dataset(pid, t.dataset_id)
    _check_lang(t.language)
    _audio_ready()
    cfg = t.model_dump()
    cfg["dataset_dir"] = ds["path"]
    out = config.PROJECTS_DIR / str(pid) / "runs" / f"audio_{uuid.uuid4().hex[:10]}"
    if t.resume_job_id:
        prev = db.one("SELECT output_dir FROM jobs WHERE id=? AND project_id=? AND engine='audio'", (t.resume_job_id, pid))
        if not prev:
            raise HTTPException(404, "Entrenamiento a reanudar no encontrado.")
        out = Path(prev["output_dir"])
    return {"job_id": jobs.submit(pid, "audio", "train", cfg, out)}


class AudioEvalIn(BaseModel):
    dataset_id: int
    job_id: int | None = None  # None = modelo base sin ajustar
    base_model: str = "openai/whisper-tiny"
    language: str = "spanish"
    max_clips: int = Field(100, ge=1, le=2000)


@app.post("/projects/{pid}/audio/eval")
def audio_eval(pid: int, e: AudioEvalIn):
    _project_or_404(pid)
    ds = _audio_dataset(pid, e.dataset_id)
    _check_lang(e.language)
    _audio_ready()
    cfg = {"dataset_dir": ds["path"], "model_path": e.base_model, "language": e.language, "max_clips": e.max_clips,
           "dataset_id": e.dataset_id, "dataset_name": ds["name"], "target_job_id": e.job_id,
           "base_model": e.base_model, "label": f"Modelo base · {e.base_model.split('/')[-1]}", "contaminated": False}
    if e.job_id is not None:
        src = db.one("SELECT * FROM jobs WHERE id=? AND project_id=? AND engine='audio' AND kind='train' AND status='completed'",
                     (e.job_id, pid))
        if not src:
            raise HTTPException(400, "El entrenamiento indicado no existe o no está completado.")
        scfg = json.loads(src["config"])
        cfg.update(model_path=str(Path(src["output_dir"]) / "model"), base_model=scfg["base_model"],
                   label=f"Entrenamiento #{src['id']} · {scfg['base_model'].split('/')[-1]}",
                   contaminated=scfg["dataset_dir"] == ds["path"])
    out = config.PROJECTS_DIR / str(pid) / "evals" / f"audio_eval_{uuid.uuid4().hex[:10]}"
    return {"job_id": jobs.submit(pid, "audio", "eval", cfg, out), "contaminated": cfg["contaminated"]}


class AudioLoadIn(BaseModel):
    job_id: int | None = None
    base_model: str = "openai/whisper-tiny"


class AudioTranscribeIn(BaseModel):
    audio_b64: str = Field(max_length=60_000_000)
    language: str | None = "spanish"


@app.post("/audio/load")
def audio_load(b: AudioLoadIn):
    serving.invalidate()
    path = b.base_model
    if b.job_id is not None:
        j = db.one("SELECT * FROM jobs WHERE id=? AND engine='audio' AND kind='train' AND status='completed'", (b.job_id,))
        if not j:
            raise HTTPException(400, "Entrenamiento no encontrado o no completado.")
        path = str(Path(j["output_dir"]) / "model")
    if db.one("SELECT 1 FROM jobs WHERE status='running'"):
        raise HTTPException(409, "Hay un trabajo en curso; la GPU está ocupada.")
    try:
        audio_infer.load(path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))
    return {"loaded": True, "model": path}


@app.post("/audio/transcribe")
def audio_transcribe(b: AudioTranscribeIn):
    if b.language and b.language != "auto":
        _check_lang(b.language)
    try:
        return audio_infer.transcribe(b.audio_b64, b.language)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@app.post("/audio/unload")
def audio_unload():
    serving.invalidate()
    audio_infer.unload()
    return {"loaded": False}


# =====================================================================
# Imagen generativa: LoRA de Stable Diffusion 1.x
# =====================================================================
SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGE_MODELS = [
    {"id": SD15, "name": "Stable Diffusion 1.5", "license": "creativeml-openrail-m",
     "note": "Entrena LoRA con ~6 GB de VRAM a 512 px. SDXL no cabe en 8 GB."},
]
IMG_FILE_EXT = {".png", ".jpg", ".jpeg", ".webp"}


@app.get("/image/models")
def image_models():
    return IMAGE_MODELS


class ImageCapsIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    source_dir: str
    default_caption: str = ""
    license: str = ""


@app.post("/projects/{pid}/datasets/imagecaps")
def import_imagecaps_dataset(pid: int, d: ImageCapsIn):
    _project_or_404(pid)
    src = Path(d.source_dir)
    if not src.is_dir():
        raise HTTPException(400, "Indica una carpeta existente con las imágenes.")
    dst = config.PROJECTS_DIR / str(pid) / "datasets" / f"imgcap_{uuid.uuid4().hex[:10]}"
    try:
        report = datasets.import_imagecaps(src, dst, d.default_caption)
    except ValueError as e:
        shutil.rmtree(dst, ignore_errors=True)
        raise HTTPException(400, str(e))
    did = db.execute(
        "INSERT INTO datasets(project_id,name,path,rows,license,report,created_at,kind) VALUES(?,?,?,?,?,?,?,'imagegen')",
        (pid, d.name.strip(), str(dst), report["images"], d.license, db.dumps(report), db.now()),
    )
    return {"id": did, "rows": report["images"], "report": report}


def _imagegen_dataset(pid: int, did: int) -> dict:
    ds = db.one("SELECT * FROM datasets WHERE id=? AND project_id=? AND kind='imagegen'", (did, pid))
    if not ds:
        raise HTTPException(404, "Dataset de imágenes con descripción no encontrado en este proyecto.")
    return ds


def _image_ready() -> None:
    if not envs.is_ready("image"):
        raise HTTPException(409, "Instala primero el entorno del motor de imagen.")


class ImageTrainIn(BaseModel):
    dataset_id: int
    base_model: str = SD15
    resolution: int = Field(512, ge=256, le=768, multiple_of=64)
    max_steps: int = Field(300, ge=10, le=20000)
    batch_size: int = Field(1, ge=1, le=8)
    grad_accum: int = Field(4, ge=1, le=64)
    learning_rate: float = Field(1e-4, gt=0, lt=1)
    lora_rank: int = Field(8, ge=1, le=128)
    lora_alpha: int = Field(8, ge=1, le=256)
    save_steps: int = Field(100, ge=1)
    sample_steps: int = Field(100, ge=0)
    validation_prompt: str = ""
    seed: int = 42
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
    random_flip: bool = True
    center_crop: bool = True
    resume_job_id: int | None = None


@app.post("/projects/{pid}/image/train")
def image_train(pid: int, t: ImageTrainIn):
    _project_or_404(pid)
    ds = _imagegen_dataset(pid, t.dataset_id)
    _image_ready()
    cfg = t.model_dump()
    cfg["dataset_dir"] = ds["path"]
    if not cfg["validation_prompt"].strip():  # por defecto, la primera descripción del dataset
        with open(Path(ds["path"]) / "metadata.jsonl", encoding="utf-8") as f:
            cfg["validation_prompt"] = json.loads(f.readline())["text"]
    out = config.PROJECTS_DIR / str(pid) / "runs" / f"image_{uuid.uuid4().hex[:10]}"
    if t.resume_job_id:
        prev = db.one("SELECT output_dir FROM jobs WHERE id=? AND project_id=? AND engine='image'", (t.resume_job_id, pid))
        if not prev:
            raise HTTPException(404, "Entrenamiento a reanudar no encontrado.")
        out = Path(prev["output_dir"])
    return {"job_id": jobs.submit(pid, "image", "train", cfg, out)}


class ImageCompareIn(BaseModel):
    job_ids: list[int] = Field(min_length=1, max_length=4)
    prompts: list[str] = Field(min_length=1, max_length=8)
    seeds: list[int] = Field(default=[1, 2], min_length=1, max_length=4)
    base_model: str = SD15
    steps: int = Field(25, ge=5, le=100)
    guidance: float = Field(7.5, ge=1, le=20)
    size: int = Field(512, ge=256, le=768, multiple_of=64)
    lora_scale: float = Field(1.0, ge=0, le=2)
    negative_prompt: str = ""


@app.post("/projects/{pid}/image/compare")
def image_compare(pid: int, c: ImageCompareIn):
    _project_or_404(pid)
    _image_ready()
    loras = []
    for jid in c.job_ids:
        j = db.one("SELECT * FROM jobs WHERE id=? AND project_id=? AND engine='image' AND kind='train' AND status='completed'",
                   (jid, pid))
        if not j:
            raise HTTPException(400, f"El entrenamiento #{jid} no existe o no está completado.")
        loras.append({"label": f"LoRA #{jid}", "path": str(Path(j["output_dir"]) / "lora")})
    cfg = c.model_dump(exclude={"job_ids"}) | {"loras": loras, "label": f"Comparación · {len(loras)} LoRA"}
    out = config.PROJECTS_DIR / str(pid) / "evals" / f"image_cmp_{uuid.uuid4().hex[:10]}"
    return {"job_id": jobs.submit(pid, "image", "eval", cfg, out)}


@app.get("/jobs/{jid}/file")
def job_file(jid: int, name: str):
    """Sirve una imagen producida por un trabajo (muestras de entrenamiento, comparaciones)."""
    j = db.one("SELECT output_dir FROM jobs WHERE id=?", (jid,))
    if not j:
        raise HTTPException(404, "Job no encontrado.")
    base = Path(j["output_dir"]).resolve()
    p = (base / name).resolve()
    if base not in p.parents or p.suffix.lower() not in IMG_FILE_EXT or not p.is_file():
        raise HTTPException(404, "Archivo no encontrado.")
    return FileResponse(p)


class ImageLoadIn(BaseModel):
    job_id: int | None = None  # None = solo el modelo base
    base_model: str = SD15
    lora_scale: float = Field(1.0, ge=0, le=2)


class ImageGenIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    negative_prompt: str = ""
    steps: int = Field(25, ge=5, le=100)
    guidance: float = Field(7.5, ge=1, le=20)
    seed: int = 1
    width: int = Field(512, ge=128, le=768, multiple_of=8)
    height: int = Field(512, ge=128, le=768, multiple_of=8)
    n: int = Field(1, ge=1, le=4)


@app.post("/image/load")
def image_load(b: ImageLoadIn):
    serving.invalidate()
    lora = None
    base = b.base_model
    if b.job_id is not None:
        j = db.one("SELECT * FROM jobs WHERE id=? AND engine='image' AND kind='train' AND status='completed'", (b.job_id,))
        if not j:
            raise HTTPException(400, "Entrenamiento no encontrado o no completado.")
        lora = str(Path(j["output_dir"]) / "lora")
        base = json.loads(j["config"])["base_model"]
    if db.one("SELECT 1 FROM jobs WHERE status='running'"):
        raise HTTPException(409, "Hay un trabajo en curso; la GPU está ocupada.")
    try:
        return image_infer.load(base, lora, b.lora_scale)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@app.post("/image/generate")
def image_generate(g: ImageGenIn):
    try:
        return image_infer.generate(g.model_dump())
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@app.post("/image/unload")
def image_unload():
    serving.invalidate()
    image_infer.unload()
    return {"loaded": False}


# =====================================================================
# Información de la API local (para la interfaz; no requiere clave)
# =====================================================================
@app.get("/api-info")
def api_info():
    import os

    return {"base_url": "http://127.0.0.1:8765/v1", "key_required": bool(os.environ.get("NOVA_API_KEY")),
            "models": [e.public() for e in serving.list_entries()], "loaded": serving.loaded()}


@app.post("/api-info/unload")
def api_unload():
    serving.unload_all()
    return {"loaded": None}
