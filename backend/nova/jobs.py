"""Cola de trabajos: un job de GPU a la vez, cada uno en un subproceso del motor."""
from __future__ import annotations

import json
import queue
import re
import subprocess
import threading
from pathlib import Path

from nova import config, db, envs

_q: "queue.Queue[int]" = queue.Queue()
_procs: dict[int, subprocess.Popen] = {}
_stop_requested: set[int] = set()
_started = False


def start() -> None:
    global _started
    if _started:
        return
    _started = True
    # Los jobs que quedaron "running" por un cierre brusco se marcan como interrumpidos.
    db.execute("UPDATE jobs SET status='interrupted', finished_at=? WHERE status IN ('running','queued')", (db.now(),))
    threading.Thread(target=_worker, daemon=True, name="job-worker").start()


def submit(project_id: int, engine: str, kind: str, cfg: dict, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    jid = db.execute(
        "INSERT INTO jobs(project_id,engine,kind,config,status,output_dir,created_at) VALUES(?,?,?,?,?,?,?)",
        (project_id, engine, kind, db.dumps(cfg), "queued", str(output_dir), db.now()),
    )
    _q.put(jid)
    return jid


def stop(jid: int) -> bool:
    p = _procs.get(jid)
    if p and p.poll() is None:
        _stop_requested.add(jid)
        # taskkill /T cierra también los procesos hijos (dataloaders, etc.)
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        return True
    job = db.one("SELECT status FROM jobs WHERE id=?", (jid,))
    if job and job["status"] == "queued":
        db.execute("UPDATE jobs SET status='stopped', finished_at=? WHERE id=?", (db.now(), jid))
        return True
    return False


def _worker() -> None:
    while True:
        jid = _q.get()
        job = db.one("SELECT * FROM jobs WHERE id=?", (jid,))
        if not job or job["status"] != "queued":
            continue
        try:
            _run(job)
        except Exception as e:  # noqa: BLE001
            db.execute("UPDATE jobs SET status='failed', error=?, finished_at=? WHERE id=?", (str(e), db.now(), jid))


# Barras de progreso (tqdm) y avisos de torch con marca de tiempo: ruido que no aporta al usuario.
_NOISE = re.compile(r"%\||^W\d{4} \d\d:\d\d:\d\d|^\s*$|unauthenticated requests|file_download\.py|HF_HUB_DISABLE_SYMLINKS_WARNING|warnings\.warn\(message\)|enable developer mode|To support symlinks on Windows")


def parse_line(raw: str) -> tuple[str, object]:
    """Clasifica una línea de salida de un motor: ('metric'|'error'|'log'|'skip', contenido)."""
    line = raw.rstrip()
    try:
        ev = json.loads(line)
    except json.JSONDecodeError:
        ev = None
    if isinstance(ev, dict):
        t = ev.get("type")
        if t in ("metric", "result", "sample"):
            return "metric", ev
        if t == "error":
            return "error", ev.get("message")
        if t == "log":
            return "log", str(ev.get("message", ""))[:2000]
        if t == "done":
            return "skip", None
        return "log", line[:2000]
    if _NOISE.search(line):
        return "skip", None
    return "log", line[:2000]


def _run(job: dict) -> None:
    jid = job["id"]
    engine = job["engine"]
    if not envs.is_ready(engine):
        raise RuntimeError(f"El entorno del motor '{engine}' no está instalado.")
    cfg_path = Path(job["output_dir"]) / "job_config.json"
    cfg_path.write_text(job["config"], encoding="utf-8")
    script = config.ENGINES_DIR / engine / f"{job['kind']}_worker.py"
    db.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?", (db.now(), jid))
    p = subprocess.Popen(
        [str(envs.python_exe(engine)), "-u", str(script), str(cfg_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
        env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8", "HF_HOME": str(config.MODELS_DIR / "hf"),
             "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"},
    )
    _procs[jid] = p
    error = None
    for raw in p.stdout:  # type: ignore[union-attr]
        kind, payload = parse_line(raw)
        if kind == "metric":
            db.execute("INSERT INTO metrics(job_id,ts,step,data) VALUES(?,?,?,?)",
                       (jid, db.now(), payload.get("step"), db.dumps(payload)))
        elif kind == "error":
            error = payload
        elif kind == "log":
            db.execute("INSERT INTO logs(job_id,ts,line) VALUES(?,?,?)", (jid, db.now(), payload))
    code = p.wait()
    _procs.pop(jid, None)
    if jid in _stop_requested:
        _stop_requested.discard(jid)
        status = "stopped"
    elif code == 0:
        status = "completed"
    else:
        status = "failed"
        error = error or f"El proceso terminó con código {code}"
    db.execute("UPDATE jobs SET status=?, error=?, finished_at=? WHERE id=?", (status, error, db.now(), jid))
