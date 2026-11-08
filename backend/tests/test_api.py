import json

import pytest
from fastapi.testclient import TestClient

from nova.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_hardware(client):
    data = client.get("/hardware").json()
    assert data["ram_total_mb"] > 0 and data["cpu_cores"] > 0


def test_project_and_dataset_flow(client, tmp_path):
    p = client.post("/projects", json={"name": "Prueba"}).json()
    src = tmp_path / "d.jsonl"
    rows = [
        {"instruction": "Hola", "output": "Buenas"},
        {"instruction": "Hola", "output": "Buenas"},          # duplicado
        {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]},
        {"instruction": "", "output": "vacío"},               # inválido
    ]
    src.write_text("\n".join(json.dumps(r) for r in rows) + "\nno es json\n", encoding="utf-8")
    r = client.post(f"/projects/{p['id']}/datasets", json={"name": "d", "source_path": str(src)}).json()
    assert r["rows"] == 2
    assert r["report"]["total"] == 5 and r["report"]["invalid"] == 2 and r["report"]["duplicates"] == 1
    prev = client.get(f"/datasets/{r['id']}/preview").json()
    assert prev[0]["messages"][-1]["role"] == "assistant"


def test_dataset_rejects_bad_path(client):
    p = client.post("/projects", json={"name": "X"}).json()
    assert client.post(f"/projects/{p['id']}/datasets", json={"name": "d", "source_path": "C:/nada.jsonl"}).status_code == 400


def test_train_requires_env(client, tmp_path):
    p = client.post("/projects", json={"name": "T"}).json()
    src = tmp_path / "d.jsonl"
    src.write_text(json.dumps({"instruction": "a", "output": "b"}) + "\n", encoding="utf-8")
    d = client.post(f"/projects/{p['id']}/datasets", json={"name": "d", "source_path": str(src)}).json()
    r = client.post(f"/projects/{p['id']}/train", json={"dataset_id": d["id"]})
    assert r.status_code == 409  # el entorno de test no tiene el motor instalado


def _make_images(root, per_class):
    from PIL import Image

    for cls, col in (("rojo", (255, 0, 0)), ("azul", (0, 0, 255))):
        (root / cls).mkdir(parents=True)
        for i in range(per_class):
            Image.new("RGB", (16, 16), col).save(root / cls / f"{i}.png")


def test_image_datasets_do_not_collide(client, tmp_path):
    """Regresión: dos imports en el mismo segundo no deben compartir carpeta."""
    p = client.post("/projects", json={"name": "Img"}).json()
    a, b = tmp_path / "a", tmp_path / "b"
    _make_images(a, 6)
    _make_images(b, 2)
    ra = client.post(f"/projects/{p['id']}/datasets/images", json={"name": "a", "source_dir": str(a)}).json()
    rb = client.post(f"/projects/{p['id']}/datasets/images", json={"name": "b", "source_dir": str(b)}).json()
    assert (ra["rows"], rb["rows"]) == (12, 4)
    ds = {d["id"]: d for d in client.get(f"/projects/{p['id']}/datasets?kind=images").json()}
    assert ds[ra["id"]]["path"] != ds[rb["id"]]["path"]
    from pathlib import Path
    assert len(list(Path(ds[ra["id"]]["path"]).rglob("*.png"))) == 12  # no fue sobrescrito por b


def test_image_import_needs_two_classes(client, tmp_path):
    from PIL import Image

    p = client.post("/projects", json={"name": "Una"}).json()
    (tmp_path / "solo").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "solo" / "x.png")
    r = client.post(f"/projects/{p['id']}/datasets/images", json={"name": "x", "source_dir": str(tmp_path)})
    assert r.status_code == 400


def _job(pid, out, status="completed", kind="train"):
    from nova import db

    return db.execute(
        "INSERT INTO jobs(project_id,engine,kind,config,status,output_dir,created_at) VALUES(?,?,?,?,?,?,?)",
        (pid, "vision", kind, "{}", status, str(out), db.now()))


def test_delete_job_keeps_shared_folder_and_removes_children(client):
    from nova import config

    pid = client.post("/projects", json={"name": "Del"}).json()["id"]
    run = config.PROJECTS_DIR / str(pid) / "runs" / "r1"
    (run / "export").mkdir(parents=True)
    (run / "model.pt").write_text("x")
    first = _job(pid, run, "stopped")
    resumed = _job(pid, run)                      # reanudación: misma carpeta
    child = _job(pid, run / "export", kind="export")

    r = client.delete(f"/jobs/{first}").json()
    assert r["files_removed"] is False and (run / "model.pt").exists()  # el otro job sigue usándola

    r = client.delete(f"/jobs/{resumed}").json()
    assert r["files_removed"] is True and not run.exists()
    assert child in r["deleted"]                  # la exportación hija desaparece con la carpeta
    assert client.get(f"/jobs/{child}").status_code == 404


def test_delete_job_refuses_paths_outside_projects(client, tmp_path):
    pid = client.post("/projects", json={"name": "Fuera"}).json()["id"]
    victim = tmp_path / "importante"
    victim.mkdir()
    (victim / "dato.txt").write_text("no borrar")
    jid = _job(pid, victim)
    assert client.delete(f"/jobs/{jid}").status_code == 400
    assert (victim / "dato.txt").exists()


def test_delete_running_job_is_refused(client):
    from nova import config

    pid = client.post("/projects", json={"name": "Run"}).json()["id"]
    jid = _job(pid, config.PROJECTS_DIR / str(pid) / "runs" / "x", "running")
    assert client.delete(f"/jobs/{jid}").status_code == 409


def _write_wav(path, seconds, sr=16000):
    import numpy as np
    import soundfile as sf

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * 440 * t)).astype("float32"), sr)


def test_audio_import_validates_clips(client, tmp_path):
    import csv
    from pathlib import Path

    src = tmp_path / "voz"
    src.mkdir()
    _write_wav(src / "ok1.wav", 1.0)
    _write_wav(src / "ok2.wav", 2.0)
    _write_wav(src / "corto.wav", 0.1)           # demasiado corto
    (src / "roto.wav").write_bytes(b"no es audio")
    rows = [("ok1.wav", "hola mundo"), ("ok2.wav", "otra frase"), ("corto.wav", "x"), ("roto.wav", "y"),
            ("falta.wav", "z"), ("ok1.wav", "   ")]
    with (src / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "text"])
        w.writerows(rows)
    p = client.post("/projects", json={"name": "Aud"}).json()
    r = client.post(f"/projects/{p['id']}/datasets/audio", json={"name": "v", "source_dir": str(src)}).json()
    assert r["rows"] == 2 and r["report"]["invalid"] == 4
    ds = client.get(f"/projects/{p['id']}/datasets?kind=audio").json()[0]
    meta = [json.loads(line) for line in (Path(ds["path"]) / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [m["text"] for m in meta] == ["hola mundo", "otra frase"]
    assert all((Path(ds["path"]) / m["file"]).exists() for m in meta)


def test_audio_import_requires_metadata(client, tmp_path):
    p = client.post("/projects", json={"name": "SinMeta"}).json()
    assert client.post(f"/projects/{p['id']}/datasets/audio", json={"name": "v", "source_dir": str(tmp_path)}).status_code == 400


def test_parse_line_classifies_engine_output():
    from nova.jobs import parse_line

    assert parse_line('{"type": "metric", "step": 3, "loss": 0.5}\n') == ("metric", {"type": "metric", "step": 3, "loss": 0.5})
    assert parse_line('{"type": "result", "result": {}}')[0] == "metric"
    assert parse_line('{"type": "error", "message": "boom"}') == ("error", "boom")
    assert parse_line('{"type": "log", "message": "CUDA disponible"}') == ("log", "CUDA disponible")
    assert parse_line('{"type": "done"}')[0] == "skip"
    assert parse_line("Loading weights:  18%|##        | 53/290")[0] == "skip"      # tqdm
    assert parse_line("W1107 15:58:29.090 Lib\\torch\\x.py:29] NOTE")[0] == "skip"  # aviso de torch
    assert parse_line("   ")[0] == "skip"
    assert parse_line("Traceback (most recent call last):") == ("log", "Traceback (most recent call last):")
    assert parse_line("42")[0] == "log"  # JSON válido pero no objeto: no debe romper


def test_delete_project_leaves_no_orphans(client):
    from nova import config, db

    pid = client.post("/projects", json={"name": "Huérfanos"}).json()["id"]
    jid = _job(pid, config.PROJECTS_DIR / str(pid) / "runs" / "a")
    db.execute("INSERT INTO metrics(job_id,ts,step,data) VALUES(?,?,?,?)", (jid, db.now(), 1, "{}"))
    db.execute("INSERT INTO logs(job_id,ts,line) VALUES(?,?,?)", (jid, db.now(), "x"))
    assert client.delete(f"/projects/{pid}").json() == {"ok": True}
    assert db.one("SELECT 1 FROM metrics WHERE job_id=?", (jid,)) is None
    assert db.one("SELECT 1 FROM logs WHERE job_id=?", (jid,)) is None
    assert db.one("SELECT 1 FROM jobs WHERE id=?", (jid,)) is None


def _png(path, size=(64, 64), color=(10, 20, 30)):
    from PIL import Image

    Image.new("RGB", size, color).save(path)


def test_imagecaps_prefers_metadata_then_txt_then_default(client, tmp_path):
    import json as _json
    from pathlib import Path

    src = tmp_path / "estilo"
    src.mkdir()
    for n in ("a", "b", "c", "d"):
        _png(src / f"{n}.png")
    (src / "metadata.csv").write_text("file,text\na.png,desde csv\n", encoding="utf-8")
    (src / "b.txt").write_text("desde txt", encoding="utf-8")
    (src / "roto.png").write_bytes(b"no es imagen")
    p = client.post("/projects", json={"name": "Caps"}).json()
    # c y d no tienen descripción propia: sin valor por defecto se descartan
    r = client.post(f"/projects/{p['id']}/datasets/imagecaps", json={"name": "x", "source_dir": str(src)}).json()
    assert r["rows"] == 2 and r["report"]["invalid"] == 3
    # con valor por defecto entran todas las legibles
    r = client.post(f"/projects/{p['id']}/datasets/imagecaps",
                    json={"name": "y", "source_dir": str(src), "default_caption": "novastyle"}).json()
    assert r["rows"] == 4
    ds = [d for d in client.get(f"/projects/{p['id']}/datasets?kind=imagegen").json() if d["name"] == "y"][0]
    caps = {_json.loads(l)["text"] for l in (Path(ds["path"]) / "metadata.jsonl").read_text(encoding="utf-8").splitlines()}
    assert caps == {"desde csv", "desde txt", "novastyle"}
    assert any("512" in w for w in r["report"]["warnings"])  # imágenes de 64 px: aviso de ampliación


def test_job_file_is_confined_to_job_folder(client, tmp_path):
    from nova import config

    pid = client.post("/projects", json={"name": "Files"}).json()["id"]
    run = config.PROJECTS_DIR / str(pid) / "runs" / "img"
    (run / "samples").mkdir(parents=True)
    _png(run / "samples" / "ok.png")
    secret = config.PROJECTS_DIR / str(pid) / "secreto.png"
    _png(secret)
    (run / "job_config.json").write_text("{}")
    jid = _job(pid, run)
    assert client.get(f"/jobs/{jid}/file", params={"name": "samples/ok.png"}).status_code == 200
    assert client.get(f"/jobs/{jid}/file", params={"name": "../secreto.png"}).status_code == 404      # fuera de la carpeta
    assert client.get(f"/jobs/{jid}/file", params={"name": "job_config.json"}).status_code == 404      # no es imagen
    assert client.get(f"/jobs/{jid}/file", params={"name": "samples/no_existe.png"}).status_code == 404


def test_image_endpoints_validate_input(client):
    assert client.post("/image/generate", json={"prompt": "x", "width": 500, "height": 512}).status_code == 422  # no múltiplo de 8
    assert client.post("/image/generate", json={"prompt": "x", "n": 9}).status_code == 422
    p = client.post("/projects", json={"name": "Val"}).json()
    assert client.post(f"/projects/{p['id']}/image/train", json={"dataset_id": 1, "resolution": 500}).status_code == 422
    assert client.post(f"/projects/{p['id']}/image/compare", json={"job_ids": [], "prompts": ["a"]}).status_code == 422


def test_parse_line_routes_samples_to_metrics():
    from nova.jobs import parse_line

    assert parse_line('{"type": "sample", "step": 50, "file": "samples/step_00050.png"}')[0] == "metric"


def test_parse_line_skips_huggingface_windows_warnings():
    from nova.jobs import parse_line

    noisy = [
        "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN",
        r"D:\x\huggingface_hub\file_download.py:149: UserWarning: `huggingface_hub` cache-",
        "To support symlinks on Windows, you either need to activate",
        "  warnings.warn(message)",
    ]
    assert all(parse_line(l)[0] == "skip" for l in noisy)
    assert parse_line("RuntimeError: CUDA out of memory")[0] == "log"  # los errores reales siguen visibles


def test_uv_command_resolution_order(monkeypatch, tmp_path):
    import sys

    from nova import envs

    fake = tmp_path / "uv.exe"
    fake.write_text("")
    monkeypatch.setenv("NOVA_UV", str(fake))
    assert envs.uv_command() == [str(fake)]                       # la variable de entorno manda
    monkeypatch.delenv("NOVA_UV")
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    assert envs.uv_command() == [str(fake)]                       # uv.exe junto al ejecutable (app empaquetada)


def test_uv_command_errors_clearly_when_missing(monkeypatch, tmp_path):
    import shutil
    import sys

    from nova import envs

    monkeypatch.delenv("NOVA_UV", raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setattr(sys, "frozen", True, raising=False)       # empaquetado: no hay `python -m uv` al que recurrir
    with pytest.raises(RuntimeError, match="No se encontró uv"):
        envs.uv_command()
