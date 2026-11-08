"""Pruebas de la API compatible con OpenAI, con los motores simulados (sin GPU)."""
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nova import audio_infer, chat, config, db, envs, image_infer, serving
from nova.api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class Engines:
    """Registra las llamadas a los motores y simula qué hay cargado."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.loaded = {"llm": False, "audio": False, "image": False}


@pytest.fixture
def eng(monkeypatch):
    e = Engines()
    db.execute("UPDATE jobs SET status='stopped' WHERE status IN ('running','queued')")  # otras pruebas pueden dejarlos
    serving.invalidate()
    monkeypatch.setattr(envs, "is_ready", lambda engine: True)
    monkeypatch.setattr(chat, "load", lambda base, adapter, q: (e.calls.append(("load", "llm", base, adapter)), e.loaded.update(llm=True)))
    monkeypatch.setattr(audio_infer, "load", lambda path: (e.calls.append(("load", "audio", path)), e.loaded.update(audio=True)))
    monkeypatch.setattr(image_infer, "load", lambda base, lora, s: (e.calls.append(("load", "image", base, lora)), e.loaded.update(image=True)))
    monkeypatch.setattr(chat, "unload", lambda: (e.calls.append(("unload", "llm")), e.loaded.update(llm=False)))
    monkeypatch.setattr(audio_infer, "unload", lambda: (e.calls.append(("unload", "audio")), e.loaded.update(audio=False)))
    monkeypatch.setattr(image_infer, "unload", lambda: (e.calls.append(("unload", "image")), e.loaded.update(image=False)))
    monkeypatch.setattr(chat, "is_loaded", lambda: e.loaded["llm"])
    monkeypatch.setattr(audio_infer, "is_loaded", lambda: e.loaded["audio"])
    monkeypatch.setattr(image_infer, "is_loaded", lambda: e.loaded["image"])

    def fake_stream(messages, max_new, temp, top_p=0.9):
        e.calls.append(("chat", messages, max_new, temp, top_p))
        yield "Hola"
        yield " mundo"
        return {"prompt_tokens": 7, "completion_tokens": 2}

    monkeypatch.setattr(chat, "generate_stream", fake_stream)
    monkeypatch.setattr(audio_infer, "transcribe", lambda b64, lang: (e.calls.append(("transcribe", len(b64), lang)), {"text": "texto transcrito"})[1])
    monkeypatch.setattr(image_infer, "generate", lambda p: (e.calls.append(("image", p)), {"images": [{"seed": p["seed"] + i, "png_b64": "UE5H"} for i in range(p["n"])]})[1])
    return e


def _snapshot(name: str, files: dict):
    snap = config.MODELS_DIR / "hf" / "hub" / f"models--{name}" / "snapshots" / "abc123"
    snap.mkdir(parents=True, exist_ok=True)
    for fn, content in files.items():
        (snap / fn).write_text(json.dumps(content), encoding="utf-8")


def _trained(engine: str, sub: str) -> int:
    d = config.PROJECTS_DIR / "serv" / engine
    (d / sub).mkdir(parents=True, exist_ok=True)
    cfg = {"base_model": {"llm": "Qwen/Qwen2.5-0.5B-Instruct", "audio": "openai/whisper-tiny",
                          "image": "stable-diffusion-v1-5/stable-diffusion-v1-5"}[engine]}
    return db.execute("INSERT INTO jobs(project_id,engine,kind,config,status,output_dir,created_at) VALUES(?,?,?,?,?,?,?)",
                      (1, engine, "train", json.dumps(cfg), "completed", str(d), db.now()))


@pytest.fixture(scope="module", autouse=True)
def catalog(client):  # depende del cliente: su arranque crea la base de datos
    _snapshot("Qwen--Qwen2.5-0.5B-Instruct", {"config.json": {"architectures": ["Qwen2ForCausalLM"]}})
    _snapshot("openai--whisper-tiny", {"config.json": {"architectures": ["WhisperForConditionalGeneration"]}})
    _snapshot("stable-diffusion-v1-5--stable-diffusion-v1-5", {"model_index.json": {"_class_name": "StableDiffusionPipeline"}})
    _snapshot("stabilityai--sdxl", {"model_index.json": {"_class_name": "StableDiffusionXLPipeline"}})   # no se sirve
    _snapshot("timm--resnet18.a1_in1k", {"config.json": {"architecture": "resnet18"}})                    # no es de este tipo
    return {"llm": _trained("llm", "adapter"), "audio": _trained("audio", "model"), "image": _trained("image", "lora")}


LLM, WHISPER, SD = "Qwen/Qwen2.5-0.5B-Instruct", "openai/whisper-tiny", "stable-diffusion-v1-5/stable-diffusion-v1-5"
CHAT = {"messages": [{"role": "user", "content": "hola"}]}


# ---------- Modelos ----------
def test_models_lists_trained_and_downloaded_only(client, catalog):
    data = {m["id"]: m for m in client.get("/v1/models").json()["data"]}
    assert {LLM, WHISPER, SD, f"nova-llm-{catalog['llm']}", f"nova-whisper-{catalog['audio']}",
            f"nova-sd-lora-{catalog['image']}"} <= set(data)
    assert "stabilityai/sdxl" not in data and "timm/resnet18.a1_in1k" not in data
    assert data[LLM]["nova"] == {"modality": "llm", "kind": "base", "base_model": LLM}
    assert data[f"nova-llm-{catalog['llm']}"]["nova"]["kind"] == "trained"
    assert client.get(f"/v1/models/{SD}").json()["id"] == SD          # el id contiene '/'
    assert client.get("/v1/models/no-existe").json()["error"]["code"] == "model_not_found"


# ---------- Chat ----------
def test_chat_completion_shape_and_model_reuse(client, eng, catalog):
    mid = f"nova-llm-{catalog['llm']}"
    r = client.post("/v1/chat/completions", json={"model": mid, **CHAT, "max_tokens": 50, "temperature": 0.2}).json()
    assert r["object"] == "chat.completion" and r["id"].startswith("chatcmpl-") and r["model"] == mid
    assert r["choices"][0]["message"] == {"role": "assistant", "content": "Hola mundo"}
    assert r["choices"][0]["finish_reason"] == "stop"
    assert r["usage"] == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
    loads = [c for c in eng.calls if c[0] == "load"]
    assert loads == [("load", "llm", LLM, str(config.PROJECTS_DIR / "serv" / "llm" / "adapter"))]  # adaptador + base
    assert [c for c in eng.calls if c[0] == "chat"][0][1:4] == ([{"role": "user", "content": "hola"}], 50, 0.2)
    client.post("/v1/chat/completions", json={"model": mid, **CHAT})
    assert len([c for c in eng.calls if c[0] == "load"]) == 1                 # la segunda petición no recarga
    client.post("/v1/chat/completions", json={"model": LLM, **CHAT})
    assert len([c for c in eng.calls if c[0] == "load"]) == 2                 # otro modelo: se recarga


def test_chat_length_finish_reason(client, eng):
    r = client.post("/v1/chat/completions", json={"model": LLM, **CHAT, "max_tokens": 2}).json()
    assert r["choices"][0]["finish_reason"] == "length"                       # completion_tokens (2) >= max_tokens (2)


def test_chat_accepts_content_parts(client, eng):
    body = {"model": LLM, "messages": [{"role": "user", "content": [{"type": "text", "text": "ho"}, {"type": "text", "text": "la"}]}]}
    assert client.post("/v1/chat/completions", json=body).status_code == 200
    assert [c for c in eng.calls if c[0] == "chat"][0][1] == [{"role": "user", "content": "hola"}]
    body["messages"][0]["content"] = [{"type": "image_url", "image_url": {"url": "x"}}]
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == 400 and r.json()["error"]["param"] == "messages"


def test_chat_streaming_sse(client, eng):
    with client.stream("POST", "/v1/chat/completions",
                       json={"model": LLM, **CHAT, "stream": True, "stream_options": {"include_usage": True}}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        lines = [l for l in r.iter_lines() if l]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(l[6:]) for l in lines[:-1]]
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]) == "Hola mundo"
    assert chunks[-2]["choices"][0]["finish_reason"] == "stop"
    assert chunks[-1]["usage"]["total_tokens"] == 9


def test_stream_load_error_is_a_normal_http_error(client, eng, monkeypatch):
    monkeypatch.setattr(chat, "load", lambda *a: (_ for _ in ()).throw(RuntimeError("sin VRAM")))
    r = client.post("/v1/chat/completions", json={"model": LLM, **CHAT, "stream": True})
    assert r.status_code == 500 and r.json()["error"]["code"] == "model_load_failed"


@pytest.mark.parametrize("body,status,code", [
    ({"model": "no-existe", **CHAT}, 404, "model_not_found"),
    ({"model": WHISPER, **CHAT}, 400, "model_not_supported"),                  # un Whisper no sirve para chat
    ({"model": LLM, **CHAT, "n": 2}, 400, None),
    ({"model": LLM, "messages": [{"role": "tool", "content": "x"}]}, 400, None),
    ({"model": LLM}, 400, "invalid_request"),                                    # falta messages
    ({"model": LLM, **CHAT, "temperature": 5}, 400, "invalid_request"),
])
def test_chat_errors_use_openai_format(client, eng, body, status, code):
    r = client.post("/v1/chat/completions", json=body)
    assert r.status_code == status
    err = r.json()["error"]
    assert set(err) == {"message", "type", "param", "code"} and err["message"]
    if code:
        assert err["code"] == code


def test_gpu_busy_and_engine_missing(client, eng, monkeypatch):
    jid = db.execute("INSERT INTO jobs(project_id,engine,kind,config,status,output_dir,created_at) VALUES(1,'llm','train','{}','running','x',0)")
    r = client.post("/v1/chat/completions", json={"model": LLM, **CHAT})
    assert r.status_code == 503 and r.json()["error"]["code"] == "gpu_busy"
    db.execute("UPDATE jobs SET status='stopped' WHERE id=?", (jid,))
    monkeypatch.setattr(envs, "is_ready", lambda engine: False)
    r = client.post("/v1/chat/completions", json={"model": LLM, **CHAT})
    assert r.status_code == 503 and r.json()["error"]["code"] == "engine_not_installed"


def test_generation_failure_maps_to_500(client, eng, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")
        yield  # noqa

    monkeypatch.setattr(chat, "generate_stream", boom)
    r = client.post("/v1/chat/completions", json={"model": LLM, **CHAT})
    assert r.status_code == 500 and r.json()["error"]["code"] == "generation_failed"


# ---------- Clave opcional ----------
def test_api_key_is_enforced_only_when_configured(client, eng, monkeypatch):
    assert client.get("/v1/models").status_code == 200                          # sin clave configurada: abierto
    monkeypatch.setenv("NOVA_API_KEY", "secreta")
    assert client.get("/v1/models").status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer mala"}).status_code == 401
    assert client.get("/v1/models", headers={"Authorization": "Bearer secreta"}).status_code == 200
    r = client.post("/v1/chat/completions", json={"model": LLM, **CHAT})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert client.get("/health").status_code == 200                              # la interfaz no usa la clave
    assert client.get("/api-info").json()["key_required"] is True


# ---------- Transcripción ----------
def test_transcriptions(client, eng):
    files = {"file": ("a.wav", b"RIFFxxxx", "audio/wav")}
    r = client.post("/v1/audio/transcriptions", files=files, data={"model": WHISPER, "language": "es"})
    assert r.json() == {"text": "texto transcrito"}
    assert [c for c in eng.calls if c[0] == "transcribe"][0][2] == "spanish"   # código ISO -> nombre de idioma de Whisper
    r = client.post("/v1/audio/transcriptions", files=files, data={"model": WHISPER, "response_format": "text"})
    assert r.text == "texto transcrito" and r.headers["content-type"].startswith("text/plain")
    assert [c for c in eng.calls if c[0] == "transcribe"][1][2] is None          # sin idioma: Whisper lo detecta


@pytest.mark.parametrize("data,status", [
    ({"model": WHISPER, "language": "xx"}, 400),
    ({"model": WHISPER, "response_format": "srt"}, 400),
    ({"model": LLM}, 400),                                                       # un LLM no transcribe
    ({"model": "no-existe"}, 404),
])
def test_transcription_errors(client, eng, data, status):
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")}, data=data)
    assert r.status_code == status and "error" in r.json()


def test_transcription_rejects_empty_and_oversized(client, eng, monkeypatch):
    assert client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"", "audio/wav")}, data={"model": WHISPER}).status_code == 400
    from nova.api import openai_compat
    monkeypatch.setattr(openai_compat, "MAX_AUDIO_BYTES", 10)
    r = client.post("/v1/audio/transcriptions", files={"file": ("a.wav", b"x" * 50, "audio/wav")}, data={"model": WHISPER})
    assert r.status_code == 413


# ---------- Imágenes ----------
def test_image_generation(client, eng, catalog):
    mid = f"nova-sd-lora-{catalog['image']}"
    r = client.post("/v1/images/generations", json={"model": mid, "prompt": "un gato", "n": 2, "size": "512x384", "seed": 10}).json()
    assert [d["b64_json"] for d in r["data"]] == ["UE5H", "UE5H"] and [d["seed"] for d in r["data"]] == [10, 11]
    params = [c for c in eng.calls if c[0] == "image"][0][1]
    assert (params["width"], params["height"], params["n"], params["seed"]) == (512, 384, 2, 10)
    assert ("load", "image", SD, str(config.PROJECTS_DIR / "serv" / "image" / "lora")) in eng.calls   # base + LoRA


@pytest.mark.parametrize("body", [
    {"size": "500x500"}, {"size": "1024x1024"}, {"size": "grande"}, {"response_format": "url"}, {"n": 9}, {"prompt": ""},
])
def test_image_validation(client, eng, body):
    r = client.post("/v1/images/generations", json={"model": SD, "prompt": "x", **body})
    assert r.status_code == 400 and "error" in r.json()


# ---------- Un solo modelo cargado ----------
def test_switching_modality_unloads_the_previous_model(client, eng):
    client.post("/v1/chat/completions", json={"model": LLM, **CHAT})
    assert eng.loaded == {"llm": True, "audio": False, "image": False}
    client.post("/v1/images/generations", json={"model": SD, "prompt": "x"})
    assert eng.loaded == {"llm": False, "audio": False, "image": True}          # el LLM se descargó para dejar VRAM
    assert client.get("/api-info").json()["loaded"] == {"modality": "image", "id": SD}


def test_manual_load_invalidates_api_state(client, eng):
    client.post("/v1/chat/completions", json={"model": LLM, **CHAT})
    assert client.get("/api-info").json()["loaded"]["id"] == LLM
    client.post("/llm/unload")                                                    # el usuario descarga a mano desde la interfaz
    assert client.get("/api-info").json()["loaded"] is None
    client.post("/v1/chat/completions", json={"model": LLM, **CHAT})
    assert len([c for c in eng.calls if c[0] == "load"]) == 2                    # y la API vuelve a cargarlo


def test_api_info_unload(client, eng):
    client.post("/v1/chat/completions", json={"model": LLM, **CHAT})
    assert client.post("/api-info/unload").json() == {"loaded": None}
    assert eng.loaded == {"llm": False, "audio": False, "image": False}


# ---------- Protocolo del proceso de chat ----------
class _FakeStdout:
    def __init__(self, lines):
        self.lines, self.pos = lines, 0

    def readline(self):
        if self.pos >= len(self.lines):
            return ""
        self.pos += 1
        return self.lines[self.pos - 1]

    def __iter__(self):
        while self.pos < len(self.lines):
            self.pos += 1
            yield self.lines[self.pos - 1]


class _FakeProc:
    def __init__(self, lines):
        self.stdin = type("S", (), {"write": lambda s, x: None, "flush": lambda s: None})()
        self.stdout = _FakeStdout(lines)

    def poll(self):
        return None


def _ev(**kw):
    return json.dumps(kw) + "\n"


def test_stream_drains_the_rest_when_consumer_leaves(monkeypatch):
    lines = [_ev(delta="a"), "ruido de una librería\n", _ev(delta="b"), _ev(delta="c"),
             _ev(done=True, ok=True, prompt_tokens=1, completion_tokens=3), _ev(delta="SIGUIENTE-RESPUESTA")]
    proc = _FakeProc(lines)
    monkeypatch.setattr(chat, "_proc", proc)
    gen = chat.generate_stream([{"role": "user", "content": "x"}], 10, 0.5)
    assert next(gen) == "a"
    gen.close()                                                                   # el cliente se desconecta a mitad
    assert proc.stdout.readline() == lines[-1]                                    # lo siguiente es de la próxima respuesta: sin desincronizar


def test_stream_returns_usage_and_raises_on_worker_error(monkeypatch):
    monkeypatch.setattr(chat, "_proc", _FakeProc([_ev(delta="a"), _ev(done=True, ok=True, prompt_tokens=4, completion_tokens=9)]))
    gen = chat.generate_stream([], 10, 0.5)
    assert next(gen) == "a"
    with pytest.raises(StopIteration) as stop:
        next(gen)
    assert stop.value.value == {"prompt_tokens": 4, "completion_tokens": 9}
    monkeypatch.setattr(chat, "_proc", _FakeProc([_ev(done=True, ok=False, error="CUDA out of memory")]))
    with pytest.raises(RuntimeError, match="out of memory"):
        list(chat.generate_stream([], 10, 0.5))
