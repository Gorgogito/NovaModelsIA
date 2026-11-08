"""API local compatible con OpenAI (solo en 127.0.0.1).

Implementado: GET /v1/models, POST /v1/chat/completions (con stream), POST /v1/audio/transcriptions,
POST /v1/images/generations (solo b64_json). Todo lo demás de la API de OpenAI no existe aquí.
Autenticación opcional: si la variable NOVA_API_KEY está definida, se exige `Authorization: Bearer <clave>`.
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import random
import re
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Header, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from nova import audio_infer, chat, image_infer, serving
from nova.serving import OpenAIError


def require_key(authorization: str | None = Header(None)) -> None:
    key = os.environ.get("NOVA_API_KEY")
    if not key:
        return
    given = (authorization or "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(given.encode(), key.encode()):
        raise OpenAIError(401, "Clave de API incorrecta.", code="invalid_api_key")


router = APIRouter(prefix="/v1", dependencies=[Depends(require_key)])

MAX_TOKENS_CAP = 4096
MAX_AUDIO_BYTES = 25 * 1024 * 1024  # mismo límite que la API de OpenAI
LANGS = {"es": "spanish", "en": "english", "fr": "french", "de": "german", "pt": "portuguese", "it": "italian"}


# ---------- Modelos ----------
@router.get("/models")
def list_models():
    return {"object": "list", "data": [e.public() for e in serving.list_entries()]}


@router.get("/models/{model_id:path}")
def get_model(model_id: str):
    for e in serving.list_entries():
        if e.id == model_id:
            return e.public()
    raise OpenAIError(404, f"El modelo `{model_id}` no existe.", code="model_not_found", param="model")


# ---------- Chat ----------
class ChatMessage(BaseModel):
    role: str
    content: str | list[Any] | None = None


class ChatReq(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(1.0, ge=0, le=2)
    top_p: float | None = Field(1.0, gt=0, le=1)
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    n: int | None = 1
    stream: bool | None = False
    stream_options: dict | None = None


def _text(m: ChatMessage) -> str:
    if m.content is None:
        return ""
    if isinstance(m.content, str):
        return m.content
    parts = []
    for p in m.content:
        if not (isinstance(p, dict) and p.get("type") == "text"):
            raise OpenAIError(400, "Solo se admite contenido de texto en los mensajes.", param="messages")
        parts.append(str(p.get("text", "")))
    return "".join(parts)


def _collect(gen):
    """Consume un generador de fragmentos y devuelve (texto, uso)."""
    parts: list[str] = []
    while True:
        try:
            parts.append(next(gen))
        except StopIteration as stop:
            return "".join(parts), stop.value or {}


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.post("/chat/completions")
def chat_completions(req: ChatReq):
    if (req.n or 1) != 1:
        raise OpenAIError(400, "Solo se admite n=1.", param="n")
    msgs = []
    for m in req.messages:
        if m.role not in {"system", "user", "assistant"}:
            raise OpenAIError(400, f"El rol '{m.role}' no está soportado (solo system, user, assistant).", param="messages")
        msgs.append({"role": m.role, "content": _text(m)})
    entry = serving.find(req.model, "llm")
    max_new = min(req.max_tokens or req.max_completion_tokens or 256, MAX_TOKENS_CAP)
    temp = 1.0 if req.temperature is None else req.temperature
    top_p = 1.0 if req.top_p is None else req.top_p
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

    def finish(usage: dict) -> str:
        return "length" if usage.get("completion_tokens", 0) >= max_new else "stop"

    if not req.stream:
        try:
            with serving.use(entry):
                text, usage = _collect(chat.generate_stream(msgs, max_new, temp, top_p))
        except RuntimeError as e:
            raise OpenAIError(500, str(e), "server_error", "generation_failed") from e
        pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        return {"id": cid, "object": "chat.completion", "created": created, "model": entry.id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish(usage)}],
                "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}}

    serving.preload(entry)  # los errores de carga salen como HTTP normal, antes de empezar a transmitir

    def chunk(delta: dict, reason: str | None = None) -> dict:
        return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": entry.id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}

    def events():
        try:
            with serving.use(entry):
                yield _sse(chunk({"role": "assistant", "content": ""}))
                gen = chat.generate_stream(msgs, max_new, temp, top_p)
                usage: dict = {}
                try:
                    while True:
                        yield _sse(chunk({"content": next(gen)}))
                except StopIteration as stop:
                    usage = stop.value or {}
                yield _sse(chunk({}, finish(usage)))
                if (req.stream_options or {}).get("include_usage"):
                    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
                    yield _sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": entry.id,
                                "choices": [], "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}})
        except (RuntimeError, OpenAIError) as e:
            err = e if isinstance(e, OpenAIError) else OpenAIError(500, str(e), "server_error", "generation_failed")
            yield _sse(err.body())
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------- Transcripción de audio ----------
@router.post("/audio/transcriptions")
def transcriptions(file: UploadFile = File(...), model: str = Form(...), language: str | None = Form(None),
                   response_format: str = Form("json"), prompt: str | None = Form(None), temperature: float | None = Form(None)):
    if response_format not in {"json", "text"}:
        raise OpenAIError(400, "Solo se admiten los formatos de respuesta 'json' y 'text'.", param="response_format")
    lang = None
    if language:
        lang = LANGS.get(language.lower(), language.lower() if language.lower() in LANGS.values() else None)
        if lang is None:
            raise OpenAIError(400, f"Idioma no soportado: '{language}'. Admitidos: {', '.join(LANGS)}.", param="language")
    entry = serving.find(model, "audio")
    data = file.file.read(MAX_AUDIO_BYTES + 1)
    if len(data) > MAX_AUDIO_BYTES:
        raise OpenAIError(413, "El archivo supera el límite de 25 MB.", param="file")
    if not data:
        raise OpenAIError(400, "El archivo de audio está vacío.", param="file")
    try:
        with serving.use(entry):
            r = audio_infer.transcribe(base64.b64encode(data).decode(), lang)
    except RuntimeError as e:
        msg = str(e)
        if "LibsndfileError" in msg or "Error opening" in msg:
            raise OpenAIError(400, "No se pudo leer el audio. Formatos admitidos: wav, flac, ogg, mp3.", param="file") from e
        raise OpenAIError(500, msg, "server_error", "transcription_failed") from e
    if response_format == "text":
        return PlainTextResponse(r["text"])
    return {"text": r["text"]}


# ---------- Imágenes ----------
class ImageReq(BaseModel):
    model: str
    prompt: str = Field(min_length=1, max_length=1000)
    n: int | None = Field(1, ge=1, le=4)
    size: str | None = "512x512"
    response_format: str | None = "b64_json"
    # Extensiones fuera del estándar (con el SDK de OpenAI: extra_body={...})
    negative_prompt: str = ""
    num_inference_steps: int = Field(25, ge=5, le=100)
    guidance_scale: float = Field(7.5, ge=1, le=20)
    seed: int | None = None


@router.post("/images/generations")
def image_generations(req: ImageReq):
    if (req.response_format or "b64_json") != "b64_json":
        raise OpenAIError(400, "Solo se admite response_format='b64_json' (no se sirven URLs).", param="response_format")
    m = re.fullmatch(r"(\d{3})x(\d{3})", req.size or "512x512")
    if not m or any(int(v) % 8 or not 128 <= int(v) <= 768 for v in m.groups()):
        raise OpenAIError(400, "size debe ser 'ANCHOxALTO' con valores múltiplos de 8 entre 128 y 768 (p. ej. 512x512).", param="size")
    entry = serving.find(req.model, "image")
    seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
    try:
        with serving.use(entry):
            r = image_infer.generate({"prompt": req.prompt, "negative_prompt": req.negative_prompt, "steps": req.num_inference_steps,
                                      "guidance": req.guidance_scale, "seed": seed, "width": int(m.group(1)),
                                      "height": int(m.group(2)), "n": req.n or 1})
    except RuntimeError as e:
        raise OpenAIError(500, str(e), "server_error", "generation_failed") from e
    return {"created": int(time.time()), "data": [{"b64_json": im["png_b64"], "seed": im["seed"]} for im in r["images"]]}


def openai_error_response(e: OpenAIError) -> JSONResponse:
    return JSONResponse(status_code=e.status, content=e.body())
