"""Proceso de inferencia (corre en el venv del motor LLM). Protocolo: JSON por línea.

Comandos:
  load         -> {"ok": true}
  chat         -> {"ok": true, "reply": "..."}                       (respuesta completa)
  chat_stream  -> varias líneas {"delta": "..."} y al final
                  {"done": true, "ok": true, "prompt_tokens": n, "completion_tokens": m}
"""
import json
import sys
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

model = tok = None


def reply(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def load(req):
    global model, tok
    kwargs = {"device_map": "auto" if torch.cuda.is_available() else None}
    if req.get("quant_4bit") and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    else:
        kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    tok = AutoTokenizer.from_pretrained(req["base_model"])
    model = AutoModelForCausalLM.from_pretrained(req["base_model"], **kwargs)
    if req.get("adapter_path"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, req["adapter_path"])
    model.eval()


def prepare(req):
    """Devuelve (inputs, parámetros de generación)."""
    text = tok.apply_chat_template(req["messages"], tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    temp = float(req.get("temperature", 0.7))
    gen = dict(max_new_tokens=int(req.get("max_new_tokens", 256)), pad_token_id=tok.eos_token_id)
    if temp > 0:
        gen.update(do_sample=True, temperature=temp, top_p=float(req.get("top_p", 0.9)))
    else:
        gen.update(do_sample=False)
    return inputs, gen


def chat(req):
    inputs, gen = prepare(req)
    with torch.no_grad():
        out = model.generate(**inputs, **gen)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def chat_stream(req):
    inputs, gen = prepare(req)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    errors = []

    def run():
        try:
            with torch.no_grad():
                model.generate(**inputs, streamer=streamer, **gen)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{type(e).__name__}: {e}")
            streamer.end()  # desbloquea el iterador si la generación falló

    t = threading.Thread(target=run, daemon=True)
    t.start()
    parts = []
    for piece in streamer:
        if piece:
            parts.append(piece)
            reply({"delta": piece})
    t.join()
    if errors:
        reply({"done": True, "ok": False, "error": errors[0]})
        return
    completion = len(tok("".join(parts), add_special_tokens=False).input_ids)  # aproximado: se vuelve a tokenizar el texto
    reply({"done": True, "ok": True, "prompt_tokens": int(inputs["input_ids"].shape[1]), "completion_tokens": completion})


for line in sys.stdin:
    try:
        req = json.loads(line)
        if req["cmd"] == "load":
            load(req)
            reply({"ok": True})
        elif req["cmd"] == "chat":
            reply({"ok": True, "reply": chat(req)})
        elif req["cmd"] == "chat_stream":
            chat_stream(req)
    except Exception as e:  # noqa: BLE001
        reply({"done": True, "ok": False, "error": f"{type(e).__name__}: {e}"})
