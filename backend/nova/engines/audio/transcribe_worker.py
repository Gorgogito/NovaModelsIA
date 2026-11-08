"""Proceso persistente de transcripción. Protocolo: una línea JSON por petición."""
import base64
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_audio, load_model, transcribe  # noqa: E402

import torch  # noqa: E402

model = processor = None
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reply(o):
    sys.stdout.write(json.dumps(o, ensure_ascii=False) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    try:
        req = json.loads(line)
        if req["cmd"] == "load":
            model, processor = load_model(req["model_path"], dev, torch.float16 if dev.type == "cuda" else None)
            reply({"ok": True})
        elif req["cmd"] == "transcribe":
            audio = load_audio(io.BytesIO(base64.b64decode(req["audio_b64"])))
            t0 = time.time()
            text = transcribe(model, processor, audio, dev, req.get("language"))
            reply({"ok": True, "text": text, "audio_seconds": round(len(audio) / 16000, 2),
                   "seconds": round(time.time() - t0, 2)})
    except Exception as e:  # noqa: BLE001
        reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
