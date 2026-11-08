"""Proceso persistente de predicción. Protocolo: una línea JSON por petición."""
import base64
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_model, make_transforms  # noqa: E402

import torch  # noqa: E402

model = meta = tf = None
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reply(o):
    sys.stdout.write(json.dumps(o, ensure_ascii=False) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    try:
        req = json.loads(line)
        if req["cmd"] == "load":
            model, meta = load_model(req["model_dir"], dev)
            _, tf = make_transforms(meta["img_size"], meta["mean"], meta["std"], False)
            reply({"ok": True, "classes": meta["classes"], "val_acc": meta.get("val_acc")})
        elif req["cmd"] == "predict":
            from PIL import Image

            im = Image.open(io.BytesIO(base64.b64decode(req["image_b64"]))).convert("RGB")
            with torch.no_grad():
                probs = model(tf(im).unsqueeze(0).to(dev)).softmax(1)[0].cpu()
            top = probs.topk(min(5, len(meta["classes"])))
            reply({"ok": True, "top": [{"class": meta["classes"][i], "prob": round(p, 4)}
                                       for p, i in zip(top.values.tolist(), top.indices.tolist())]})
    except Exception as e:  # noqa: BLE001
        reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
