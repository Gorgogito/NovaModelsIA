"""Proceso persistente de generación de imágenes. Protocolo: una línea JSON por petición."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_pipeline, pick_dtype, png_b64  # noqa: E402

import torch  # noqa: E402

pipe = None
has_lora = False
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reply(o):
    sys.stdout.write(json.dumps(o, ensure_ascii=False) + "\n")
    sys.stdout.flush()


for line in sys.stdin:
    try:
        req = json.loads(line)
        if req["cmd"] == "load":
            pipe, has_lora = None, False
            torch.cuda.empty_cache()
            pipe = load_pipeline(req["base_model"], pick_dtype(), dev)
            if req.get("lora_path"):
                pipe.load_lora_weights(req["lora_path"], adapter_name="nova")
                pipe.set_adapters(["nova"], adapter_weights=[float(req.get("lora_scale", 1.0))])
                has_lora = True
            reply({"ok": True, "lora": has_lora})
        elif req["cmd"] == "generate":
            w, h = int(req["width"]), int(req["height"])
            if w % 8 or h % 8:
                raise ValueError("El ancho y el alto deben ser múltiplos de 8.")
            images, t0 = [], time.time()
            for i in range(int(req["n"])):
                seed = int(req["seed"]) + i
                g = torch.Generator(device=dev).manual_seed(seed)
                img = pipe(req["prompt"], negative_prompt=req.get("negative_prompt") or None,
                           num_inference_steps=int(req["steps"]), guidance_scale=float(req["guidance"]),
                           width=w, height=h, generator=g).images[0]
                images.append({"seed": seed, "png_b64": png_b64(img)})
            reply({"ok": True, "images": images, "seconds": round(time.time() - t0, 2),
                   "vram_mb": torch.cuda.max_memory_allocated() // 2**20 if dev.type == "cuda" else 0})
    except Exception as e:  # noqa: BLE001
        reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
