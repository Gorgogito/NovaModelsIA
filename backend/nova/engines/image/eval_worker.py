"""Comparación visual: mismas indicaciones y semillas con el modelo base y con cada LoRA.

No hay una métrica numérica fiable para imágenes generadas con pocas muestras; la comparación
lado a lado (misma semilla) es la forma honesta de ver qué aporta cada LoRA.
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit, load_pipeline, pick_dtype  # noqa: E402


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    out = os.path.dirname(os.path.abspath(cfg_path))
    os.makedirs(os.path.join(out, "compare"), exist_ok=True)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("La comparación necesita una GPU con CUDA.")
    dev = torch.device("cuda")
    pipe = load_pipeline(cfg["base_model"], pick_dtype(), dev)
    loras = cfg["loras"]  # [{label, path}]
    for i, l in enumerate(loras):
        pipe.load_lora_weights(l["path"], adapter_name=f"l{i}")

    columns = [{"label": "Modelo base", "adapter": None}] + [{"label": l["label"], "adapter": f"l{i}"} for i, l in enumerate(loras)]
    cells, t0, total = [], time.time(), len(cfg["prompts"]) * len(cfg["seeds"]) * len(columns)
    n = 0
    for pi, prompt in enumerate(cfg["prompts"]):
        for seed in cfg["seeds"]:
            row = {"prompt": prompt, "seed": seed, "images": []}
            for ci, col in enumerate(columns):
                if col["adapter"] is None:
                    pipe.disable_lora()
                else:
                    pipe.enable_lora()
                    pipe.set_adapters([col["adapter"]], adapter_weights=[cfg["lora_scale"]])
                g = torch.Generator(device=dev).manual_seed(int(seed))
                img = pipe(prompt, negative_prompt=cfg.get("negative_prompt") or None, num_inference_steps=cfg["steps"],
                           guidance_scale=cfg["guidance"], width=cfg["size"], height=cfg["size"], generator=g).images[0]
                rel = f"compare/p{pi}_s{seed}_c{ci}.png"
                img.save(os.path.join(out, rel))
                row["images"].append(rel)
                n += 1
                emit(type="log", message=f"{n}/{total}")
            cells.append(row)
    emit(type="result", result={"columns": [c["label"] for c in columns], "rows": cells, "size": cfg["size"],
                                "seconds": round(time.time() - t0, 1), "rows_count": len(cells)})
    emit(type="done")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit(type="error", message=f"{type(e).__name__}: {e}")
        sys.exit(1)
