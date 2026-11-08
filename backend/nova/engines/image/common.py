"""Utilidades compartidas por los procesos del motor de imagen (corren en su venv)."""
import base64
import io
import json
import os
import sys

SD15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def emit(**ev):
    sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def read_metadata(root):
    with open(os.path.join(root, "metadata.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_dtype():
    import torch

    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_pipeline(base_model, dtype, device):
    """Carga el pipeline de Stable Diffusion en fp16/bf16 con un planificador rápido (DPM++).

    Sin comprobador de seguridad: no se descarga (ahorra 580 MB) y no filtra el contenido generado.
    """
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        base_model, torch_dtype=dtype, variant="fp16" if dtype != __import__("torch").float32 else None,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def png_b64(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
