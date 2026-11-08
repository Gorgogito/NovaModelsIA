"""Utilidades compartidas por los procesos del motor de audio (corren en su venv)."""
import json
import os
import random
import re
import sys
import unicodedata

SR = 16000  # Whisper trabaja a 16 kHz mono
CHUNK_S = 28.0  # ventanas de transcripción (el límite de Whisper es 30 s)


def emit(**ev):
    sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def load_audio(path_or_file):
    """Lee cualquier audio soportado por libsndfile y lo devuelve como float32 mono a 16 kHz."""
    import numpy as np
    import soundfile as sf
    import soxr

    data, sr = sf.read(path_or_file, dtype="float32", always_2d=True)
    data = data.mean(axis=1)
    if sr != SR:
        data = soxr.resample(data, sr, SR)
    return np.ascontiguousarray(data, dtype="float32")


def read_metadata(root):
    with open(os.path.join(root, "metadata.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split(rows, val_split, seed=42):
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    n_val = max(1, int(len(rows) * val_split)) if len(rows) > 1 else 0
    return rows[n_val:], rows[:n_val]


def normalize(text):
    """Minúsculas, sin acentos de puntuación ni signos: así el WER no penaliza mayúsculas ni comas."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def error_counts(refs, hyps):
    """Devuelve (errores_palabra, palabras_ref, errores_caracter, caracteres_ref) sobre texto normalizado."""
    we = wn = ce = cn = 0
    for r, h in zip(refs, hyps):
        r, h = normalize(r), normalize(h)
        rw, hw = r.split(), h.split()
        we += _edit_distance(rw, hw)
        wn += len(rw)
        ce += _edit_distance(r, h)
        cn += len(r)
    return we, wn, ce, cn


def transcribe(model, processor, audio, device, language, max_new_tokens=225):
    """Transcribe un array de audio (puede durar más de 30 s: se corta en ventanas de CHUNK_S)."""
    import torch

    step = int(CHUNK_S * SR)
    parts = []
    for i in range(0, max(len(audio), 1), step):
        chunk = audio[i : i + step]
        if len(chunk) < SR * 0.2:
            continue
        feats = processor.feature_extractor(chunk, sampling_rate=SR, return_tensors="pt").input_features.to(device)
        feats = feats.to(model.dtype)
        kw = {"task": "transcribe"}
        if language and language != "auto":
            kw["language"] = language
        with torch.no_grad():
            ids = model.generate(input_features=feats, max_new_tokens=max_new_tokens, **kw)
        parts.append(processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip())
    return " ".join(p for p in parts if p)


def load_model(model_path, device, dtype=None):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path, **({"dtype": dtype} if dtype else {}))
    for obj in (model.config, model.generation_config):  # evita conflicto con language/task explícitos
        if getattr(obj, "forced_decoder_ids", None):
            obj.forced_decoder_ids = None
    return model.to(device).eval(), processor
