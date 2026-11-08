"""Importación y validación de datasets JSONL para LLM."""
from __future__ import annotations

import json
from pathlib import Path


def normalize(rec: dict) -> list[dict] | None:
    """Convierte un registro a formato chat [{role, content}] o None si es inválido."""
    if isinstance(rec.get("messages"), list):
        msgs = rec["messages"]
        ok = all(isinstance(m, dict) and m.get("role") in {"system", "user", "assistant"}
                 and isinstance(m.get("content"), str) and m["content"].strip() for m in msgs)
        return msgs if ok and msgs and msgs[-1]["role"] == "assistant" else None
    pairs = (("instruction", "output"), ("prompt", "completion"), ("question", "answer"), ("input", "output"))
    for q, a in pairs:
        if isinstance(rec.get(q), str) and isinstance(rec.get(a), str) and rec[q].strip() and rec[a].strip():
            return [{"role": "user", "content": rec[q]}, {"role": "assistant", "content": rec[a]}]
    return None


def validate_file(path: Path) -> dict:
    """Lee el JSONL y devuelve un informe; no modifica el archivo."""
    total = valid = dupes = 0
    bad: list[dict] = []
    seen: set[str] = set()
    lengths: list[int] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            total += 1
            try:
                msgs = normalize(json.loads(line))
            except json.JSONDecodeError:
                msgs = None
            if msgs is None:
                if len(bad) < 20:
                    bad.append({"line": n, "reason": "formato inválido o campos vacíos"})
                continue
            key = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
            if key in seen:
                dupes += 1
            seen.add(key)
            valid += 1
            lengths.append(sum(len(m["content"]) for m in msgs))
    return {
        "total": total, "valid": valid, "invalid": total - valid, "duplicates": dupes,
        "avg_chars": round(sum(lengths) / len(lengths)) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0, "bad_samples": bad,
    }


def clean_file(src: Path, dst: Path) -> int:
    """Escribe una copia normalizada (formato messages) sin inválidos ni duplicados."""
    seen: set[str] = set()
    n = 0
    with src.open(encoding="utf-8") as f, dst.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            try:
                msgs = normalize(json.loads(line))
            except json.JSONDecodeError:
                continue
            if msgs is None:
                continue
            key = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            out.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            n += 1
    return n


# ---------- Imágenes (carpetas por clase) ----------
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_imagefolder(root: Path) -> dict:
    """Valida una carpeta con una subcarpeta por clase. No modifica nada."""
    from PIL import Image

    classes: dict[str, list[Path]] = {}
    invalid = 0
    bad: list[str] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        good = []
        for f in sorted(sub.rglob("*")):
            if not f.is_file() or f.suffix.lower() not in IMG_EXT:
                continue
            try:
                with Image.open(f) as im:
                    im.verify()
                good.append(f)
            except Exception:  # noqa: BLE001
                invalid += 1
                if len(bad) < 20:
                    bad.append(str(f))
        if good:
            classes[sub.name] = good
    counts = {k: len(v) for k, v in classes.items()}
    warnings = []
    if len(classes) < 2:
        warnings.append("Se necesitan al menos 2 clases (subcarpetas con imágenes).")
    low = [k for k, n in counts.items() if n < 10]
    if low:
        warnings.append(f"Clases con menos de 10 imágenes: {', '.join(low)}.")
    if counts and max(counts.values()) > 5 * max(min(counts.values()), 1):
        warnings.append("Clases muy desbalanceadas (más de 5 a 1).")
    return {"classes": counts, "total": sum(counts.values()), "invalid": invalid,
            "bad_samples": bad, "warnings": warnings, "_files": classes}


def import_imagefolder(root: Path, dst: Path) -> dict:
    """Copia las imágenes válidas a dst/<clase>/ y devuelve el informe (sin rutas internas)."""
    import shutil

    rep = scan_imagefolder(root)
    files = rep.pop("_files")
    if len(files) < 2:
        raise ValueError("Se necesitan al menos 2 clases con imágenes válidas.")
    for cls, items in files.items():
        (dst / cls).mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(items):
            shutil.copy2(f, dst / cls / f"{i:05d}{f.suffix.lower()}")
    return rep


# ---------- Audio (clips + transcripciones) ----------
AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3"}
MIN_SEC, MAX_SEC = 0.3, 30.0  # Whisper procesa ventanas de 30 s


def _read_metadata(root: Path) -> list[tuple[str, str]]:
    """Lee metadata.csv (file,text) o metadata.jsonl ({file|file_name, text|transcription})."""
    import csv

    pairs: list[tuple[str, str]] = []
    csv_path, jsonl_path = root / "metadata.csv", root / "metadata.jsonl"
    if csv_path.exists():
        with csv_path.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                fn = r.get("file") or r.get("file_name") or r.get("path") or ""
                pairs.append((fn, r.get("text") or r.get("transcription") or ""))
    elif jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    pairs.append((r.get("file") or r.get("file_name") or "", r.get("text") or r.get("transcription") or ""))
    else:
        raise ValueError("Falta metadata.csv (columnas file,text) o metadata.jsonl en la carpeta.")
    return pairs


def scan_audiofolder(root: Path) -> dict:
    """Valida los clips y sus transcripciones. No modifica nada."""
    import soundfile as sf

    good: list[dict] = []
    reasons: dict[str, int] = {}
    bad: list[str] = []

    def reject(why: str, name: str) -> None:
        reasons[why] = reasons.get(why, 0) + 1
        if len(bad) < 20:
            bad.append(f"{name}: {why}")

    for fn, text in _read_metadata(root):
        p = root / fn
        text = (text or "").strip()
        if not fn or not p.is_file() or p.suffix.lower() not in AUDIO_EXT:
            reject("archivo de audio no encontrado o formato no soportado", fn or "(vacío)")
            continue
        if not text:
            reject("transcripción vacía", fn)
            continue
        try:
            dur = sf.info(str(p)).duration
        except Exception:  # noqa: BLE001
            reject("audio ilegible", fn)
            continue
        if dur < MIN_SEC or dur > MAX_SEC:
            reject(f"duración fuera de {MIN_SEC}-{MAX_SEC:.0f} s", fn)
            continue
        good.append({"src": p, "text": text, "duration": round(dur, 3)})

    total_s = sum(g["duration"] for g in good)
    warnings = []
    if len(good) < 20:
        warnings.append("Menos de 20 clips: el fine-tuning probablemente no generalice.")
    elif total_s < 600:
        warnings.append("Menos de 10 minutos de audio: útil para probar el flujo; para resultados reales conviene más.")
    return {"clips": len(good), "invalid": sum(reasons.values()), "reasons": reasons, "bad_samples": bad,
            "hours": round(total_s / 3600, 3), "minutes": round(total_s / 60, 2),
            "avg_seconds": round(total_s / len(good), 2) if good else 0, "warnings": warnings, "_good": good}


def import_audiofolder(root: Path, dst: Path) -> dict:
    """Copia los clips válidos a dst/audio/ y escribe dst/metadata.jsonl normalizado."""
    import shutil

    rep = scan_audiofolder(root)
    good = rep.pop("_good")
    if not good:
        raise ValueError("No hay clips válidos (revisa metadata y formatos .wav/.flac/.ogg/.mp3).")
    (dst / "audio").mkdir(parents=True, exist_ok=True)
    with (dst / "metadata.jsonl").open("w", encoding="utf-8") as out:
        for i, g in enumerate(good):
            name = f"audio/{i:05d}{g['src'].suffix.lower()}"
            shutil.copy2(g["src"], dst / name)
            out.write(json.dumps({"file": name, "text": g["text"], "duration": g["duration"]}, ensure_ascii=False) + "\n")
    return rep


# ---------- Imágenes con descripción (para generación: LoRA de Stable Diffusion) ----------
def _caption_for(root: Path, img: Path, table: dict[str, str], default: str) -> str:
    """Prioridad: metadata.csv/jsonl > archivo .txt con el mismo nombre > descripción por defecto."""
    if img.name in table:
        return table[img.name].strip()
    txt = img.with_suffix(".txt")
    if txt.is_file():
        return txt.read_text(encoding="utf-8").strip()
    return default.strip()


def scan_imagecaps(root: Path, default_caption: str = "") -> dict:
    """Valida imágenes y descripciones. No modifica nada."""
    from PIL import Image

    table: dict[str, str] = {}
    try:
        table = {Path(f).name: t for f, t in _read_metadata(root)}
    except ValueError:
        pass  # sin metadata: valen los .txt o la descripción por defecto
    good: list[dict] = []
    reasons: dict[str, int] = {}
    bad: list[str] = []
    small = 0

    def reject(why: str, name: str) -> None:
        reasons[why] = reasons.get(why, 0) + 1
        if len(bad) < 20:
            bad.append(f"{name}: {why}")

    for f in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT):
        try:
            with Image.open(f) as im:
                im.verify()
            with Image.open(f) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001
            reject("imagen ilegible", f.name)
            continue
        cap = _caption_for(root, f, table, default_caption)
        if not cap:
            reject("sin descripción (añade metadata.csv, un .txt o una descripción por defecto)", f.name)
            continue
        small += min(w, h) < 512
        good.append({"src": f, "text": cap, "w": w, "h": h})

    warnings = []
    if 0 < len(good) < 8:
        warnings.append("Menos de 8 imágenes: el LoRA tenderá a memorizarlas.")
    if good and small:
        warnings.append(f"{small} imágenes miden menos de 512 px por lado y se ampliarán (pierden nitidez).")
    return {"images": len(good), "invalid": sum(reasons.values()), "reasons": reasons, "bad_samples": bad,
            "min_side": min((min(g["w"], g["h"]) for g in good), default=0), "warnings": warnings,
            "sample_captions": [g["text"] for g in good[:3]], "_good": good}


def import_imagecaps(root: Path, dst: Path, default_caption: str = "") -> dict:
    """Copia las imágenes válidas a dst/images/ y escribe dst/metadata.jsonl normalizado."""
    import shutil

    rep = scan_imagecaps(root, default_caption)
    good = rep.pop("_good")
    if not good:
        raise ValueError("No hay imágenes válidas con descripción.")
    (dst / "images").mkdir(parents=True, exist_ok=True)
    with (dst / "metadata.jsonl").open("w", encoding="utf-8") as out:
        for i, g in enumerate(good):
            name = f"images/{i:05d}{g['src'].suffix.lower()}"
            shutil.copy2(g["src"], dst / name)
            out.write(json.dumps({"file": name, "text": g["text"]}, ensure_ascii=False) + "\n")
    return rep
