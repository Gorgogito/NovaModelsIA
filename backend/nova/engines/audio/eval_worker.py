"""Evalúa un modelo Whisper (hub o carpeta local) sobre un dataset de audio: WER, CER y velocidad."""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit, error_counts, load_audio, load_model, read_metadata, transcribe  # noqa: E402


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))

    import torch

    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    model, processor = load_model(cfg["model_path"], dev, torch.float16 if cuda else None)
    rows = read_metadata(cfg["dataset_dir"])[: cfg["max_clips"]]
    emit(type="log", message=f"Evaluando {len(rows)} clips con {cfg['model_path']}")

    refs, hyps, samples, audio_s = [], [], [], 0.0
    t0 = time.time()
    for i, r in enumerate(rows):
        audio = load_audio(os.path.join(cfg["dataset_dir"], r["file"]))
        audio_s += len(audio) / 16000
        hyp = transcribe(model, processor, audio, dev, cfg["language"])
        refs.append(r["text"])
        hyps.append(hyp)
        we, wn, _, _ = error_counts([r["text"]], [hyp])
        if len(samples) < 20:
            samples.append({"prompt": r["file"], "reference": r["text"], "prediction": hyp, "wer": round(we / max(wn, 1), 3)})
        emit(type="log", message=f"{i + 1}/{len(rows)}")
    elapsed = time.time() - t0
    we, wn, ce, cn = error_counts(refs, hyps)
    emit(type="result", result={
        "rows": len(rows), "wer": round(we / max(wn, 1), 4), "cer": round(ce / max(cn, 1), 4),
        "words": wn, "audio_seconds": round(audio_s, 1),
        "rtf": round(elapsed / max(audio_s, 1e-9), 3),  # <1 = más rápido que tiempo real
        "samples": samples,
    })
    emit(type="done")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit(type="error", message=f"{type(e).__name__}: {e}")
        sys.exit(1)
