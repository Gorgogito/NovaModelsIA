"""Fine-tuning de Whisper con clips + transcripciones (bucle propio, sin Trainer).

Uso: python train_worker.py job_config.json
Mejor modelo (menor WER de validación) en <job>/model. Reanuda desde last.pt si existe.
"""
import json
import math
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit, error_counts, load_audio, load_model, read_metadata, split, transcribe  # noqa: E402


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    out = os.path.dirname(os.path.abspath(cfg_path))

    import torch
    from torch.utils.data import DataLoader
    from transformers import get_scheduler

    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    emit(type="log", message=f"Dispositivo: {torch.cuda.get_device_name(0) if cuda else 'CPU'}")

    model, processor = load_model(cfg["base_model"], dev)
    tok = processor.tokenizer
    tok.set_prefix_tokens(language=cfg["language"], task="transcribe")
    start_id = model.config.decoder_start_token_id
    model.config.use_cache = False
    if cfg["freeze_encoder"]:
        for p in model.model.encoder.parameters():
            p.requires_grad = False
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    rows = read_metadata(cfg["dataset_dir"])
    train_rows, val_rows = split(rows, cfg["val_split"])
    val_rows = val_rows[: cfg["max_val_clips"]]
    emit(type="log", message=f"Clips de entrenamiento: {len(train_rows)} · validación: {len(val_rows)}")

    def encode(row):
        audio = load_audio(os.path.join(cfg["dataset_dir"], row["file"]))
        feats = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features[0]
        return feats, tok(row["text"]).input_ids

    class DS:
        def __init__(self, r):
            self.r = r

        def __len__(self):
            return len(self.r)

        def __getitem__(self, i):
            return encode(self.r[i])

    def collate(batch):
        feats = torch.stack([b[0] for b in batch])
        longest = max(len(b[1]) for b in batch)
        labels = torch.full((len(batch), longest), -100, dtype=torch.long)
        for i, (_, ids) in enumerate(batch):
            labels[i, : len(ids)] = torch.tensor(ids)
        if (labels[:, 0] == start_id).all():  # el modelo vuelve a añadir el token de inicio al desplazar
            labels = labels[:, 1:]
        return feats, labels

    nw = cfg.get("num_workers", 0)  # 0: en Windows los workers no pueden serializar las clases locales
    dl = DataLoader(DS(train_rows), batch_size=cfg["batch_size"], shuffle=True, collate_fn=collate, num_workers=nw,
                    persistent_workers=nw > 0)
    dl_val = DataLoader(DS(val_rows), batch_size=cfg["batch_size"], collate_fn=collate, num_workers=0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(dl) / cfg["grad_accum"])
    total = steps_per_epoch * cfg["epochs"]
    sched = get_scheduler("linear", opt, num_warmup_steps=max(1, int(0.1 * total)), num_training_steps=total)
    amp = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(enabled=cuda and amp == torch.float16)

    start, best_wer, best_loss = 0, 9.9, 1e9
    last = os.path.join(out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best_wer, best_loss = ck["epoch"] + 1, ck["best_wer"], ck["best_loss"]
        emit(type="log", message=f"Reanudando desde la época {start + 1}")

    for ep in range(start, cfg["epochs"]):
        model.train()
        tl, tn = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for k, (x, y) in enumerate(dl):
            x, y = x.to(dev), y.to(dev)
            with torch.autocast(device_type=dev.type, dtype=amp, enabled=cuda):
                loss = model(input_features=x, labels=y).loss
            scaler.scale(loss / cfg["grad_accum"]).backward()
            tl += loss.item() * len(x)
            tn += len(x)
            if (k + 1) % cfg["grad_accum"] == 0 or k + 1 == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)

        model.eval()
        vl, vn = 0.0, 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(device_type=dev.type, dtype=amp, enabled=cuda):
                    vl += model(input_features=x, labels=y).loss.item() * len(x)
                vn += len(x)
        refs, hyps = [], []
        for r in val_rows:
            audio = load_audio(os.path.join(cfg["dataset_dir"], r["file"]))
            model.config.use_cache = True
            with torch.autocast(device_type=dev.type, dtype=amp, enabled=cuda):
                hyps.append(transcribe(model, processor, audio, dev, cfg["language"]))
            model.config.use_cache = False
            refs.append(r["text"])
        we, wn, ce, cn = error_counts(refs, hyps)
        wer, cer = we / max(wn, 1), ce / max(cn, 1)
        vloss = vl / max(vn, 1)
        emit(type="metric", step=ep + 1, max_steps=cfg["epochs"], loss=tl / max(tn, 1), eval_loss=vloss, wer=wer,
             cer=cer, learning_rate=sched.get_last_lr()[0],
             vram_mb=torch.cuda.max_memory_allocated() // 2**20 if cuda else 0)

        if wer < best_wer or (wer == best_wer and vloss < best_loss):
            best_wer, best_loss = wer, vloss
            model.save_pretrained(os.path.join(out, "model"))
            processor.save_pretrained(os.path.join(out, "model"))
            json.dump({"base_model": cfg["base_model"], "language": cfg["language"], "wer": wer, "cer": cer,
                       "epoch": ep + 1}, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"))
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best_wer": best_wer, "best_loss": best_loss}, last)

    if not os.path.exists(os.path.join(out, "model")):
        raise RuntimeError("No se guardó ningún modelo (¿0 épocas?).")
    if os.path.exists(last):  # solo sirve para reanudar; ocupa varios GB
        os.remove(last)
    emit(type="log", message=f"Mejor WER de validación: {best_wer:.3f}")
    emit(type="done")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit(type="error", message=f"{type(e).__name__}: {e}")
        sys.exit(1)
