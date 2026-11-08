"""Fine-tuning de un clasificador de imágenes (timm) con carpetas por clase.

Uso: python train_worker.py job_config.json
Reanuda automáticamente desde last.pt si existe en la carpeta del job.
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ImageList, emit, list_classes, make_transforms, split  # noqa: E402


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    out = os.path.dirname(os.path.abspath(cfg_path))

    import timm
    import torch
    from torch.utils.data import DataLoader

    cuda = torch.cuda.is_available()
    dev = torch.device("cuda" if cuda else "cpu")
    emit(type="log", message=f"Dispositivo: {torch.cuda.get_device_name(0) if cuda else 'CPU'}")

    classes = list_classes(cfg["dataset_dir"])
    train_items, val_items = split(cfg["dataset_dir"], classes, cfg["val_split"])
    emit(type="log", message=f"Clases: {len(classes)} · entrenamiento: {len(train_items)} · validación: {len(val_items)}")

    model = timm.create_model(cfg["model_name"], pretrained=True, num_classes=len(classes))
    dc = timm.data.resolve_data_config({}, model=model)
    mean, std = list(dc["mean"]), list(dc["std"])
    if cfg["freeze_backbone"]:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
    model.to(dev)

    tf_train, tf_val = make_transforms(cfg["img_size"], mean, std, cfg["augment"])
    nw = cfg.get("num_workers", 2)
    dl_train = DataLoader(ImageList(train_items, tf_train), batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=nw, drop_last=len(train_items) > cfg["batch_size"], persistent_workers=nw > 0)
    dl_val = DataLoader(ImageList(val_items, tf_val), batch_size=cfg["batch_size"], num_workers=nw,
                        persistent_workers=nw > 0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["learning_rate"], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    amp_dtype = torch.bfloat16 if cuda and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler(enabled=cuda and amp_dtype == torch.float16)
    crit = torch.nn.CrossEntropyLoss()

    start, best_acc, best_loss = 0, -1.0, 1e9
    last = os.path.join(out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best_acc, best_loss = ck["epoch"] + 1, ck["best_acc"], ck["best_loss"]
        emit(type="log", message=f"Reanudando desde la época {start + 1}")

    for ep in range(start, cfg["epochs"]):
        model.train()
        tl, tn = 0.0, 0
        for x, y in dl_train:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=cuda):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(y)
            tn += len(y)
        sched.step()

        model.eval()
        vl, vc, vn = 0.0, 0, 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(dev), y.to(dev)
                with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=cuda):
                    lo = model(x)
                vl += crit(lo.float(), y).item() * len(y)
                vc += (lo.argmax(1) == y).sum().item()
                vn += len(y)
        vloss, vacc = vl / max(vn, 1), vc / max(vn, 1)
        emit(type="metric", step=ep + 1, max_steps=cfg["epochs"], loss=tl / max(tn, 1), eval_loss=vloss,
             val_acc=vacc, learning_rate=opt.param_groups[0]["lr"],
             vram_mb=torch.cuda.max_memory_allocated() // 2**20 if cuda else 0)

        if vacc > best_acc or (vacc == best_acc and vloss < best_loss):
            best_acc, best_loss = vacc, vloss
            torch.save(model.state_dict(), os.path.join(out, "model.pt"))
            json.dump({"model_name": cfg["model_name"], "classes": classes, "img_size": cfg["img_size"],
                       "mean": mean, "std": std, "val_acc": vacc, "epoch": ep + 1},
                      open(os.path.join(out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep, "best_acc": best_acc, "best_loss": best_loss}, last)

    if not os.path.exists(os.path.join(out, "model.pt")):
        raise RuntimeError("No se guardó ningún modelo (¿0 épocas?).")
    emit(type="log", message=f"Mejor precisión de validación: {best_acc:.3f}")
    emit(type="done")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit(type="error", message=f"{type(e).__name__}: {e}")
        sys.exit(1)
