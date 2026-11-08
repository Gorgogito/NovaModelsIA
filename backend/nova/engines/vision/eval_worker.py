"""Evalúa un clasificador entrenado sobre un dataset de carpetas por clase.

Las clases se emparejan por nombre; las imágenes de clases que el modelo no conoce se omiten y se cuentan.
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ImageList, emit, list_classes, list_images, load_model, make_transforms  # noqa: E402


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))

    import torch
    from torch.utils.data import DataLoader

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, meta = load_model(cfg["model_dir"], dev)
    classes = meta["classes"]
    _, tf = make_transforms(meta["img_size"], meta["mean"], meta["std"], False)

    items, skipped = [], 0
    for c in list_classes(cfg["dataset_dir"]):
        imgs = list_images(cfg["dataset_dir"], c)
        if c in classes:
            items += [(p, classes.index(c)) for p in imgs[: cfg["max_per_class"]]]
        else:
            skipped += len(imgs)
    if not items:
        raise RuntimeError("El dataset no comparte ninguna clase con el modelo.")
    emit(type="log", message=f"Evaluando {len(items)} imágenes ({skipped} omitidas por clase desconocida)")

    n = len(classes)
    cm = [[0] * n for _ in range(n)]
    errors = []
    t0 = time.time()
    dl = DataLoader(ImageList(items, tf), batch_size=32, num_workers=0)
    k = 0
    with torch.no_grad():
        for x, y in dl:
            pred = model(x.to(dev)).argmax(1).cpu()
            for yi, pi in zip(y.tolist(), pred.tolist()):
                cm[yi][pi] += 1
                if yi != pi and len(errors) < 12:
                    errors.append({"file": os.path.basename(items[k][0]), "true": classes[yi], "pred": classes[pi]})
                k += 1
    total = sum(sum(r) for r in cm)
    correct = sum(cm[i][i] for i in range(n))
    per_class = []
    for i, c in enumerate(classes):
        support = sum(cm[i])
        tp = cm[i][i]
        predicted = sum(cm[r][i] for r in range(n))
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        per_class.append({"class": c, "support": support, "precision": round(prec, 4), "recall": round(rec, 4),
                          "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0})
    emit(type="result", result={
        "rows": total, "accuracy": round(correct / total, 4), "skipped": skipped, "classes": classes,
        "confusion": cm, "per_class": per_class, "errors": errors,
        "macro_f1": round(sum(p["f1"] for p in per_class) / n, 4),
        "ms_per_image": round((time.time() - t0) / total * 1000, 2),
    })
    emit(type="done")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit(type="error", message=f"{type(e).__name__}: {e}")
        sys.exit(1)
