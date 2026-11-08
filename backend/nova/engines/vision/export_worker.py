"""Exporta el clasificador a ONNX y verifica que onnxruntime reproduce la salida de PyTorch."""
import json
import os
import shutil
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit, load_model  # noqa: E402


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    out = cfg["out_dir"]
    os.makedirs(out, exist_ok=True)

    import numpy as np
    import onnxruntime as ort
    import torch

    model, meta = load_model(cfg["model_dir"], torch.device("cpu"))
    s = meta["img_size"]
    dummy = torch.randn(1, 3, s, s)
    path = os.path.join(out, "model.onnx")
    try:
        torch.onnx.export(model, dummy, path, input_names=["image"], output_names=["logits"],
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    except Exception as e:  # noqa: BLE001
        emit(type="log", message=f"Exportador clásico falló ({type(e).__name__}); probando el nuevo.")
        torch.onnx.export(model, (dummy,), path, input_names=["image"], output_names=["logits"], dynamo=True)
    shutil.copy(os.path.join(cfg["model_dir"], "meta.json"), os.path.join(out, "labels_and_preprocessing.json"))

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x = torch.randn(2, 3, s, s)
    with torch.no_grad():
        ref = model(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    emit(type="log", message=f"ONNX guardado en {path} ({os.path.getsize(path) // 1024} KB). Diferencia máx. con PyTorch: {diff:.2e}")
    if diff > 1e-3:
        raise RuntimeError(f"La salida de ONNX difiere de PyTorch ({diff:.2e}).")
    emit(type="done")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit(type="error", message=f"{type(e).__name__}: {e}")
        sys.exit(1)
