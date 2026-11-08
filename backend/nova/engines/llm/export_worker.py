"""Exporta: fusiona el adaptador LoRA en el modelo base y, opcionalmente, convierte a GGUF."""
import json
import os
import subprocess
import sys
import traceback


def emit(**ev):
    sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def ensure_llama_cpp(path):
    if os.path.exists(os.path.join(path, "convert_hf_to_gguf.py")):
        return
    emit(type="log", message="Descargando llama.cpp (solo el script de conversión)...")
    subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", path], check=True)


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    out = cfg["out_dir"]
    merged = os.path.join(out, "merged")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    emit(type="log", message="Fusionando adaptador con el modelo base (en CPU)...")
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, cfg["adapter_path"]).merge_and_unload()
    model.save_pretrained(merged, safe_serialization=True)
    AutoTokenizer.from_pretrained(cfg["base_model"]).save_pretrained(merged)
    emit(type="log", message=f"Modelo fusionado en {merged}")

    if cfg.get("gguf"):
        ensure_llama_cpp(cfg["llama_cpp_dir"])
        outfile = os.path.join(out, f"model-{cfg['gguf_quant']}.gguf")
        subprocess.run([sys.executable, os.path.join(cfg["llama_cpp_dir"], "convert_hf_to_gguf.py"),
                        merged, "--outfile", outfile, "--outtype", cfg["gguf_quant"].lower()], check=True)
        emit(type="log", message=f"GGUF generado: {outfile}")
    emit(type="done")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit(type="error", message=f"{type(e).__name__}: {e}")
        sys.exit(1)
