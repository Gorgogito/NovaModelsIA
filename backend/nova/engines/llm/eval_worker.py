"""Evaluación de un modelo (base o con adaptador LoRA) sobre un dataset JSONL en formato messages.

Métricas: pérdida y perplejidad sobre la respuesta de referencia, F1 de tokens y
coincidencia exacta de la respuesta generada (greedy) frente a la referencia.
"""
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter


def emit(**ev):
    sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def toks(s):
    return re.findall(r"\w+", s.lower())


def f1(pred, ref):
    p, r = toks(pred), toks(ref)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(r)
    return 2 * prec * rec / (prec + rec)


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    kwargs = {}
    if cfg.get("quant_4bit") and cuda:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.bfloat16 if cuda else torch.float32
        if cuda:
            kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kwargs)
    if cfg.get("adapter_path"):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg["adapter_path"])
    model.eval()

    rows = []
    with open(cfg["dataset_path"], encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line)["messages"])
            if len(rows) >= cfg["max_rows"]:
                break
    emit(type="log", message=f"Evaluando {len(rows)} filas")

    losses, f1s, exact, samples, gen_times = [], [], [], [], []
    total_nll, total_tokens = 0.0, 0
    for i, msgs in enumerate(rows):
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        ref = msgs[-1]["content"]
        p_ids = tok(prompt, return_tensors="pt")["input_ids"]
        r_ids = tok(ref + (tok.eos_token or ""), return_tensors="pt", add_special_tokens=False)["input_ids"]
        ids = torch.cat([p_ids, r_ids], dim=1).to(model.device)
        labels = ids.clone()
        labels[:, : p_ids.shape[1]] = -100  # solo se mide la respuesta
        with torch.no_grad():
            loss = model(input_ids=ids, labels=labels).loss.item()
        n = r_ids.shape[1]
        total_nll += loss * n
        total_tokens += n
        losses.append(loss)

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(input_ids=p_ids.to(model.device), max_new_tokens=cfg["max_new_tokens"],
                                 do_sample=False, pad_token_id=tok.eos_token_id)
        gen_times.append(time.time() - t0)
        pred = tok.decode(out[0][p_ids.shape[1]:], skip_special_tokens=True).strip()
        f1s.append(f1(pred, ref))
        exact.append(float(pred.strip().lower() == ref.strip().lower()))
        if len(samples) < 20:
            samples.append({"prompt": msgs[-2]["content"], "reference": ref, "prediction": pred})
        emit(type="log", message=f"{i + 1}/{len(rows)}")

    mean_loss = total_nll / max(total_tokens, 1)
    emit(type="result", result={
        "rows": len(rows),
        "loss": round(mean_loss, 4),
        "perplexity": round(math.exp(min(mean_loss, 20)), 3),
        "f1": round(sum(f1s) / len(f1s), 4),
        "exact_match": round(sum(exact) / len(exact), 4),
        "avg_gen_seconds": round(sum(gen_times) / len(gen_times), 3),
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
