"""Entrenamiento LoRA/QLoRA (corre en el venv del motor LLM).

Uso: python train_worker.py job_config.json
Emite eventos JSON por línea en stdout: metric | log | error | done.
"""
import glob
import json
import os
import sys
import traceback


def emit(**ev):
    sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    out = os.path.dirname(os.path.abspath(cfg_path))

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              TrainerCallback)
    from trl import SFTConfig, SFTTrainer

    cuda = torch.cuda.is_available()
    emit(type="log", message=f"CUDA disponible: {cuda}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""))
    bf16 = cuda and torch.cuda.is_bf16_supported()

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs = {}
    if cfg["quant_4bit"] and cuda:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16)
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["dtype"] = torch.bfloat16 if bf16 else torch.float32
        if cuda:
            kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], **kwargs)
    model.config.use_cache = False

    ds = load_dataset("json", data_files=cfg["dataset_path"], split="train")
    ds = ds.select_columns(["messages"])
    eval_ds = None
    if len(ds) >= 50:
        split = ds.train_test_split(test_size=0.05, seed=42)
        ds, eval_ds = split["train"], split["test"]
    emit(type="log", message=f"Filas de entrenamiento: {len(ds)}" + (f", validación: {len(eval_ds)}" if eval_ds else ""))

    optim = cfg["optimizer"] if cuda else "adamw_torch"
    args = SFTConfig(
        output_dir=out,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["scheduler"],
        warmup_steps=cfg["warmup_ratio"],
        optim=optim,
        bf16=bf16, fp16=cuda and not bf16,
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps", save_steps=cfg["save_steps"], save_total_limit=2,
        eval_strategy="steps" if eval_ds else "no", eval_steps=cfg["save_steps"],
        max_length=cfg["max_seq_len"],
        report_to="none", disable_tqdm=True,
    )

    class Stream(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if not logs:
                return
            ev = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            ev.update(type="metric", step=state.global_step, max_steps=state.max_steps)
            if cuda:
                ev["vram_mb"] = torch.cuda.max_memory_allocated() // 2**20
            emit(**ev)

    trainer = SFTTrainer(
        model=model, args=args, train_dataset=ds, eval_dataset=eval_ds, processing_class=tok,
        peft_config=LoraConfig(
            r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
            target_modules="all-linear", task_type="CAUSAL_LM"),
        callbacks=[Stream()],
    )
    resume = bool(glob.glob(os.path.join(out, "checkpoint-*")))
    if resume:
        emit(type="log", message="Reanudando desde el último checkpoint.")
    trainer.train(resume_from_checkpoint=resume or None)
    trainer.model.save_pretrained(os.path.join(out, "adapter"))
    tok.save_pretrained(os.path.join(out, "adapter"))
    emit(type="log", message="Adaptador guardado.")
    emit(type="done")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit(type="error", message=f"{type(e).__name__}: {e}")
        sys.exit(1)
