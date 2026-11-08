"""Entrenamiento de un LoRA de Stable Diffusion 1.x sobre imágenes con descripción.

Uso: python train_worker.py job_config.json
Bucle propio (sin accelerate): UNet en bf16/fp16 congelado + adaptadores LoRA en fp32.
Guarda muestras de vista previa en <job>/samples y el LoRA final en <job>/lora.
Reanuda desde last.pt si existe.
"""
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import emit, pick_dtype, read_metadata  # noqa: E402


def main(cfg_path):
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    out = os.path.dirname(os.path.abspath(cfg_path))
    os.makedirs(os.path.join(out, "samples"), exist_ok=True)

    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from diffusers.optimization import get_scheduler
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image
    from torchvision import transforms as T
    from transformers import CLIPTextModel, CLIPTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("El entrenamiento de LoRA para Stable Diffusion necesita una GPU con CUDA.")
    dev = torch.device("cuda")
    dtype = pick_dtype()
    torch.manual_seed(cfg["seed"])
    emit(type="log", message=f"GPU: {torch.cuda.get_device_name(0)} · precisión {str(dtype).split('.')[-1]}")

    base = cfg["base_model"]
    kw = {"torch_dtype": dtype, "variant": "fp16"}
    tokenizer = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", **kw).to(dev)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", **kw).to(dev)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", **kw).to(dev)
    noise_sched = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    for m in (text_encoder, vae, unet):
        m.requires_grad_(False)

    unet.add_adapter(LoraConfig(r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], init_lora_weights="gaussian",
                                target_modules=["to_k", "to_q", "to_v", "to_out.0"]))
    for p in unet.parameters():  # los pesos entrenables en fp32; la base queda en bf16/fp16
        if p.requires_grad:
            p.data = p.data.float()
    if cfg["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    trainable = [p for p in unet.parameters() if p.requires_grad]
    emit(type="log", message=f"Parámetros entrenables (LoRA): {sum(p.numel() for p in trainable) / 1e6:.2f} M")

    rows = read_metadata(cfg["dataset_dir"])
    res = cfg["resolution"]
    ops = [T.Resize(res, interpolation=T.InterpolationMode.BILINEAR),
           T.CenterCrop(res) if cfg["center_crop"] else T.RandomCrop(res)]
    if cfg["random_flip"]:
        ops.append(T.RandomHorizontalFlip())
    tf = T.Compose(ops + [T.ToTensor(), T.Normalize([0.5], [0.5])])

    class DS:  # sin num_workers: en Windows los workers no pueden serializar clases locales
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            with Image.open(os.path.join(cfg["dataset_dir"], rows[i]["file"])) as im:
                px = tf(im.convert("RGB"))
            ids = tokenizer(rows[i]["text"], max_length=tokenizer.model_max_length, padding="max_length",
                            truncation=True, return_tensors="pt").input_ids[0]
            return px, ids

    dl = torch.utils.data.DataLoader(DS(), batch_size=cfg["batch_size"], shuffle=True, drop_last=len(rows) > cfg["batch_size"])

    if cfg["use_8bit_adam"]:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(trainable, lr=cfg["learning_rate"], weight_decay=1e-2)
    else:
        opt = torch.optim.AdamW(trainable, lr=cfg["learning_rate"], weight_decay=1e-2)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=max(1, cfg["max_steps"] // 20),
                          num_training_steps=cfg["max_steps"])
    scaler = torch.amp.GradScaler(enabled=dtype == torch.float16)

    step = 0
    last = os.path.join(out, "last.pt")
    names = [n for n, p in unet.named_parameters() if p.requires_grad]
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=True)
        for n, p in unet.named_parameters():
            if n in ck["lora"]:
                p.data.copy_(ck["lora"][n].to(dev))
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        emit(type="log", message=f"Reanudando desde el paso {step}")

    def save_lora(path):
        sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
        StableDiffusionPipeline.save_lora_weights(save_directory=path, unet_lora_layers=sd, safe_serialization=True)

    def preview(at):
        """Genera una imagen con el LoRA actual para ver cómo evoluciona el entrenamiento."""
        unet.eval()
        pipe = StableDiffusionPipeline(vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                                       scheduler=DPMSolverMultistepScheduler.from_config(noise_sched.config),
                                       safety_checker=None, feature_extractor=None, requires_safety_checker=False)
        pipe.set_progress_bar_config(disable=True)
        g = torch.Generator(device=dev).manual_seed(cfg["seed"])
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            img = pipe(cfg["validation_prompt"], num_inference_steps=20, guidance_scale=7.5, height=res, width=res,
                       generator=g).images[0]
        rel = f"samples/step_{at:05d}.png"
        img.save(os.path.join(out, rel))
        emit(type="sample", step=at, file=rel)
        unet.train()
        torch.cuda.empty_cache()

    unet.train()
    done = step >= cfg["max_steps"]
    micro = 0
    opt.zero_grad(set_to_none=True)
    while not done:
        for px, ids in dl:
            px, ids = px.to(dev, dtype), ids.to(dev)
            with torch.no_grad():
                latents = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
                enc = text_encoder(ids)[0]
            noise = torch.randn_like(latents)
            t = torch.randint(0, noise_sched.config.num_train_timesteps, (latents.shape[0],), device=dev).long()
            noisy = noise_sched.add_noise(latents, noise, t)
            target = noise if noise_sched.config.prediction_type == "epsilon" else noise_sched.get_velocity(latents, noise, t)
            with torch.autocast("cuda", dtype=dtype):
                pred = unet(noisy, t, enc).sample
            loss = F.mse_loss(pred.float(), target.float())
            scaler.scale(loss / cfg["grad_accum"]).backward()
            micro += 1
            if micro % cfg["grad_accum"]:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            emit(type="metric", step=step, max_steps=cfg["max_steps"], loss=loss.item(),
                 learning_rate=sched.get_last_lr()[0], vram_mb=torch.cuda.max_memory_allocated() // 2**20)
            if step % cfg["save_steps"] == 0 and step < cfg["max_steps"]:
                torch.save({"lora": {n: p.detach().cpu() for n, p in unet.named_parameters() if n in names},
                            "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, last)
            if cfg["sample_steps"] and step % cfg["sample_steps"] == 0 and step < cfg["max_steps"]:
                preview(step)
            if step >= cfg["max_steps"]:
                done = True
                break

    save_lora(os.path.join(out, "lora"))
    preview(step)
    json.dump({"base_model": base, "steps": step, "resolution": res, "rank": cfg["lora_rank"],
               "validation_prompt": cfg["validation_prompt"]}, open(os.path.join(out, "meta.json"), "w", encoding="utf-8"))
    if os.path.exists(last):
        os.remove(last)
    emit(type="log", message="LoRA guardado.")
    emit(type="done")


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        emit(type="error", message=f"{type(e).__name__}: {e}")
        sys.exit(1)
