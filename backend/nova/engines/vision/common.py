"""Utilidades compartidas por los procesos del motor de visión (corren en su venv)."""
import json
import os
import random
import sys

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def emit(**ev):
    sys.stdout.write(json.dumps(ev, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def list_classes(root):
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def list_images(root, cls):
    base = os.path.join(root, cls)
    out = []
    for dp, _, fns in os.walk(base):
        out += [os.path.join(dp, f) for f in fns if os.path.splitext(f)[1].lower() in IMG_EXT]
    return sorted(out)


def split(root, classes, val_split, seed=42):
    """Partición estratificada y determinista: devuelve (train, val) como listas (ruta, índice_clase)."""
    train, val = [], []
    rng = random.Random(seed)
    for ci, c in enumerate(classes):
        items = list_images(root, c)
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_split)) if len(items) > 1 else 0
        val += [(p, ci) for p in items[:n_val]]
        train += [(p, ci) for p in items[n_val:]]
    return train, val


def make_transforms(img_size, mean, std, augment):
    from torchvision import transforms as T

    norm = T.Normalize(mean=mean, std=std)
    train = [T.RandomResizedCrop(img_size, scale=(0.6, 1.0)), T.RandomHorizontalFlip()]
    if augment:
        train.append(T.ColorJitter(0.2, 0.2, 0.2))
    train += [T.ToTensor(), norm]
    if not augment:
        train = [T.Resize((img_size, img_size)), T.ToTensor(), norm]
    val = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])
    return T.Compose(train), val


class ImageList:
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from PIL import Image

        p, y = self.items[i]
        with Image.open(p) as im:
            return self.tf(im.convert("RGB")), y


def load_model(out_dir, device):
    """Reconstruye el modelo entrenado desde meta.json + model.pt."""
    import timm
    import torch

    meta = json.load(open(os.path.join(out_dir, "meta.json"), encoding="utf-8"))
    model = timm.create_model(meta["model_name"], pretrained=False, num_classes=len(meta["classes"]))
    model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt"), map_location="cpu", weights_only=True))
    return model.to(device).eval(), meta
