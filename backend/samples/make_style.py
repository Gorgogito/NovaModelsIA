"""Genera un dataset sintético de ilustraciones planas para probar el LoRA de Stable Diffusion.

Uso: python make_style.py <carpeta_salida> <n_imagenes> <semilla>
Estilo: forma plana con contorno grueso sobre fondo liso, con la palabra clave 'novastyle'.
Se excluye a propósito la combinación 'triángulo morado' para comprobar si el LoRA generaliza.
Crea <carpeta>/NNN.png y <carpeta>/metadata.csv (columnas: file,text).
"""
import csv
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {"red": (220, 40, 40), "blue": (40, 90, 220), "green": (40, 170, 70), "yellow": (240, 200, 30),
          "orange": (240, 130, 30), "purple": (140, 60, 190)}
SHAPES = ["circle", "square", "triangle"]
BGS = {"white": (250, 250, 250), "black": (20, 20, 20), "gray": (150, 150, 150)}
HELD_OUT = ("triangle", "purple")
S = 512


def draw(shape, color, bg, rng):
    im = Image.new("RGB", (S, S), BGS[bg])
    d = ImageDraw.Draw(im)
    r = rng.randint(110, 170)
    cx, cy = rng.randint(r + 20, S - r - 20), rng.randint(r + 20, S - r - 20)
    outline = (255, 255, 255) if bg == "black" else (10, 10, 10)
    kw = {"fill": COLORS[color], "outline": outline, "width": 14}
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], **kw)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], **kw)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], **kw)
    return im


if __name__ == "__main__":
    out, n, seed = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        while True:
            shape, color, bg = rng.choice(SHAPES), rng.choice(list(COLORS)), rng.choice(list(BGS))
            if (shape, color) != HELD_OUT:
                break
        draw(shape, color, bg, rng).save(out / f"{i:03d}.png")
        rows.append((f"{i:03d}.png", f"a flat illustration of a {color} {shape} on a {bg} background, novastyle"))
    with (out / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "text"])
        w.writerows(rows)
    print(len(rows), "imágenes en", out)
