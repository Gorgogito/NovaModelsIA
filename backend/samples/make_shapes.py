"""Genera un dataset sintético de formas (círculo, cuadrado, triángulo) para probar el motor de visión.

Uso: python make_shapes.py <carpeta_salida> <imagenes_por_clase> <semilla>
Cada imagen tiene color, tamaño, posición, fondo y ruido aleatorios.
"""
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


def make(shape: str, rng: random.Random, size: int = 128) -> Image.Image:
    bg = tuple(rng.randint(150, 255) for _ in range(3))
    im = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(im)
    for _ in range(rng.randint(0, 6)):  # distractores de ruido
        x, y = rng.randint(0, size), rng.randint(0, size)
        d.line([x, y, x + rng.randint(-20, 20), y + rng.randint(-20, 20)],
               fill=tuple(rng.randint(120, 255) for _ in range(3)), width=1)
    col = tuple(rng.randint(0, 120) for _ in range(3))
    r = rng.randint(size // 6, size // 3)
    cx = rng.randint(r, size - r)
    cy = rng.randint(r, size - r)
    if shape == "circulo":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    elif shape == "cuadrado":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=col)
    else:
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=col)
    if rng.random() < 0.5:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.2)))
    return im


if __name__ == "__main__":
    out, n, seed = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    rng = random.Random(seed)
    for shape in ("circulo", "cuadrado", "triangulo"):
        (out / shape).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            make(shape, rng).save(out / shape / f"{i:03d}.png")
    print("generado en", out)
