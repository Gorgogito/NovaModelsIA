"""Genera un dataset sintético de voz en español (Windows SAPI, voz Helena) para probar el motor de audio.

Uso: python make_speech.py <carpeta_salida> <n_frases> <semilla>
Crea <carpeta>/clip_000.wav ... y <carpeta>/metadata.csv (columnas: file,text).
Las frases combinan términos técnicos; semillas distintas producen combinaciones distintas.
Solo sirve para validar el flujo: es una única voz sintética, no habla real.
"""
import csv
import random
import subprocess
import sys
import tempfile
from pathlib import Path

MODELOS = ["Qwen", "ResNet", "Whisper", "MobileNet", "EfficientNet", "NovaModelsIA"]
TECNICAS = ["la cuantización", "LoRA", "el gradiente acumulado", "la validación cruzada", "el aumento de datos",
            "la perplejidad", "el checkpoint", "la tasa de aprendizaje"]
ACCIONES = ["reducir la memoria", "mejorar la precisión", "acelerar el entrenamiento", "evitar el sobreajuste",
            "medir el error", "ajustar el modelo"]
PLANTILLAS = [
    "El modelo {m} usa {t} para {a}.",
    "Con {m} aplicamos {t} y logramos {a}.",
    "Para {a} conviene revisar {t} en {m}.",
    "En {m}, {t} ayuda a {a}.",
]


def sentences(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out: set[str] = set()
    while len(out) < n:
        out.add(rng.choice(PLANTILLAS).format(m=rng.choice(MODELOS), t=rng.choice(TECNICAS), a=rng.choice(ACCIONES)))
    return sorted(out)


PS = r"""
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SelectVoice('Microsoft Helena Desktop')
Import-Csv -Path '{csv}' -Encoding UTF8 | ForEach-Object {{
    $s.Rate = [int]$_.rate
    $s.SetOutputToWaveFile((Join-Path '{out}' $_.file))
    $s.Speak($_.text)
    $s.SetOutputToNull()
}}
"""

if __name__ == "__main__":
    out, n, seed = Path(sys.argv[1]).resolve(), int(sys.argv[2]), int(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows = [{"file": f"clip_{i:03d}.wav", "text": t, "rate": rng.randint(-2, 1)} for i, t in enumerate(sentences(n, seed))]
    with tempfile.TemporaryDirectory() as tmp:
        spec = Path(tmp) / "spec.csv"
        with spec.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "text", "rate"])
            w.writeheader()
            w.writerows(rows)
        ps1 = Path(tmp) / "speak.ps1"
        ps1.write_text(PS.format(csv=spec, out=out), encoding="utf-8-sig")
        subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ps1)], check=True)
    with (out / "metadata.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "text"])
        for r in rows:
            w.writerow([r["file"], r["text"]])
    print(len(rows), "clips en", out)
