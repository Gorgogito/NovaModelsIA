# NovaModelsIA

Plataforma de escritorio para **crear, entrenar, ajustar, evaluar y servir modelos de IA en local**, sin APIs de pago ni suscripciones.

| Tipo | Qué hace | Base |
|---|---|---|
| Texto (LLM) | Fine-tuning QLoRA, chat, exportar a GGUF | Transformers, PEFT, TRL |
| Visión | Clasificación de imágenes, matriz de confusión, exportar a ONNX | timm (PyTorch) |
| Audio | Fine-tuning de Whisper, transcripción, WER/CER | Transformers |
| Imagen | LoRA de Stable Diffusion 1.5, generación, comparación | Diffusers, PEFT |
| API local | Compatible con OpenAI en `http://127.0.0.1:8765/v1` | FastAPI |

## Arquitectura

```
frontend/   Tauri 2 + React/TypeScript   (ventana de escritorio)
backend/    FastAPI (Python, gestionado con uv): proyectos, cola de trabajos, API local
  nova/engines/<motor>/   scripts de cada motor; corren en su PROPIO entorno virtual
data/       entornos (envs/), modelos (models/), proyectos (projects/), base de datos (nova.db)
```

Cada motor (`llm`, `vision`, `audio`, `image`) tiene su venv aislado en `data/envs/<motor>` y se comunica con el backend por JSON de una línea. Se instalan la primera vez que se usan desde la propia aplicación (descargan varios GB).

## Requisitos

Windows 11, GPU NVIDIA con CUDA (probado con RTX 3050 de 8 GB), Python 3.12, Node.js y Rust (solo para desarrollar). Con 8 GB de VRAM: QLoRA de modelos de 0,5–3B, Whisper tiny/base/small, LoRA de SD 1.5 a 512 px. SDXL no cabe para entrenar.

## Desarrollo

```
cd backend  && python -m uv sync          # primera vez
cd frontend && npm install                # primera vez
cd frontend && npm run tauri dev          # abre la ventana; ella arranca y cierra el backend
```

El backend también puede arrancarse solo: `cd backend && python -m uv run python -m nova`.

## Pruebas

```
cd backend  && python -m uv run pytest -q     # motores simulados, sin GPU
cd frontend && npm run tour                   # recorre todas las pantallas en Edge y hace capturas
cd frontend && npm run tour:tauri             # lo mismo dentro de la ventana real de Tauri
```

Los recorridos dejan capturas e informe en `frontend/e2e/screenshots*/report.html`. Detectan errores de consola, peticiones fallidas y desbordes. `tour:tauri` necesita el puerto 8765 libre. Con `TAURI_EXE=ruta\frontend.exe` prueba un ejecutable de release.

## Empaquetar

```
cd backend  && python -m uv run python build_exe.py   # dist/nova-backend/ (nova-backend.exe + uv.exe, ~120 MB)
cd frontend && npm run build:app                      # instalador NSIS en src-tauri/target/release/bundle/nsis/
```

Usa `npm run build:app`, no `npm run tauri build`: el segundo no incluye el backend y la app instalada no podría arrancarlo. El script avisa si falta construir antes el backend. Si cambias el icono, ejecuta `touch src-tauri/build.rs` antes de compilar: Cargo no vuelve a incrustar el icono cuando solo cambia el `.ico`.

PyTorch no va dentro del instalador: cada motor se instala en el primer uso con el `uv.exe` incluido (necesita internet esa vez). En la app instalada los datos van a `%LOCALAPPDATA%\NovaModelsIA`; la variable `NOVA_DATA` permite elegir otro disco (conviene un SSD: los modelos pesan decenas de GB y cargan lento desde un disco duro).

## API local compatible con OpenAI

`http://127.0.0.1:8765/v1`, solo en tu equipo. La pantalla **API local** muestra la URL, los modelos disponibles y ejemplos.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8765/v1", api_key="no-se-usa")
r = client.chat.completions.create(model="nova-llm-16", messages=[{"role": "user", "content": "Hola"}])
```

- Chat (con `stream`), transcripción de audio (`json`/`text`) e imágenes (`b64_json`, 128–768 px). Sin `n>1`, `stop`, herramientas, URLs de imagen ni marcas de tiempo.
- Solo sirve modelos ya descargados o entrenados: nunca descarga nada por su cuenta.
- Un único modelo cargado a la vez en la GPU: pedir otro descarga el actual. Con un entrenamiento en curso responde 503 (`gpu_busy`).
- Clave opcional: define `NOVA_API_KEY` antes de arrancar y se exigirá `Authorization: Bearer <clave>`.

## Licencias y decisiones

- Cada modelo base muestra su licencia (consultada al Hub). Stable Diffusion 1.5 usa CreativeML OpenRAIL-M, con restricciones de uso. Comprueba la licencia de tus datasets: entrenar con ellos es responsabilidad tuya.
- **Stable Diffusion se usa sin su comprobador de seguridad** (decisión del propietario del proyecto): las imágenes generadas no se filtran.
- Antes de usar un modelo de otra familia, revisa su licencia: por ejemplo, Qwen2.5-3B tiene una licencia propia, distinta de la Apache-2.0 de los de 0,5B, 1,5B y 7B.
