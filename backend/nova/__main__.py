"""Arranque del servidor: `python -m nova` o el ejecutable empaquetado."""
import os

import uvicorn

from nova.api.app import app  # importación directa (no por nombre): necesaria para el ejecutable empaquetado


def main() -> None:
    # 127.0.0.1: nunca expuesto a la red.
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("NOVA_PORT", "8765")), log_level="info")


if __name__ == "__main__":
    main()
