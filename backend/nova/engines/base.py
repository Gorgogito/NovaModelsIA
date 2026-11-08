"""Contrato común que implementa cada motor (LLM, visión, audio, ...)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from pydantic import BaseModel

EventSink = Callable[[dict[str, Any]], None]


class EngineInfo(BaseModel):
    id: str
    name: str
    domain: str  # llm | vision | audio | diffusion | ...
    version: str


class Engine(ABC):
    """Cada motor corre en su propio proceso y entorno virtual."""

    @abstractmethod
    def info(self) -> EngineInfo: ...

    @abstractmethod
    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Devuelve la lista de errores (vacía si es válida)."""

    @abstractmethod
    def estimate_vram_mb(self, config: dict[str, Any]) -> int: ...

    @abstractmethod
    def train(self, config: dict[str, Any], emit: EventSink) -> None:
        """Emite eventos {type: 'metric'|'log'|'checkpoint', ...}."""

    @abstractmethod
    def stop(self) -> None: ...
