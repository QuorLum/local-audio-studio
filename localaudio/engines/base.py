"""Common engine interface."""

from __future__ import annotations

import abc
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import numpy as np

from ..device import DevicePlan
from ..models import ModelPaths

Logger = Callable[[str], None]


def stderr_logger(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass
class AudioResult:
    """Generated audio: ``(channels, frames)`` float32 plus provenance."""

    audio: np.ndarray
    sample_rate: int
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def channels(self) -> int:
        return int(self.audio.shape[0])

    @property
    def duration(self) -> float:
        return self.audio.shape[-1] / float(self.sample_rate)


class Engine(abc.ABC):
    """A model family that can be loaded, used and fully unloaded."""

    key: str = ""  # "tts" | "sfx" | "music"
    display_name: str = ""

    def __init__(self, paths: ModelPaths, plan: DevicePlan, log: Logger = stderr_logger):
        self.paths = paths
        self.plan = plan
        self.log = log
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @abc.abstractmethod
    def load(self) -> None:
        """Read weights into memory (idempotent)."""

    @abc.abstractmethod
    def unload(self) -> None:
        """Drop every reference to model memory. The manager then runs cleanup."""

    @abc.abstractmethod
    def generate(self, job: Any) -> AudioResult:
        """Run one job."""
