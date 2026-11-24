"""Keeps exactly one model family in memory and cleans up on every switch.

On a 4 GB GPU, Kokoro, the Stable Audio 3 SFX model and the music model must
never be resident together. ``ModelManager.use("sfx")`` unloads whatever is
loaded, runs ``gc.collect()`` + ``torch.cuda.empty_cache()``, and only then
loads the requested engine.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from . import config as C
from .device import DevicePlan, cleanup_memory, plan_device
from .engines.base import Engine, Logger, stderr_logger
from .errors import RequestError
from .models import ModelPaths

EngineFactory = Callable[[str, DevicePlan], Engine]


class ModelManager:
    def __init__(
        self,
        paths: Optional[ModelPaths] = None,
        *,
        device: str = "auto",
        dtype: str = "auto",
        sa3_source: str = "auto",
        log: Logger = stderr_logger,
        progress: bool = True,
        engine_factory: Optional[EngineFactory] = None,
        cleanup: Callable[[], None] = cleanup_memory,
    ):
        self.paths = paths or ModelPaths()
        self.device = device
        self.dtype = dtype
        self.sa3_source = sa3_source
        self.log = log
        self.progress = progress
        self._factory = engine_factory or self._default_factory
        self._cleanup = cleanup
        self._engine: Optional[Engine] = None
        self._key: Optional[str] = None
        self._plans: Dict[str, DevicePlan] = {}

    # ------------------------------------------------------------------ plans
    def plan_for(self, key: str) -> DevicePlan:
        """Device/precision decision for an engine key (``tts`` / ``sfx`` / ``music``)."""
        if key not in C.MODES:
            raise RequestError(f"unknown mode {key!r}; choose one of {', '.join(C.MODES)}")
        if key not in self._plans:
            if key == "tts":
                # Kokoro is tiny and trained/validated in fp32.
                plan = plan_device(
                    self.device, "fp32", min_vram_gb=C.MIN_VRAM_GB_KOKORO, min_free_gb=None,
                    cuda_default_dtype="float32",
                )
            else:
                plan = plan_device(
                    self.device, self.dtype, min_vram_gb=C.MIN_VRAM_GB_SA3, min_free_gb=C.MIN_FREE_VRAM_GB_SA3,
                    cuda_default_dtype="float16",
                )
            self._plans[key] = plan
        return self._plans[key]

    def _default_factory(self, key: str, plan: DevicePlan) -> Engine:
        if key == "tts":
            from .engines.kokoro_engine import KokoroEngine

            return KokoroEngine(self.paths, plan, self.log)
        from .engines.sa3_engine import Sa3Engine

        return Sa3Engine(self.paths, plan, C.MODE_TO_VARIANT[key], self.sa3_source, self.log, self.progress)

    # ------------------------------------------------------------- lifecycle
    @property
    def current(self) -> Optional[str]:
        return self._key

    def use(self, key: str) -> Engine:
        """Return a loaded engine for ``key``, unloading any other model first."""
        if self._engine is not None and self._key == key and self._engine.loaded:
            return self._engine
        self.release()
        plan = self.plan_for(key)
        self.log(f"[{key}] device: {plan.device}/{plan.dtype} - {plan.reason}")
        engine = self._factory(key, plan)
        try:
            engine.load()
        except BaseException:
            try:
                engine.unload()
            finally:
                self._cleanup()
            raise
        self._engine, self._key = engine, key
        return engine

    def release(self) -> None:
        """Unload the resident model and run the memory-cleanup ritual."""
        engine, self._engine, self._key = self._engine, None, None
        if engine is not None:
            self.log(f"unloading {engine.display_name or engine.key}")
            try:
                engine.unload()
            finally:
                del engine
                self._cleanup()

    def __enter__(self) -> "ModelManager":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()
