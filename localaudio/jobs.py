"""Request objects (``SpeechJob`` / ``SoundJob``) and their validation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional

from . import config as C
from .errors import RequestError
from . import textchunk as T

EXTEND_METHODS = ("auto", "continue", "loop")


@dataclass
class SpeechJob:
    """Text-to-speech request (Kokoro)."""

    text: str
    voice: str = C.DEFAULT_VOICE
    speed: float = 1.0
    lang: Optional[str] = None  # Kokoro language code; inferred from the voice when None
    seed: Optional[int] = None
    max_chars: int = T.DEFAULT_MAX_CHARS
    sentence_pause: float = T.DEFAULT_SENTENCE_PAUSE
    paragraph_pause: float = T.DEFAULT_PARAGRAPH_PAUSE

    def validate(self) -> "SpeechJob":
        if not (self.text or "").strip():
            raise RequestError("text is empty; pass --text or --text-file")
        lo, hi = C.KOKORO_SPEED_RANGE
        if not (lo <= float(self.speed) <= hi):
            raise RequestError(f"--speed must be between {lo} and {hi} (got {self.speed})")
        if self.sentence_pause < 0 or self.paragraph_pause < 0:
            raise RequestError("pauses must be >= 0")
        if self.max_chars < 40:
            raise RequestError("--max-chars must be at least 40")
        return self


@dataclass
class SoundJob:
    """Sound-effect or music request (Stable Audio 3 Small)."""

    prompt: str
    mode: str = "sfx"  # "sfx" | "music"
    duration: float = 10.0
    steps: int = C.SA3_DEFAULT_STEPS
    cfg_scale: float = C.SA3_DEFAULT_CFG
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    extend_method: str = "auto"  # auto | continue | loop (only used beyond 120 s)
    context_seconds: float = 30.0  # audio carried over when continuing past 120 s
    crossfade_seconds: float = 2.0  # stitch / loop crossfade length
    loopable: bool = False  # make the result loop seamlessly

    def validate(self) -> "SoundJob":
        if self.mode not in ("sfx", "music"):
            raise RequestError(f"mode must be 'sfx' or 'music', not {self.mode!r}")
        if not (self.prompt or "").strip():
            raise RequestError("prompt is empty; pass --prompt")
        if not (C.SA3_MIN_SECONDS <= float(self.duration)):
            raise RequestError(f"--duration must be at least {C.SA3_MIN_SECONDS} s")
        if self.duration > 3600:
            raise RequestError("--duration above one hour is not supported")
        if not (1 <= int(self.steps) <= C.SA3_MAX_STEPS):
            raise RequestError(
                f"--steps must be between 1 and {C.SA3_MAX_STEPS} (the Small models are tuned for "
                f"{C.SA3_DEFAULT_STEPS}); got {self.steps}"
            )
        if self.cfg_scale < 0:
            raise RequestError("--cfg-scale must be >= 0")
        if self.extend_method not in EXTEND_METHODS:
            raise RequestError(f"--extend-method must be one of {', '.join(EXTEND_METHODS)}")
        if not (4.0 <= self.context_seconds <= C.SA3_MAX_SECONDS - 10.0):
            raise RequestError(f"--context must be between 4 and {C.SA3_MAX_SECONDS - 10:.0f} seconds")
        if not (0.0 <= self.crossfade_seconds <= 20.0):
            raise RequestError("--crossfade must be between 0 and 20 seconds")
        return self

    @property
    def needs_extension(self) -> bool:
        return self.duration > C.SA3_MAX_SECONDS + 1e-6


def job_from_dict(mode: str, data: Dict[str, Any]) -> Any:
    """Build a job from a plain dict (batch files). Unknown keys raise RequestError."""
    cls = SpeechJob if mode == "tts" else SoundJob
    allowed = {f.name for f in fields(cls)}
    aliases = {"prompt": "text"} if mode == "tts" else {"text": "prompt"}
    clean: Dict[str, Any] = {}
    for key, value in data.items():
        key = aliases.get(key, key)
        if key in ("mode", "output", "enrich", "enrich_options") and key not in allowed:
            continue
        if key not in allowed:
            raise RequestError(f"unknown field {key!r} for {mode} job; allowed: {sorted(allowed)}")
        clean[key] = value
    if mode != "tts":
        clean["mode"] = mode
    try:
        return cls(**clean).validate()
    except TypeError as exc:
        raise RequestError(f"bad {mode} job: {exc}") from exc
