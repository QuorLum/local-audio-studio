"""High-level API shared by the CLI and Python users."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import config as C
from . import prompts as P
from . import textchunk as T
from .audio import fade, peak_normalize, resample, trim_silence, write_audio
from .engines.base import Logger, stderr_logger
from .engines.kokoro_engine import resolve_voice_files
from .errors import DependencyError, LocalAudioError, RequestError
from .jobs import SoundJob, SpeechJob, job_from_dict
from .manager import ModelManager
from .models import ModelPaths


# prompt-enrichment options may be given at the top level of a batch job, like the CLI flags
ENRICH_KEYS = frozenset({"mic", "room", "texture", "bpm", "genre", "instruments", "mood", "mix", "vocals", "stem",
                         "tags"})


@dataclass
class PostOptions:
    """What happens to the audio after generation and before it is written."""

    normalize_db: Optional[float] = -1.0  # peak target in dBFS; None disables
    trim_silence: bool = False
    fade_in: float = 0.0
    fade_out: float = 0.0
    edge_fade_ms: float = 5.0  # click guard for sfx/music
    sample_rate: Optional[int] = None  # resample to this rate (e.g. 48000)
    bit_depth: str = "16"  # 16 | 24 | 32f


@dataclass
class GenerationResult:
    path: Path
    mode: str
    sample_rate: int
    channels: int
    duration: float
    elapsed: float
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


def default_output_path(mode: str, directory: Optional[Path] = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return (directory or Path.cwd() / "output") / f"{mode}_{stamp}.wav"


def apply_post(audio: np.ndarray, sr: int, post: PostOptions, mode: str, loopable: bool = False):
    """Trim, resample, fade and normalize. Returns ``(audio, sample_rate)``."""
    if post.trim_silence:
        audio = trim_silence(audio, sr)
    if post.sample_rate and int(post.sample_rate) != sr:
        audio = resample(audio, sr, int(post.sample_rate))
        sr = int(post.sample_rate)
    edge = 0.0 if (mode == "tts" or loopable) else post.edge_fade_ms / 1000.0
    fade_in, fade_out = max(post.fade_in, edge), max(post.fade_out, edge)
    if fade_in or fade_out:
        audio = fade(audio, sr, fade_in, fade_out)
    if post.normalize_db is not None:
        audio = peak_normalize(audio, post.normalize_db)
    return audio, sr


class Studio:
    """Unified TTS + SFX + music generator.

    >>> with Studio() as studio:                                    # doctest: +SKIP
    ...     studio.tts("Hello!", output="hello.wav")
    ...     studio.sfx("door slam in a hallway", duration=3, output="slam.wav")
    """

    def __init__(
        self,
        models_dir: Optional[str] = None,
        *,
        device: str = "auto",
        dtype: str = "auto",
        sa3_source: str = "auto",
        quiet: bool = False,
        log: Optional[Logger] = None,
        manager: Optional[ModelManager] = None,
    ):
        self.paths = ModelPaths(models_dir)
        self.quiet = quiet
        self.log: Logger = (lambda _m: None) if quiet else (log or stderr_logger)
        self.manager = manager or ModelManager(
            self.paths, device=device, dtype=dtype, sa3_source=sa3_source, log=self.log, progress=not quiet
        )

    # ------------------------------------------------------------ lifecycle
    def close(self) -> None:
        self.manager.release()

    def __enter__(self) -> "Studio":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------ rendering
    def _render(self, mode: str, job: Any, output: Optional[str], post: Optional[PostOptions],
                extra_info: Optional[Dict[str, Any]] = None) -> GenerationResult:
        post = post or PostOptions()
        started = time.time()
        engine = self.manager.use(mode)
        self.log(f"generating {mode} ...")
        result = engine.generate(job)
        audio, sr = apply_post(result.audio, result.sample_rate, post, mode, getattr(job, "loopable", False))
        path = write_audio(output or default_output_path(mode), audio, sr, post.bit_depth)
        info = dict(result.info)
        if extra_info:
            info.update(extra_info)
        elapsed = time.time() - started
        self.log(f"wrote {path} ({audio.shape[-1] / sr:.2f} s, {sr} Hz, {audio.shape[0]} ch) in {elapsed:.1f} s")
        return GenerationResult(Path(path), mode, sr, int(audio.shape[0]), audio.shape[-1] / sr, elapsed, info)

    def tts(self, text: str, *, voice: str = C.DEFAULT_VOICE, speed: float = 1.0, lang: Optional[str] = None,
            seed: Optional[int] = None, output: Optional[str] = None, post: Optional[PostOptions] = None,
            **job_kwargs: Any) -> GenerationResult:
        job = SpeechJob(text=text, voice=voice, speed=speed, lang=lang, seed=seed, **job_kwargs).validate()
        return self._render("tts", job, output, post)

    @staticmethod
    def _build_sound_job(mode: str, prompt: str, enrich: bool, enrich_options: Optional[Dict[str, Any]],
                         **job_kwargs: Any):
        """Enrich the prompt (if asked) and build a validated job. Loads nothing."""
        enrichment = P.enrich(mode, prompt, **(enrich_options or {})) if enrich else None
        try:
            job = SoundJob(prompt=enrichment.prompt if enrichment else prompt, mode=mode, **job_kwargs)
        except TypeError as exc:
            raise RequestError(f"unsupported option for a {mode} job: {exc}") from exc
        return job.validate(), enrichment

    def _sound(self, mode: str, prompt: str, *, enrich: bool, enrich_options: Optional[Dict[str, Any]],
               output: Optional[str], post: Optional[PostOptions], **job_kwargs: Any) -> GenerationResult:
        job, enrichment = self._build_sound_job(mode, prompt, enrich, enrich_options, **job_kwargs)
        extra = {}
        if enrichment is not None:
            extra = {"enrichment": enrichment.to_dict()}
            self.log(f"prompt: {enrichment.prompt}")
        return self._render(mode, job, output, post, extra)

    def sfx(self, prompt: str, *, duration: float = 5.0, enrich: bool = True,
            enrich_options: Optional[Dict[str, Any]] = None, output: Optional[str] = None,
            post: Optional[PostOptions] = None, **job_kwargs: Any) -> GenerationResult:
        return self._sound("sfx", prompt, enrich=enrich, enrich_options=enrich_options, output=output,
                           post=post, duration=duration, **job_kwargs)

    def music(self, prompt: str, *, duration: float = 30.0, enrich: bool = True,
              enrich_options: Optional[Dict[str, Any]] = None, output: Optional[str] = None,
              post: Optional[PostOptions] = None, **job_kwargs: Any) -> GenerationResult:
        return self._sound("music", prompt, enrich=enrich, enrich_options=enrich_options, output=output,
                           post=post, duration=duration, **job_kwargs)

    # -------------------------------------------------------------- batching
    def batch(self, specs: Sequence[Dict[str, Any]], post: Optional[PostOptions] = None,
              stop_on_error: bool = False, dry_run: bool = False) -> List[Dict[str, Any]]:
        """Run many jobs in one process, grouped by model to avoid reloads.

        Each spec: ``{"mode": "tts"|"sfx"|"music", "output": "...", <job fields>,
        "enrich": true, "enrich_options": {...}}``. Results come back in the
        original order as ``{"index", "ok", "result" | "error"}`` dicts. With
        ``dry_run=True`` nothing is loaded and each entry carries a ``plan`` instead.
        """
        order = {"tts": 0, "sfx": 1, "music": 2}
        indexed = list(enumerate(specs))
        for _, spec in indexed:
            if spec.get("mode") not in order:
                raise RequestError(f"every batch job needs a mode of tts/sfx/music, got {spec.get('mode')!r}")
        results: Dict[int, Dict[str, Any]] = {}
        for index, spec in sorted(indexed, key=lambda it: (order[it[1]["mode"]], it[0])):
            mode = spec["mode"]
            try:
                fields = {k: v for k, v in spec.items() if k not in ("mode", "output", "enrich", "enrich_options")}
                enrich_options = dict(spec.get("enrich_options") or {})
                if mode != "tts":
                    for key in [k for k in fields if k in ENRICH_KEYS]:
                        enrich_options[key] = fields.pop(key)
                output = spec.get("output")
                if mode == "tts":
                    job = job_from_dict("tts", fields)
                    if dry_run:
                        results[index] = {"index": index, "ok": True, "plan": self.plan("tts", job, None, output, post)}
                        continue
                    res = self._render("tts", job, output, post)
                else:
                    prompt = fields.pop("prompt", None) or fields.pop("text", None)
                    if not prompt:
                        raise RequestError("sfx/music jobs need a 'prompt'")
                    kwargs = {k: v for k, v in fields.items() if k != "mode"}
                    if dry_run:
                        job, enrichment = self._build_sound_job(mode, prompt, bool(spec.get("enrich", True)),
                                                                enrich_options, **kwargs)
                        results[index] = {"index": index, "ok": True,
                                          "plan": self.plan(mode, job, enrichment, output, post)}
                        continue
                    res = self._sound(mode, prompt, enrich=bool(spec.get("enrich", True)),
                                      enrich_options=enrich_options, output=output, post=post, **kwargs)
                results[index] = {"index": index, "ok": True, "result": res.to_dict()}
            except (LocalAudioError, ValueError, TypeError, KeyError) as exc:
                results[index] = {"index": index, "ok": False, "error": str(exc)}
                self.log(f"job {index} failed: {exc}")
                if stop_on_error:
                    break
        return [results[i] for i in sorted(results)]

    # --------------------------------------------------------------- dry run
    def plan(self, mode: str, job: Any, enrichment: Optional[P.EnrichedPrompt] = None,
             output: Optional[str] = None, post: Optional[PostOptions] = None) -> Dict[str, Any]:
        """Describe what *would* run, without loading any model."""
        if mode not in C.MODES:
            raise RequestError(f"unknown mode {mode!r}")
        out: Dict[str, Any] = {"mode": mode, "output": str(output or default_output_path(mode))}
        try:
            out["device"] = self.manager.plan_for(mode).to_dict()
        except DependencyError as exc:
            out["device"] = {"device": "unknown", "reason": str(exc)}
        status = self.paths.status(self.manager.sa3_source)[mode]
        out["models"] = {"dir": str(self.paths.root), **status}
        if mode == "tts":
            out["engine"] = "Kokoro-82M"
            chunks = T.chunk_text(job.text, max_chars=job.max_chars, sentence_pause=job.sentence_pause,
                                  paragraph_pause=job.paragraph_pause)
            lang = job.lang or C.lang_for_voice(job.voice.split(",")[0].split(":")[0])
            try:
                resolve_voice_files(job.voice, self.paths.voices_dir, self.paths.installed_voices())
                voice_ok = True
            except RequestError as exc:
                voice_ok = str(exc)
            out["speech"] = {
                "voice": job.voice, "voice_available": voice_ok, "lang": lang, "speed": job.speed,
                "chunks": len(chunks), "characters": len(job.text),
                "preview": [c.text[:80] for c in chunks[:3]],
                "sample_rate": C.KOKORO_SAMPLE_RATE,
            }
        else:
            variant = C.MODE_TO_VARIANT[mode]
            out["engine"] = f"Stable Audio 3 {variant}"
            clips = 1
            if job.needs_extension:
                ctx = int(job.context_seconds * C.SA3_SAMPLE_RATE) // C.SA3_LATENT_HOP * C.SA3_LATENT_HOP
                step_new = C.SA3_MAX_SECONDS - ctx / C.SA3_SAMPLE_RATE
                if (job.extend_method if job.extend_method != "auto" else "continue") == "continue":
                    clips = 1 + math.ceil((job.duration - C.SA3_MAX_SECONDS) / step_new)
                out["extension"] = {
                    "method": job.extend_method if job.extend_method != "auto" else "continue",
                    "model_calls": clips,
                    "context_seconds": job.context_seconds,
                    "crossfade_seconds": job.crossfade_seconds,
                }
            out["sound"] = {
                "prompt": job.prompt, "duration": job.duration, "steps": job.steps, "cfg_scale": job.cfg_scale,
                "sampler": C.SA3_SAMPLER, "seed": job.seed if job.seed is not None else "random",
                "loopable": job.loopable, "sample_rate": C.SA3_SAMPLE_RATE, "channels": C.SA3_CHANNELS,
            }
            if enrichment is not None:
                out["enrichment"] = enrichment.to_dict()
        return out


def read_batch_file(path: str) -> List[Dict[str, Any]]:
    """Load a batch file: a JSON list, ``{"jobs": [...]}`` or JSON Lines."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        raise RequestError(f"batch file {path} is empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        try:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]
        except json.JSONDecodeError as exc:
            raise RequestError(f"batch file {path} is neither JSON nor JSON Lines: {exc}") from exc
    if isinstance(data, dict):
        data = data.get("jobs", [])
    if not isinstance(data, list) or not all(isinstance(d, dict) for d in data):
        raise RequestError("batch file must contain a list of job objects")
    return data
