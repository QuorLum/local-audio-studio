"""Kokoro-82M text-to-speech engine.

Weights are read from ``<models>/kokoro`` (downloaded by ``setup_models.py``);
nothing is fetched at run time. The model is only 82M parameters, so it runs in
fp32 on both CPU and GPU.
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import config as C
from .. import textchunk as T
from ..audio import assemble_speech
from ..device import import_torch
from ..errors import DependencyError, ModelsMissingError, RequestError
from ..jobs import SpeechJob
from .base import AudioResult, Engine
from .sa3_loader import quiet_stdout

_VOICE_NAME = re.compile(r"^[A-Za-z0-9_\-]+$")


def parse_voice_spec(spec: str) -> List[Tuple[str, float]]:
    """``"af_heart"`` or a blend such as ``"af_heart:0.7,af_bella:0.3"``.

    Returns ``[(name, weight), ...]``. An explicit ``*.pt`` file path is passed
    through unchanged with weight 1.0.
    """
    spec = (spec or "").strip()
    if not spec:
        raise RequestError("voice is empty")
    if spec.lower().endswith(".pt"):
        return [(spec, 1.0)]
    items: List[Tuple[str, float]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, weight = part.partition(":")
        name = name.strip()
        if not _VOICE_NAME.match(name):
            raise RequestError(f"invalid voice name {name!r}")
        try:
            w = float(weight) if weight else 1.0
        except ValueError as exc:
            raise RequestError(f"invalid blend weight in {part!r}") from exc
        if w <= 0:
            raise RequestError("blend weights must be > 0")
        items.append((name, w))
    if not items:
        raise RequestError("voice is empty")
    return items


def resolve_voice_files(spec: str, voices_dir: Path, installed: Optional[List[str]] = None) -> List[Tuple[Path, float]]:
    """Map a voice spec to files on disk; helpful error with close matches."""
    resolved: List[Tuple[Path, float]] = []
    for name, weight in parse_voice_spec(spec):
        path = Path(name) if name.lower().endswith(".pt") else voices_dir / f"{name}.pt"
        if not path.is_file():
            pool = installed if installed else list(C.KOKORO_VOICES)
            close = difflib.get_close_matches(Path(name).stem, pool, n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            raise RequestError(f"voice {name!r} not found in {voices_dir}.{hint} (see --list-voices)")
        resolved.append((path, weight))
    return resolved


class KokoroEngine(Engine):
    key = "tts"
    display_name = "Kokoro-82M"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._model: Any = None
        self._pipelines: Dict[str, Any] = {}
        self._blend_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self._loaded:
            return
        missing = self.paths.missing("tts")
        if missing:
            raise ModelsMissingError(
                "Kokoro files are missing:\n  " + "\n  ".join(missing) +
                "\nRun: python scripts/setup_models.py --only tts"
            )
        import_torch()  # fail early with a friendly message if PyTorch is missing
        try:
            with quiet_stdout():
                from kokoro import KModel
        except Exception as exc:
            raise DependencyError(
                f"The `kokoro` package could not be imported [{type(exc).__name__}: {exc}]. "
                "Run ./install.sh (or .\\install.ps1)."
            ) from exc
        self.log(f"loading Kokoro-82M on {self.plan.short()} ...")
        with quiet_stdout():
            model = KModel(
                repo_id=C.KOKORO_REPO,
                config=str(self.paths.kokoro_config),
                model=str(self.paths.kokoro_weights),
            )
            self._model = model.to(self.plan.device).eval()
        self._loaded = True

    def unload(self) -> None:
        self._pipelines.clear()
        self._blend_cache.clear()
        self._model = None
        self._loaded = False

    # -------------------------------------------------------------- helpers
    def _ensure_g2p_assets(self, lang: str) -> None:
        if lang in ("a", "b"):
            try:
                import spacy.util

                if not spacy.util.is_package("en_core_web_sm"):
                    raise DependencyError(
                        "The spaCy model `en_core_web_sm` is not installed (Kokoro's English text front-end "
                        "needs it). Run: python scripts/setup_models.py --only tts   "
                        "(or: python -m spacy download en_core_web_sm)"
                    )
            except ImportError as exc:
                raise DependencyError(f"spaCy is required for English TTS: {exc}") from exc

    def _pipeline(self, lang: str) -> Any:
        if lang not in C.KOKORO_LANGS:
            raise RequestError(f"unknown language code {lang!r}; choose one of {', '.join(sorted(C.KOKORO_LANGS))}")
        if lang not in self._pipelines:
            self._ensure_g2p_assets(lang)
            try:
                with quiet_stdout():
                    from kokoro import KPipeline

                    self._pipelines[lang] = KPipeline(lang_code=lang, repo_id=C.KOKORO_REPO, model=self._model)
            except (ImportError, ModuleNotFoundError) as exc:
                extra = {"j": 'pip install "misaki[ja]"', "z": 'pip install "misaki[zh]"'}.get(lang, "")
                raise DependencyError(
                    f"Language {lang!r} ({C.KOKORO_LANGS[lang]}) needs extra packages: {exc}. {extra}".strip()
                ) from exc
        return self._pipelines[lang]

    def _voice_argument(self, files: List[Tuple[Path, float]]) -> Any:
        """A path string for one voice, or a CPU float tensor for a weighted blend."""
        if len(files) == 1:
            return str(files[0][0])
        key = ",".join(f"{p}:{w}" for p, w in files)
        if key not in self._blend_cache:
            torch = import_torch()
            packs = [torch.load(str(p), map_location="cpu", weights_only=True).float() for p, _ in files]
            weights = torch.tensor([w for _, w in files], dtype=torch.float32)
            weights = weights / weights.sum()
            blended = sum(w * pack for w, pack in zip(weights, packs))
            self._blend_cache[key] = blended.float().cpu()  # must be torch.FloatTensor for KPipeline
        return self._blend_cache[key]

    # ------------------------------------------------------------- generate
    def generate(self, job: SpeechJob) -> AudioResult:
        job.validate()
        if not self._loaded:
            self.load()
        torch = import_torch()
        files = resolve_voice_files(job.voice, self.paths.voices_dir, self.paths.installed_voices())
        lang = (job.lang or C.lang_for_voice(files[0][0].name)).lower()
        pipeline = self._pipeline(lang)
        voice_arg = self._voice_argument(files)
        chunks = T.chunk_text(
            job.text,
            max_chars=job.max_chars,
            sentence_pause=job.sentence_pause,
            paragraph_pause=job.paragraph_pause,
        )
        if not chunks:
            raise RequestError("nothing to speak after cleaning the text")
        if job.seed is not None:
            torch.manual_seed(int(job.seed))

        segments: List[Tuple[np.ndarray, float]] = []
        sr = C.KOKORO_SAMPLE_RATE
        with quiet_stdout():
            for index, chunk in enumerate(chunks, 1):
                if len(chunks) > 1:
                    self.log(f"  speaking chunk {index}/{len(chunks)} ({len(chunk.text)} chars)")
                pieces: List[np.ndarray] = []
                for result in pipeline(chunk.text, voice=voice_arg, speed=float(job.speed), split_pattern=None):
                    audio = getattr(result, "audio", None)
                    if audio is None:
                        continue
                    pieces.append(audio.detach().float().cpu().numpy().reshape(-1))
                if not pieces:
                    self.log(f"  warning: chunk {index} produced no audio: {chunk.text[:60]!r}")
                    continue
                segments.append((np.concatenate(pieces)[None, :].astype(np.float32), chunk.pause_after))
        if not segments:
            raise RequestError("the text produced no speakable audio (only symbols?)")
        audio = assemble_speech(segments, sr, channels=1)
        info = {
            "engine": self.display_name,
            "voice": job.voice,
            "lang": lang,
            "speed": job.speed,
            "chunks": len(chunks),
            "seed": job.seed,
            "device": self.plan.to_dict(),
        }
        return AudioResult(audio, sr, info)
