"""Where model files live on disk and how to tell whether they are complete."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from . import config as C


@dataclass(frozen=True)
class Sa3Assets:
    """Everything the Stable Audio 3 loader needs, resolved to local paths."""

    variant: str
    source: str  # "comfy" (un-gated repack) or "official" (gated Stability bundle)
    checkpoint: Path
    config_json: Path
    # official layout: a Hugging Face style folder holding the T5Gemma encoder
    t5_dir: Optional[Path] = None
    # comfy layout: single-file encoder weights + vendored config + tokenizer
    t5_weights: Optional[Path] = None
    t5_config: Optional[Path] = None
    tokenizer_model: Optional[Path] = None


class ModelPaths:
    """Resolves every model path below one root directory (``./models`` by default)."""

    def __init__(self, root: Path | str | None = None):
        self.root = Path(root).expanduser().resolve() if root else C.default_models_dir()

    # -- Kokoro -------------------------------------------------------------
    @property
    def kokoro_dir(self) -> Path:
        return self.root / "kokoro"

    @property
    def kokoro_config(self) -> Path:
        return self.kokoro_dir / "config.json"

    @property
    def kokoro_weights(self) -> Path:
        return self.kokoro_dir / "kokoro-v1_0.pth"

    @property
    def voices_dir(self) -> Path:
        return self.kokoro_dir / "voices"

    def voice_path(self, name: str) -> Path:
        return self.voices_dir / f"{name}.pt"

    def installed_voices(self) -> List[str]:
        if not self.voices_dir.is_dir():
            return []
        return sorted(p.stem for p in self.voices_dir.glob("*.pt"))

    # -- Stable Audio 3: un-gated ComfyUI repack ------------------------------
    @property
    def comfy_dir(self) -> Path:
        return self.root / "comfy"

    def comfy_checkpoint(self, variant: str) -> Path:
        return self.comfy_dir / C.SA3_VARIANTS[variant].comfy_file

    @property
    def comfy_t5_weights(self) -> Path:
        return self.comfy_dir / C.T5_COMFY_FILE

    @property
    def tokenizer_model(self) -> Path:
        return self.comfy_dir / "tokenizer" / "tokenizer.model"

    # -- Stable Audio 3: official gated bundle (optional) -------------------------
    def official_dir(self, variant: str) -> Path:
        return self.root / "official" / variant

    # -- resolution ------------------------------------------------------------
    def _official_assets(self, variant: str) -> Optional[Sa3Assets]:
        d = self.official_dir(variant)
        cfg, ckpt, t5 = d / "model_config.json", d / "model.safetensors", d / "t5gemma-b-b-ul2"
        has_t5 = t5.is_dir() and (t5 / "config.json").is_file()
        if cfg.is_file() and ckpt.is_file() and has_t5:
            return Sa3Assets(variant, "official", ckpt, cfg, t5_dir=t5)
        return None

    def _comfy_assets(self, variant: str) -> Optional[Sa3Assets]:
        spec = C.SA3_VARIANTS[variant]
        ckpt = self.comfy_checkpoint(variant)
        cfg = C.ASSETS_DIR / spec.config_asset
        t5cfg = C.ASSETS_DIR / C.T5_CONFIG_ASSET
        if (
            ckpt.is_file()
            and self.comfy_t5_weights.is_file()
            and self.tokenizer_model.is_file()
            and cfg.is_file()
            and t5cfg.is_file()
        ):
            return Sa3Assets(
                variant,
                "comfy",
                ckpt,
                cfg,
                t5_weights=self.comfy_t5_weights,
                t5_config=t5cfg,
                tokenizer_model=self.tokenizer_model,
            )
        return None

    def sa3_assets(self, variant: str, source: str = "auto") -> Optional[Sa3Assets]:
        """Return complete assets for ``variant`` or ``None`` if not installed.

        ``source``: ``auto`` prefers the official bundle when present (it is
        the reference layout), then falls back to the un-gated ComfyUI files.
        """
        if variant not in C.SA3_VARIANTS:
            raise KeyError(variant)
        if source in ("auto", "official"):
            found = self._official_assets(variant)
            if found or source == "official":
                return found
        return self._comfy_assets(variant)

    # -- reporting -------------------------------------------------------------
    def missing(self, mode: str, source: str = "auto") -> List[str]:
        """Human readable list of what still has to be downloaded for ``mode``."""
        out: List[str] = []
        if mode == "tts":
            for p in (self.kokoro_config, self.kokoro_weights):
                if not p.is_file():
                    out.append(str(p))
            if not self.installed_voices():
                out.append(f"{self.voices_dir}/*.pt")
            return out
        variant = C.MODE_TO_VARIANT[mode]
        if self.sa3_assets(variant, source):
            return out
        if source == "official":
            out.append(str(self.official_dir(variant)))
            return out
        for p in (self.comfy_checkpoint(variant), self.comfy_t5_weights, self.tokenizer_model):
            if not p.is_file():
                out.append(str(p))
        return out

    def status(self, source: str = "auto") -> Dict[str, dict]:
        report: Dict[str, dict] = {}
        for mode in C.MODES:
            miss = self.missing(mode, source)
            entry: dict = {"ready": not miss, "missing": miss}
            if mode != "tts":
                assets = self.sa3_assets(C.MODE_TO_VARIANT[mode], source)
                entry["source"] = assets.source if assets else None
            report[mode] = entry
        return report
