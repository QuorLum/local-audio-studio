"""Static configuration: pinned revisions, model registry, limits.

Nothing in here touches the network or imports heavy dependencies.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
ASSETS_DIR = PACKAGE_DIR / "assets"

ENV_MODELS_DIR = "LOCALAUDIO_MODELS_DIR"
ENV_MIN_VRAM = "LOCALAUDIO_MIN_VRAM_GB"

# --------------------------------------------------------------------------- #
# Device policy
# --------------------------------------------------------------------------- #
# Spec: use CUDA + fp16 when >= 3.5 GB VRAM is available, otherwise CPU.
MIN_VRAM_GB_SA3 = 3.5
# Kokoro is only 82M parameters; it runs happily on much smaller GPUs.
MIN_VRAM_GB_KOKORO = 1.5
# On top of the capacity test we also need this much *free* VRAM for SA3 small
# (upstream reports ~2.4 GB peak for a 120 s clip). Anything less -> CPU.
MIN_FREE_VRAM_GB_SA3 = 2.6

# --------------------------------------------------------------------------- #
# Kokoro-82M (text-to-speech)  -- Apache-2.0
# --------------------------------------------------------------------------- #
KOKORO_REPO = "hexgrad/Kokoro-82M"
# Pinned so setup is reproducible (this is the commit the docs of several
# deployments pin; the repository has been static since early 2025).
KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
KOKORO_SAMPLE_RATE = 24_000
KOKORO_CORE_FILES: Tuple[str, ...] = ("config.json", "kokoro-v1_0.pth")
KOKORO_VOICES_GLOB = "voices/*.pt"
DEFAULT_VOICE = "af_heart"
KOKORO_SPEED_RANGE = (0.5, 2.0)
# Kokoro language codes are the first letter of every voice name.
KOKORO_LANGS: Dict[str, str] = {
    "a": "American English",
    "b": "British English",
    "e": "Spanish",
    "f": "French",
    "h": "Hindi",
    "i": "Italian",
    "j": "Japanese",
    "p": "Brazilian Portuguese",
    "z": "Mandarin Chinese",
}
# Catalogue of the 54 voices shipped with Kokoro v1.0 (used for --list-voices
# when the weights are not downloaded yet, and for validation hints).
KOKORO_VOICES: Tuple[str, ...] = (
    # American English
    "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    # British English
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    # Japanese
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro", "jm_kumo",
    # Mandarin Chinese
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang",
    # Spanish / French / Hindi / Italian / Brazilian Portuguese
    "ef_dora", "em_alex", "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta", "hm_omega", "hm_psi",
    "if_sara", "im_nicola",
    "pf_dora", "pm_alex", "pm_santa",
)

# --------------------------------------------------------------------------- #
# Stable Audio 3 (sound effects + music)
# --------------------------------------------------------------------------- #
# Upstream inference code (MIT): https://github.com/Stability-AI/stable-audio-3
SA3_COMMIT = "790793b89ddceea612be9ce42ff6830d18f6fa57"
SA3_GIT_URL = f"git+https://github.com/Stability-AI/stable-audio-3@{SA3_COMMIT}"

# Un-gated ComfyUI repackaging of the official weights (checkpoints are the
# same fp32 `model.safetensors` layout Stability publishes).
COMFY_REPO = "Comfy-Org/stable-audio-3"
T5_COMFY_FILE = "text_encoders/t5gemma_b_b_ul2.safetensors"
T5_CONFIG_ASSET = "t5gemma_b_b_ul2_config.json"

# SentencePiece model of the Gemma tokenizer used by T5Gemma. Stability's own
# optimized runtimes tokenize with exactly this file (no BOS/EOS, pad id 0,
# truncate to 256 tokens). Pinned to the commit above and verified by hash.
TOKENIZER_URL = (
    "https://raw.githubusercontent.com/Stability-AI/stable-audio-3/"
    f"{SA3_COMMIT}/optimized/tflite/models/tokenizer.model"
)
TOKENIZER_SHA256 = "61a7b147390c64585d6c3543dd6fc636906c9af3865a5548f27f31aee1d4c8e2"
TOKENIZER_SIZE = 4_241_003
T5_MAX_TOKENS = 256

SA3_SAMPLE_RATE = 44_100
SA3_CHANNELS = 2
SA3_MAX_SECONDS = 120.0  # hard limit of a single Small clip
SA3_MIN_SECONDS = 0.5
SA3_LATENT_HOP = 4096  # audio samples per latent frame
SA3_DEFAULT_STEPS = 8
SA3_MAX_STEPS = 12
SA3_DEFAULT_CFG = 1.0
SA3_SAMPLER = "pingpong"


@dataclass(frozen=True)
class Sa3Variant:
    key: str  # directory / registry key
    mode: str  # CLI mode that routes here
    comfy_file: str  # path inside COMFY_REPO
    official_repo: str  # gated Stability repo (optional source)
    config_asset: str  # vendored inference config
    blurb: str


SA3_VARIANTS: Dict[str, Sa3Variant] = {
    "small-sfx": Sa3Variant(
        key="small-sfx",
        mode="sfx",
        comfy_file="checkpoints/stable_audio_3_small_sfx.safetensors",
        official_repo="stabilityai/stable-audio-3-small-sfx",
        config_asset="sa3_small_sfx_config.json",
        blurb="foley, impacts, UI/sci-fi sounds and ambience beds (up to 120 s)",
    ),
    "small-music": Sa3Variant(
        key="small-music",
        mode="music",
        comfy_file="checkpoints/stable_audio_3_small_music.safetensors",
        official_repo="stabilityai/stable-audio-3-small-music",
        config_asset="sa3_small_music_config.json",
        blurb="instrumental background music, scores, loops, stems (up to 120 s)",
    ),
}
MODE_TO_VARIANT: Dict[str, str] = {v.mode: k for k, v in SA3_VARIANTS.items()}

MODES = ("tts", "sfx", "music")


# --------------------------------------------------------------------------- #
# Where models live
# --------------------------------------------------------------------------- #
def _is_source_checkout() -> bool:
    return (PROJECT_ROOT / "SKILL.md").exists() or (PROJECT_ROOT / "pyproject.toml").exists()


def _user_data_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "local-audio-studio"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "local-audio-studio"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "local-audio-studio"


def default_models_dir(override: Optional[str] = None) -> Path:
    """Resolve the models directory.

    Priority: explicit argument > $LOCALAUDIO_MODELS_DIR > ``<repo>/models`` when
    running from a source checkout > per-user data directory.
    """
    if override:
        return Path(override).expanduser().resolve()
    env = os.environ.get(ENV_MODELS_DIR)
    if env:
        return Path(env).expanduser().resolve()
    if _is_source_checkout():
        return (PROJECT_ROOT / "models").resolve()
    return _user_data_dir() / "models"


def min_vram_gb(default: float) -> float:
    raw = os.environ.get(ENV_MIN_VRAM)
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return default


def lang_for_voice(voice_name: str) -> str:
    """Kokoro language code for a voice name such as ``af_heart`` -> ``a``."""
    name = Path(voice_name).stem
    return name[:1].lower() if name else "a"
