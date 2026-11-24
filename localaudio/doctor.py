"""``--doctor``: is this machine ready to generate audio?"""

from __future__ import annotations

import importlib.metadata as md
import importlib.util
import json
import shutil
import sys
from typing import Any, Dict, List, Optional

from . import config as C
from .device import describe_hardware
from .models import ModelPaths

REQUIRED_PACKAGES = (
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("transformers", "transformers"),
    ("safetensors", "safetensors"),
    ("sentencepiece", "sentencepiece"),
    ("soundfile", "soundfile"),
    ("numpy", "numpy"),
    ("huggingface-hub", "huggingface_hub"),
    ("einops", "einops"),
    ("einops-exts", "einops_exts"),
    ("kokoro", "kokoro"),
    ("misaki", "misaki"),
    ("stable-audio-3", "stable_audio_3"),
)


def _check(name: str, status: str, detail: str, hint: str = "") -> Dict[str, str]:
    return {"name": name, "status": status, "detail": detail, "hint": hint}


def collect(models_dir: Optional[str] = None, sa3_source: str = "auto") -> Dict[str, Any]:
    checks: List[Dict[str, str]] = []
    v = sys.version_info
    if (3, 10) <= (v.major, v.minor) < (3, 13):
        checks.append(_check("python", "ok", f"{v.major}.{v.minor}.{v.micro}"))
    else:
        checks.append(_check("python", "fail", f"{v.major}.{v.minor}.{v.micro}",
                             "Use Python 3.10, 3.11 or 3.12 (Kokoro's text front-end does not support 3.13+ yet)."))

    for dist, module in REQUIRED_PACKAGES:
        try:
            version = md.version(dist)
            checks.append(_check(dist, "ok", version))
        except md.PackageNotFoundError:
            spec = importlib.util.find_spec(module)
            if spec is None:
                checks.append(_check(dist, "fail", "not installed", "Run ./install.sh (or .\\install.ps1)."))
            else:
                checks.append(_check(dist, "ok", "installed (version unknown)"))

    if importlib.util.find_spec("en_core_web_sm") is None:
        checks.append(_check("spacy model en_core_web_sm", "warn", "missing",
                             "python scripts/setup_models.py --only tts   (needed for English speech)"))
    else:
        checks.append(_check("spacy model en_core_web_sm", "ok", "installed"))
    if importlib.util.find_spec("espeakng_loader") is None:
        checks.append(_check("espeak-ng", "warn", "espeakng-loader not installed",
                             "Needed for Hindi/Spanish/French/Italian/Portuguese speech (pip install espeakng-loader)."))
    else:
        checks.append(_check("espeak-ng", "ok", "bundled via espeakng-loader"))

    hardware = describe_hardware()
    cuda = hardware.get("cuda")
    if cuda:
        status = "ok" if cuda["vram_total_gb"] >= C.MIN_VRAM_GB_SA3 else "warn"
        hint = "" if status == "ok" else f"< {C.MIN_VRAM_GB_SA3} GiB: SFX/music will run on the CPU in --device auto."
        checks.append(_check("gpu", status, f"{cuda['name']}, {cuda['vram_total_gb']} GiB "
                                             f"({cuda['vram_free_gb']} GiB free)", hint))
    else:
        checks.append(_check("gpu", "warn", "no CUDA GPU detected", "CPU mode works; long music renders slower."))
    ram = hardware.get("ram_gb")
    if ram is not None:
        checks.append(_check("ram", "ok" if ram >= 8 else "warn", f"{ram} GiB",
                             "" if ram >= 8 else "8 GiB or more is recommended for Stable Audio 3."))

    paths = ModelPaths(models_dir)
    status = paths.status(sa3_source)
    for mode in C.MODES:
        entry = status[mode]
        if entry["ready"]:
            extra = f" ({entry['source']} files)" if entry.get("source") else ""
            checks.append(_check(f"models:{mode}", "ok", "ready" + extra))
        else:
            checks.append(_check(f"models:{mode}", "warn", f"missing {len(entry['missing'])} file(s)",
                                 f"python scripts/setup_models.py --only {mode}"))
    probe = paths.root
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free / 1024**3
        checks.append(_check("disk", "ok" if free >= 8 else "warn", f"{free:.1f} GiB free near {paths.root}",
                             "" if free >= 8 else "All models need about 6 GiB."))
    except OSError:
        pass
    ok = not any(c["status"] == "fail" for c in checks)
    return {"ok": ok, "models_dir": str(paths.root), "hardware": hardware, "checks": checks}


def run_doctor(models_dir: Optional[str] = None, as_json: bool = False, sa3_source: str = "auto") -> int:
    report = collect(models_dir, sa3_source)
    if as_json:
        print(json.dumps(report, indent=2))
    else:
        icon = {"ok": "[ ok ]", "warn": "[warn]", "fail": "[FAIL]"}
        print(f"local-audio-studio doctor  (models: {report['models_dir']})\n")
        for c in report["checks"]:
            print(f"{icon[c['status']]} {c['name']:<28} {c['detail']}")
            if c["hint"]:
                print(f"       -> {c['hint']}")
        print("\n" + ("Ready." if report["ok"] else "Some required components are missing."))
    return 0 if report["ok"] else 1
