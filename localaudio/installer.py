"""Model downloader: fetch only the weights you need into ``./models``.

Default sources need **no Hugging Face token**:

* Kokoro-82M ......... ``hexgrad/Kokoro-82M`` (Apache-2.0, public)
* Stable Audio 3 ..... ``Comfy-Org/stable-audio-3`` (public ComfyUI repackaging)
* T5Gemma tokenizer .. one 4 MB file from Stability's GitHub repo, hash-checked

``--source official`` instead pulls Stability's own bundles
(``stabilityai/stable-audio-3-small-*``), which are *gated*: accept the license
on the model page and pass a token (``--hf-token`` or ``hf auth login``).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import config as C
from .errors import DependencyError, LocalAudioError, ModelFormatError, RequestError
from .models import ModelPaths

MB = 1000**2
GB = 1000**3

# Approximate sizes for planning/disk checks (exact numbers come from the Hub when online).
STATIC_SIZES: Dict[str, int] = {
    "kokoro": 363 * MB,
    "small-sfx": int(2.27 * GB),
    "small-music": int(2.27 * GB),
    "t5gemma": int(1.2 * GB),
    "tokenizer": C.TOKENIZER_SIZE,
    "official-small-sfx": int(3.5 * GB),
    "official-small-music": int(3.5 * GB),
}

GROUPS = ("tts", "sfx", "music")


def human(n: float) -> str:
    if n < 1000:
        return f"{n:.0f} B"
    if n < MB:
        return f"{n / 1000:.0f} kB"
    if n < GB:
        return f"{n / MB:.0f} MB"
    return f"{n / GB:.2f} GB"


def parse_only(values: Optional[Sequence[str]]) -> List[str]:
    """``["all"]`` / ``["tts,sfx"]`` / ``["music"]`` -> ordered list of groups."""
    if not values:
        return list(GROUPS)
    chosen: List[str] = []
    for value in values:
        for part in str(value).replace(" ", ",").split(","):
            part = part.strip().lower()
            if not part:
                continue
            if part == "all":
                return list(GROUPS)
            aliases = {"speech": "tts", "voice": "tts", "kokoro": "tts", "effects": "sfx", "foley": "sfx",
                       "ambience": "sfx", "score": "music"}
            part = aliases.get(part, part)
            if part not in GROUPS:
                raise RequestError(f"unknown component {part!r}; choose from: all, {', '.join(GROUPS)}")
            if part not in chosen:
                chosen.append(part)
    return chosen or list(GROUPS)


# --------------------------------------------------------------------------- #
# Primitive downloads
# --------------------------------------------------------------------------- #
def _hub():
    try:
        import huggingface_hub

        return huggingface_hub
    except Exception as exc:
        raise DependencyError(f"huggingface_hub is required for downloads: {exc}") from exc


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def download_url(url: str, dest: Path, sha256: Optional[str] = None, log: Callable[[str], None] = print) -> Path:
    """Download ``url`` atomically and verify its SHA-256 (if given)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "local-audio-studio"})
        with urllib.request.urlopen(request, timeout=60) as response, open(tmp, "wb") as out:
            shutil.copyfileobj(response, out)
        if sha256 and sha256_file(tmp) != sha256:
            raise ModelFormatError(f"checksum mismatch for {url} (expected {sha256})")
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    log(f"  downloaded {dest.name} ({human(dest.stat().st_size)})")
    return dest


def _gated_hint(repo: str) -> str:
    return (
        f"Access to {repo} is gated. Open https://huggingface.co/{repo}, accept the license, then pass a "
        "token: --hf-token hf_xxx (or run `hf auth login`). The default --source comfy needs no token."
    )


def _translate_hub_errors(exc: Exception, repo: str) -> LocalAudioError:
    name = type(exc).__name__
    text = str(exc)
    if "Gated" in name or "401" in text or "403" in text:
        return LocalAudioError(_gated_hint(repo))
    if "RepositoryNotFound" in name or "404" in text:
        return LocalAudioError(f"repository or file not found on the Hub: {repo} ({text[:200]})")
    return LocalAudioError(f"download from {repo} failed: {name}: {text[:300]}")


def hub_download(repo: str, filename: str, dest_dir: Path, revision: Optional[str] = None,
                 token: Optional[str] = None) -> Path:
    hub = _hub()
    try:
        path = hub.hf_hub_download(repo_id=repo, filename=filename, revision=revision,
                                   local_dir=str(dest_dir), token=token)
    except Exception as exc:  # noqa: BLE001 - translated for the user
        raise _translate_hub_errors(exc, repo) from exc
    return Path(path)


def hub_snapshot(repo: str, dest_dir: Path, revision: Optional[str] = None, token: Optional[str] = None,
                 allow_patterns: Optional[List[str]] = None) -> Path:
    hub = _hub()
    try:
        path = hub.snapshot_download(repo_id=repo, revision=revision, local_dir=str(dest_dir), token=token,
                                     allow_patterns=allow_patterns)
    except Exception as exc:  # noqa: BLE001
        raise _translate_hub_errors(exc, repo) from exc
    return Path(path)


def remote_sizes(repo: str, files: Sequence[str], revision: Optional[str] = None,
                 token: Optional[str] = None) -> Dict[str, Optional[int]]:
    """Exact file sizes from the Hub (best effort; ``None`` when offline)."""
    try:
        hub = _hub()
        infos = hub.HfApi().get_paths_info(repo, list(files), revision=revision, token=token)
        return {i.path: getattr(i, "size", None) for i in infos}
    except Exception:
        return {f: None for f in files}


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def ensure_spacy_model(log: Callable[[str], None] = print) -> bool:
    """Kokoro's English front-end needs ``en_core_web_sm``; install it once, here."""
    try:
        import spacy.util

        if spacy.util.is_package("en_core_web_sm"):
            return True
    except ImportError:
        log("  spaCy is not installed; run ./install.sh first")
        return False
    log("  installing spaCy model en_core_web_sm (one-time, ~13 MB) ...")
    result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    if result.returncode != 0:
        log("  could not install en_core_web_sm automatically; run: python -m spacy download en_core_web_sm")
        return False
    return True


def install_tts(paths: ModelPaths, force: bool, token: Optional[str], log: Callable[[str], None],
                spacy_model: bool = True) -> None:
    log(f"[tts] Kokoro-82M from {C.KOKORO_REPO} (Apache-2.0)")
    if not force and not paths.missing("tts"):
        log("  already installed")
    else:
        hub_snapshot(
            C.KOKORO_REPO, paths.kokoro_dir, revision=C.KOKORO_REVISION, token=token,
            allow_patterns=list(C.KOKORO_CORE_FILES) + [C.KOKORO_VOICES_GLOB],
        )
        log(f"  {len(paths.installed_voices())} voices installed")
    if spacy_model:
        ensure_spacy_model(log)


def install_shared_sa3(paths: ModelPaths, force: bool, revision: Optional[str], token: Optional[str],
                       log: Callable[[str], None]) -> None:
    """T5Gemma text encoder + tokenizer, shared by the SFX and music models."""
    if force or not paths.comfy_t5_weights.is_file():
        log(f"[sa3] T5Gemma-b-b-ul2 text encoder from {C.COMFY_REPO}")
        hub_download(C.COMFY_REPO, C.T5_COMFY_FILE, paths.comfy_dir, revision, token)
    if force or not paths.tokenizer_model.is_file():
        log("[sa3] SentencePiece tokenizer from Stability AI's GitHub repo (pinned commit, hash-checked)")
        download_url(C.TOKENIZER_URL, paths.tokenizer_model, C.TOKENIZER_SHA256, log)


def install_sa3_comfy(paths: ModelPaths, variant: str, force: bool, revision: Optional[str],
                      token: Optional[str], log: Callable[[str], None]) -> None:
    spec = C.SA3_VARIANTS[variant]
    target = paths.comfy_checkpoint(variant)
    if force or not target.is_file():
        log(f"[{spec.mode}] Stable Audio 3 {variant} checkpoint from {C.COMFY_REPO} (~2.3 GB)")
        hub_download(C.COMFY_REPO, spec.comfy_file, paths.comfy_dir, revision, token)
    else:
        log(f"[{spec.mode}] {variant} checkpoint already installed")
    install_shared_sa3(paths, force, revision, token, log)


def install_sa3_official(paths: ModelPaths, variant: str, force: bool, token: Optional[str],
                         log: Callable[[str], None]) -> None:
    spec = C.SA3_VARIANTS[variant]
    dest = paths.official_dir(variant)
    if not force and paths.sa3_assets(variant, "official"):
        log(f"[{spec.mode}] official bundle already installed")
        return
    log(f"[{spec.mode}] official bundle {spec.official_repo} (gated; ~3.5 GB)")
    hub_snapshot(spec.official_repo, dest, token=token)


def verify_sa3(paths: ModelPaths, variant: str, source: str, log: Callable[[str], None]) -> bool:
    """Header-only architecture check of the downloaded checkpoint (no weights are read)."""
    assets = paths.sa3_assets(variant, source)
    if assets is None:
        log(f"  cannot verify {variant}: files missing")
        return False
    try:
        from .engines.sa3_loader import verify_assets

        report = verify_assets(assets)
    except DependencyError as exc:
        log(f"  skipped architecture check ({exc})")
        return True
    except ModelFormatError as exc:
        log(f"  VERIFY FAILED for {variant}: {exc}")
        return False
    except Exception as exc:  # noqa: BLE001
        log(f"  architecture check could not run ({type(exc).__name__}: {exc})")
        return True
    log(f"  verified {variant}: {report.summary()}")
    return True


# --------------------------------------------------------------------------- #
# Planning / reporting
# --------------------------------------------------------------------------- #
@dataclass
class PlanRow:
    group: str
    what: str
    size: int
    installed: bool


def build_plan(groups: Sequence[str], source: str, paths: ModelPaths) -> List[PlanRow]:
    rows: List[PlanRow] = []
    if "tts" in groups:
        rows.append(PlanRow("tts", f"Kokoro-82M + voices ({C.KOKORO_REPO})", STATIC_SIZES["kokoro"],
                            not paths.missing("tts")))
    needs_shared = False
    for group in ("sfx", "music"):
        if group not in groups:
            continue
        variant = C.MODE_TO_VARIANT[group]
        if source == "official":
            rows.append(PlanRow(group, f"{C.SA3_VARIANTS[variant].official_repo} (gated bundle)",
                                STATIC_SIZES[f"official-{variant}"], bool(paths.sa3_assets(variant, "official"))))
        else:
            needs_shared = True
            rows.append(PlanRow(group, f"Stable Audio 3 {variant} checkpoint ({C.COMFY_REPO})",
                                STATIC_SIZES[variant], paths.comfy_checkpoint(variant).is_file()))
    if needs_shared:
        rows.append(PlanRow("sfx/music", "T5Gemma-b-b-ul2 text encoder (shared)", STATIC_SIZES["t5gemma"],
                            paths.comfy_t5_weights.is_file()))
        rows.append(PlanRow("sfx/music", "SentencePiece tokenizer (shared, GitHub)", STATIC_SIZES["tokenizer"],
                            paths.tokenizer_model.is_file()))
    return rows


def print_plan(rows: Sequence[PlanRow], paths: ModelPaths, log: Callable[[str], None] = print) -> int:
    log(f"models directory: {paths.root}\n")
    log(f"{'component':<10}{'size':>10}  {'status':<14}what")
    to_download = 0
    for r in rows:
        status = "installed" if r.installed else "to download"
        if not r.installed:
            to_download += r.size
        log(f"{r.group:<10}{human(r.size):>10}  {status:<14}{r.what}")
    log(f"\nstill to download: ~{human(to_download)}")
    try:
        probe = paths.root
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        free = shutil.disk_usage(probe).free
        flag = "" if free > to_download + GB else "   <-- not enough free space!"
        log(f"free disk space:   {human(free)}{flag}")
    except OSError:
        pass
    return to_download


def prompt_groups(rows: Sequence[PlanRow], input_fn: Callable[[str], str] = input) -> List[str]:
    """Interactive menu used when the script runs in a terminal without --only."""
    print("\nWhat would you like to download?")
    print("  [a] everything (about 6 GB)")
    print("  [t] speech (Kokoro, ~0.4 GB)")
    print("  [s] sound effects (Stable Audio 3 Small SFX + shared text encoder, ~3.6 GB)")
    print("  [m] music (Stable Audio 3 Small Music + shared text encoder, ~3.6 GB)")
    print("  [q] quit")
    answer = input_fn("choice (e.g. 't,s' or 'a') [a]: ").strip().lower() or "a"
    if answer.startswith("q"):
        return []
    mapping = {"a": "all", "t": "tts", "s": "sfx", "m": "music"}
    parts = [mapping.get(p.strip(), p.strip()) for p in answer.replace(" ", ",").split(",") if p.strip()]
    return parse_only(parts)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="setup_models.py",
        description="Download the model weights for local-audio-studio into ./models (resumable).",
        epilog="No Hugging Face token is needed for the default sources.",
    )
    p.add_argument("--only", action="append", metavar="PART",
                   help="what to download: all | tts | sfx | music (comma separated or repeated). "
                        "Default: ask interactively in a terminal, otherwise all")
    p.add_argument("--models-dir", help="target directory (default ./models or $LOCALAUDIO_MODELS_DIR)")
    p.add_argument("--source", choices=("comfy", "official"), default="comfy",
                   help="comfy = un-gated ComfyUI repackaging (default); official = Stability's gated bundles")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="Hugging Face token (only needed for --source official)")
    p.add_argument("--revision", help="Hub revision for the Comfy-Org repo (default: main)")
    p.add_argument("--force", action="store_true", help="re-download even if files exist")
    p.add_argument("--dry-run", "--list", dest="dry_run", action="store_true",
                   help="show what would be downloaded and the disk space needed")
    p.add_argument("--check", action="store_true", help="verify what is installed and exit")
    p.add_argument("--no-verify", action="store_true", help="skip the post-download architecture check")
    p.add_argument("--no-spacy", action="store_true", help="do not install the spaCy English model")
    p.add_argument("-y", "--yes", action="store_true", help="do not ask questions")
    return p


def check_installation(paths: ModelPaths, source: str = "auto", log: Callable[[str], None] = print,
                       deep: bool = True) -> bool:
    status = paths.status(source)
    ok = True
    log(f"models directory: {paths.root}")
    for mode in GROUPS:
        entry = status[mode]
        if entry["ready"]:
            src = f" ({entry['source']} files)" if entry.get("source") else ""
            log(f"  [ ok ] {mode}{src}")
        else:
            ok = False
            log(f"  [miss] {mode}: " + "; ".join(entry["missing"][:3]))
    if deep:
        for mode in ("sfx", "music"):
            if status[mode]["ready"]:
                ok = verify_sa3(paths, C.MODE_TO_VARIANT[mode], source, log) and ok
    return ok


def main(argv: Optional[List[str]] = None, input_fn: Callable[[str], str] = input,
         log: Callable[[str], None] = print) -> int:
    args = build_parser().parse_args(argv)
    paths = ModelPaths(args.models_dir)
    try:
        if args.check:
            return 0 if check_installation(paths, "auto" if args.source == "comfy" else "official", log) else 1

        interactive = args.only is None and not args.yes and not args.dry_run and sys.stdin.isatty()
        groups = parse_only(args.only)
        rows = build_plan(groups, args.source, paths)
        if interactive:
            print_plan(build_plan(list(GROUPS), args.source, paths), paths, log)
            groups = prompt_groups(rows, input_fn)
            if not groups:
                log("nothing selected")
                return 0
            rows = build_plan(groups, args.source, paths)
        print_plan(rows, paths, log)
        if args.dry_run:
            return 0
        if args.source == "official" and not args.hf_token:
            log("\nnote: --source official needs a Hugging Face token (the repos are gated). "
                + _gated_hint("stabilityai/stable-audio-3-small-sfx"))

        log("")
        token = args.hf_token if args.source == "official" else None
        if "tts" in groups:
            install_tts(paths, args.force, None, log, spacy_model=not args.no_spacy)
        for group in ("sfx", "music"):
            if group not in groups:
                continue
            variant = C.MODE_TO_VARIANT[group]
            if args.source == "official":
                install_sa3_official(paths, variant, args.force, token, log)
            else:
                install_sa3_comfy(paths, variant, args.force, args.revision, None, log)
        if not args.no_verify:
            log("\nverifying ...")
            for group in ("sfx", "music"):
                if group in groups:
                    if not verify_sa3(paths, C.MODE_TO_VARIANT[group], "official" if args.source == "official" else "comfy", log):
                        return 6
        log("\nDone. Try:  python scripts/generate_audio.py --mode sfx --prompt \"door slam\" --duration 3 --dry-run")
        return 0
    except LocalAudioError as exc:
        log(f"error: {exc}")
        return exc.exit_code
    except KeyboardInterrupt:
        log("interrupted; re-run to resume")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
