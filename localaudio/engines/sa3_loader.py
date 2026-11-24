"""Build Stability AI's official ``StableAudioModel`` from local files only.

Why this module exists
----------------------
The official ``stable_audio_3.StableAudioModel.from_pretrained`` downloads from
*gated* Hugging Face repositories. This project's default weight source is the
un-gated ComfyUI repackaging (``Comfy-Org/stable-audio-3``), which ships the
same ``model.safetensors`` tensors but no ``model_config.json``, no Hugging Face
style text-encoder folder and no tokenizer. We therefore

1. build the network with the official ``create_diffusion_cond_from_config``
   using a vendored, inference-only copy of the published config,
2. construct the T5Gemma text encoder from its single ``.safetensors`` file and
   hand it to the official ``T5GemmaConditioner`` (via narrowly scoped
   monkey-patches of three ``from_pretrained`` entry points), and
3. load the checkpoint **strictly**: every shape is checked and a mismatch
   raises :class:`ModelFormatError` instead of silently producing garbage like
   the upstream lenient ``copy_state_dict`` would.

The official gated bundle (``--source official``) goes through the same
strict loader but uses its own config and Hugging Face style T5Gemma folder.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from .. import config as C
from ..device import DevicePlan, import_torch
from ..errors import DependencyError, ModelFormatError
from ..models import Sa3Assets

Logger = Callable[[str], None]


def _noop(_msg: str) -> None:  # pragma: no cover
    return None


# --------------------------------------------------------------------------- #
# Environment helpers
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def quiet_stdout() -> Iterator[None]:
    """Route the upstream package's ``print`` noise to stderr.

    Callers (agents, shell pipelines) rely on stdout carrying only the result
    path / JSON, so nothing from third-party libraries may leak onto it.
    """
    with contextlib.redirect_stdout(sys.stderr):
        yield


@contextlib.contextmanager
def offline_env() -> Iterator[None]:
    """Belt and braces: make any accidental Hugging Face lookup fail fast."""
    wanted = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"}
    old = {k: os.environ.get(k) for k in wanted}
    os.environ.update(wanted)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def import_sa3() -> SimpleNamespace:
    """Import the official package lazily with an actionable error message."""
    try:
        with quiet_stdout():
            from stable_audio_3.factory import create_diffusion_cond_from_config
            from stable_audio_3.model import StableAudioModel
    except Exception as exc:
        raise DependencyError(
            "The official `stable_audio_3` package could not be imported "
            f"[{type(exc).__name__}: {exc}]. Install it with:\n"
            f'    pip install --no-deps "stable-audio-3 @ {C.SA3_GIT_URL}"\n'
            "(./install.sh / install.ps1 do this for you)."
        ) from exc
    return SimpleNamespace(
        StableAudioModel=StableAudioModel,
        create_diffusion_cond_from_config=create_diffusion_cond_from_config,
    )


# --------------------------------------------------------------------------- #
# Tokenizer: raw SentencePiece, exactly what Stability's own runtimes use
# --------------------------------------------------------------------------- #
class SentencePieceTokenizer:
    """The slice of the Hugging Face tokenizer API that SA3's conditioner uses.

    Stability's optimized runtimes encode prompts with plain SentencePiece:
    no BOS/EOS, truncate to 256 tokens, pad with id 0, and they validate that
    this reproduces the PyTorch reference ``input_ids`` exactly.
    """

    def __init__(self, model_file: Path | str, max_length: int = C.T5_MAX_TOKENS, pad_id: int = 0):
        try:
            import sentencepiece as spm
        except Exception as exc:  # pragma: no cover
            raise DependencyError(f"`sentencepiece` is required: {exc}") from exc
        self._sp = spm.SentencePieceProcessor(model_file=str(model_file))
        self.max_length = max_length
        self.pad_id = pad_id

    def encode(self, text: str) -> List[int]:
        return list(self._sp.encode(text))[: self.max_length]

    def __call__(self, texts, truncation: bool = True, max_length: Optional[int] = None,
                 padding: str = "max_length", return_tensors: str = "pt", **_: Any) -> Dict[str, Any]:
        import torch

        if isinstance(texts, str):
            texts = [texts]
        length = int(max_length or self.max_length)
        input_ids = torch.full((len(texts), length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(texts), length), dtype=torch.long)
        for row, text in enumerate(texts):
            ids = list(self._sp.encode(text))[:length]
            if ids:
                input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
                attention_mask[row, : len(ids)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


# --------------------------------------------------------------------------- #
# Strict weight loading
# --------------------------------------------------------------------------- #
@dataclass
class WeightReport:
    matched: int = 0
    total_expected: int = 0
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    mismatched: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = field(default_factory=list)
    expected_numel: int = 0
    missing_numel: int = 0
    file_numel: int = 0
    unexpected_numel: int = 0

    @property
    def coverage(self) -> float:
        return 1.0 - (self.missing_numel / self.expected_numel) if self.expected_numel else 0.0

    def summary(self) -> str:
        return (
            f"{self.matched}/{self.total_expected} tensors matched "
            f"({self.coverage * 100:.3f}% of parameters), "
            f"{len(self.missing)} missing, {len(self.unexpected)} unexpected, "
            f"{len(self.mismatched)} shape mismatches"
        )


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return n


def _remap_like_upstream(key: str, expected: Dict[str, Any]) -> Optional[str]:
    """Mirror ``stable_audio_3.loading_utils.remap_state_dict_keys`` for one key."""
    if key in expected:
        return key
    parts = key.split(".")
    for i in range(1, len(parts)):
        candidate = ".".join(parts[:i]) + "." + ".".join(parts[i + 1 :])
        if candidate in expected:
            return candidate
    return None


def load_weights_strict(
    model: Any,
    ckpt_path: Path | str,
    *,
    copy: bool = True,
    max_missing_ratio: float = 0.001,
    log: Logger = _noop,
) -> WeightReport:
    """Copy ``ckpt_path`` into ``model`` tensor by tensor, verifying every shape.

    Raises :class:`ModelFormatError` if any shape differs or if more than
    ``max_missing_ratio`` of the parameters (by element count) are absent from
    the file - both mean the config and the checkpoint do not belong together.
    With ``copy=False`` only the header is inspected (cheap, nothing is read).
    """
    from safetensors import safe_open

    torch = import_torch()
    expected = model.state_dict()  # shares storage with the parameters
    shapes = {k: tuple(v.shape) for k, v in expected.items()}
    report = WeightReport(total_expected=len(shapes), expected_numel=sum(_numel(s) for s in shapes.values()))
    matched: Dict[str, str] = {}

    with safe_open(str(ckpt_path), framework="pt", device="cpu") as handle:
        for file_key in handle.keys():
            file_shape = tuple(handle.get_slice(file_key).get_shape())
            report.file_numel += _numel(file_shape)
            model_key = _remap_like_upstream(file_key, shapes)
            if model_key is None:
                report.unexpected.append(file_key)
                report.unexpected_numel += _numel(file_shape)
            elif file_shape != shapes[model_key]:
                report.mismatched.append((file_key, file_shape, shapes[model_key]))
            else:
                matched[file_key] = model_key
        report.matched = len(matched)
        report.missing = sorted(set(shapes) - set(matched.values()))
        report.missing_numel = sum(_numel(shapes[k]) for k in report.missing)

        if report.mismatched:
            fk, fs, ms = report.mismatched[0]
            raise ModelFormatError(
                f"{Path(ckpt_path).name}: {len(report.mismatched)} tensors have a different shape than the "
                f"architecture expects (first: {fk} is {fs}, expected {ms}). The checkpoint does not match "
                "the bundled config. Try `python scripts/setup_models.py --source official` (needs a free "
                "Hugging Face token and the license accepted) or report an issue."
            )
        if report.expected_numel and report.missing_numel / report.expected_numel > max_missing_ratio:
            raise ModelFormatError(
                f"{Path(ckpt_path).name}: {report.summary()}; {len(report.missing)} tensors the network needs "
                f"are absent from the file (e.g. {report.missing[:3]}). Wrong or incomplete download? "
                "Re-run scripts/setup_models.py --force."
            )
        if report.missing:
            log(f"note: {len(report.missing)} small tensors absent from checkpoint (e.g. {report.missing[:2]})")
        if report.unexpected:
            log(f"note: {len(report.unexpected)} unused tensors in checkpoint (e.g. {report.unexpected[:2]})")

        if copy:
            with torch.no_grad():
                for file_key, model_key in matched.items():
                    expected[model_key].copy_(handle.get_tensor(file_key))
    return report


# --------------------------------------------------------------------------- #
# T5Gemma text encoder from a single .safetensors file (ComfyUI layout)
# --------------------------------------------------------------------------- #
def _t5_key_map(file_keys: Sequence[str], expected: Sequence[str]) -> Dict[str, str]:
    """Map keys of a T5Gemma checkpoint onto ``T5GemmaEncoderModel`` keys.

    Accepts the layouts seen in the wild: ``model.encoder.*`` (full T5Gemma
    export), ``encoder.*`` (encoder-only export) and bare ``layers.*`` names.
    Decoder / LM-head tensors are ignored.
    """
    want = set(expected)
    out: Dict[str, str] = {}
    for key in file_keys:
        if key.startswith(("model.decoder.", "decoder.", "lm_head.", "model.lm_head.")):
            continue
        candidates = [key]
        if key.startswith("model."):
            candidates.append(key[len("model."):])
        candidates += ["encoder." + key, "encoder." + key[len("model."):] if key.startswith("model.") else key]
        for cand in candidates:
            if cand in want:
                out[key] = cand
                break
    return out


def build_t5_encoder_from_single_file(weights: Path, config_json: Path, dtype: Any, config: Any = None) -> Any:
    """Create a ``T5GemmaEncoderModel`` and fill it from ``weights`` (strict)."""
    import torch
    import transformers
    from safetensors import safe_open

    if config is None:
        config = transformers.T5GemmaConfig.from_json_file(str(config_json))
    config.is_encoder_decoder = False
    encoder = transformers.T5GemmaEncoderModel(config)
    shapes = {k: tuple(v.shape) for k, v in encoder.state_dict().items()}
    params = encoder.state_dict()

    with safe_open(str(weights), framework="pt", device="cpu") as handle:
        mapping = _t5_key_map(list(handle.keys()), list(shapes))
        missing = sorted(set(shapes) - set(mapping.values()))
        if missing:
            raise ModelFormatError(
                f"{Path(weights).name}: {len(missing)} of {len(shapes)} T5Gemma encoder tensors are missing "
                f"(e.g. {missing[:3]}). This is not the t5gemma-b-b-ul2 encoder file."
            )
        for file_key, model_key in mapping.items():
            fshape = tuple(handle.get_slice(file_key).get_shape())
            if fshape != shapes[model_key]:
                raise ModelFormatError(
                    f"{Path(weights).name}: {file_key} has shape {fshape}, expected {shapes[model_key]}"
                )
        with torch.no_grad():
            for file_key, model_key in mapping.items():
                params[model_key].copy_(handle.get_tensor(file_key))
    return encoder.to(dtype).eval().requires_grad_(False)


@contextlib.contextmanager
def comfy_t5_patches(assets: Sa3Assets, t5_dtype: Any, verify_only: bool = False,
                     log: Logger = _noop) -> Iterator[None]:
    """Make the official ``T5GemmaConditioner`` use our local single-file encoder.

    The conditioner calls ``AutoTokenizer.from_pretrained``,
    ``AutoConfig.from_pretrained`` and ``T5GemmaEncoderModel.from_pretrained``
    during ``__init__``. Inside this context those three return the local
    objects instead of touching the network. Nothing else is altered and the
    patches are removed on exit.
    """
    from unittest import mock

    import torch
    import transformers

    tokenizer = None if verify_only else SentencePieceTokenizer(assets.tokenizer_model)

    def fake_config(*_a: Any, **_k: Any) -> Any:
        return transformers.T5GemmaConfig.from_json_file(str(assets.t5_config))

    def fake_tokenizer(*_a: Any, **_k: Any) -> Any:
        return tokenizer if tokenizer is not None else (lambda *a, **k: {})

    def fake_model(*_a: Any, config: Any = None, **_k: Any) -> Any:
        if verify_only:
            return torch.nn.Identity()
        log("loading T5Gemma text encoder ...")
        return build_t5_encoder_from_single_file(assets.t5_weights, assets.t5_config, t5_dtype, config=config)

    with mock.patch.object(transformers.AutoConfig, "from_pretrained", staticmethod(fake_config)), \
         mock.patch.object(transformers.AutoTokenizer, "from_pretrained", staticmethod(fake_tokenizer)), \
         mock.patch.object(transformers.T5GemmaEncoderModel, "from_pretrained", staticmethod(fake_model)):
        yield


# --------------------------------------------------------------------------- #
# Config + model assembly
# --------------------------------------------------------------------------- #
def load_model_config(assets: Sa3Assets) -> Dict[str, Any]:
    """Read the model config and point the text-encoder entry at local data."""
    cfg = json.loads(Path(assets.config_json).read_text(encoding="utf-8"))
    for entry in cfg["model"]["conditioning"]["configs"]:
        if entry.get("type") != "t5gemma":
            continue
        sub = entry["config"]
        sub.pop("repo_id", None)
        sub.pop("subfolder", None)
        sub.pop("model_path", None)
        if assets.source == "official":
            sub["model_path"] = str(assets.t5_dir)
    return cfg


def choose_t5_dtype(torch: Any, plan: DevicePlan) -> Any:
    """bf16 on GPUs that support it (Gemma is trained in bf16), fp32 otherwise."""
    if plan.is_cuda:
        try:
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        except Exception:
            pass
    return torch.float32


def build_stable_audio_model(assets: Sa3Assets, plan: DevicePlan, log: Logger = _noop) -> Tuple[Any, WeightReport]:
    """Return ``(StableAudioModel, WeightReport)`` ready for ``generate``.

    Peak-memory notes for 4 GB GPUs: the network is built and filled on the CPU,
    cast to fp16 *there*, and only then moved to the GPU, so VRAM never has to
    hold the 2.3 GB fp32 copy that the upstream loader transfers first.
    """
    torch = import_torch()
    sa3 = import_sa3()
    with offline_env(), quiet_stdout():
        cfg = load_model_config(assets)
        t5_dtype = choose_t5_dtype(torch, plan)
        log("building network ...")
        ctx = comfy_t5_patches(assets, t5_dtype, log=log) if assets.source == "comfy" else contextlib.nullcontext()
        with ctx:
            model = sa3.create_diffusion_cond_from_config(cfg)
        log(f"loading {assets.checkpoint.name} (strict) ...")
        report = load_weights_strict(model, assets.checkpoint, log=log)
        log(report.summary())
        model.eval().requires_grad_(False)
        model.use_lora = False
        model.lora_names = []
        dtype = plan.torch_dtype(torch)
        if dtype != torch.float32:
            model.to(dtype)
        model.to(plan.device)
        studio_model = sa3.StableAudioModel(model, cfg, plan.device, plan.dtype == "float16")
    return studio_model, report


def verify_assets(assets: Sa3Assets, log: Logger = _noop) -> WeightReport:
    """Header-only check that the downloaded checkpoint fits the architecture.

    Builds the network on the ``meta`` device (no memory, no weights read) and
    compares every tensor name/shape with the safetensors header.
    """
    torch = import_torch()
    sa3 = import_sa3()
    with offline_env(), quiet_stdout():
        cfg = load_model_config(assets)
        ctx = comfy_t5_patches(assets, torch.float32, verify_only=True) if assets.source == "comfy" \
            else contextlib.nullcontext()
        with ctx, torch.device("meta"):
            model = sa3.create_diffusion_cond_from_config(cfg)
        return load_weights_strict(model, assets.checkpoint, copy=False, log=log)
