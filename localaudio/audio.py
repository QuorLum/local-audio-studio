"""Small audio helpers built on numpy (+ soundfile for writing).

Convention used everywhere in this project: audio is a float32 ``numpy`` array
shaped ``(channels, frames)`` with samples in ``[-1, 1]``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np

from .errors import DependencyError, RequestError

Array = np.ndarray
BIT_DEPTHS = {"16": "PCM_16", "24": "PCM_24", "32f": "FLOAT"}


def as_channels_first(x) -> Array:
    """Coerce a 1-D or 2-D array into ``(channels, frames)`` float32."""
    a = np.asarray(x, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    if a.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got shape {a.shape}")
    return np.ascontiguousarray(a)


def duration_seconds(x: Array, sr: int) -> float:
    return x.shape[-1] / float(sr)


def silence(seconds: float, sr: int, channels: int = 1) -> Array:
    return np.zeros((channels, max(0, int(round(seconds * sr)))), dtype=np.float32)


def _gains(n: int, curve: str) -> Tuple[Array, Array]:
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    if curve == "linear":
        return 1.0 - t, t
    if curve == "equal_power":
        return np.cos(t * math.pi / 2).astype(np.float32), np.sin(t * math.pi / 2).astype(np.float32)
    raise ValueError(f"unknown crossfade curve {curve!r}")


def crossfade(a: Array, b: Array, n: int, curve: str = "equal_power") -> Array:
    """Join ``a`` and ``b`` overlapping the last/first ``n`` samples.

    ``equal_power`` suits uncorrelated material (loops, unrelated clips);
    ``linear`` suits two renderings of the *same* audio (continuation stitching)
    because the gains sum to one and add no 3 dB bump.
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError("channel count mismatch in crossfade")
    n = int(min(n, a.shape[1], b.shape[1]))
    if n <= 0:
        return np.concatenate([a, b], axis=1)
    g_out, g_in = _gains(n, curve)
    blend = a[:, -n:] * g_out + b[:, :n] * g_in
    return np.concatenate([a[:, :-n], blend, b[:, n:]], axis=1).astype(np.float32, copy=False)


def fade(x: Array, sr: int, fade_in: float = 0.0, fade_out: float = 0.0) -> Array:
    """Raised-cosine fades at the clip edges (seconds)."""
    out = x.copy()
    n_in = min(int(round(fade_in * sr)), out.shape[1])
    n_out = min(int(round(fade_out * sr)), out.shape[1])
    if n_in > 1:
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, math.pi, n_in))).astype(np.float32)
        out[:, :n_in] *= ramp
    if n_out > 1:
        ramp = (0.5 + 0.5 * np.cos(np.linspace(0, math.pi, n_out))).astype(np.float32)
        out[:, -n_out:] *= ramp
    return out


def peak_normalize(x: Array, target_db: float = -1.0) -> Array:
    """Scale so the loudest sample sits at ``target_db`` dBFS (silence is left alone)."""
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-8:
        return x
    gain = (10.0 ** (target_db / 20.0)) / peak
    return (x * gain).astype(np.float32)


def trim_silence(x: Array, sr: int, threshold_db: float = -50.0, pad_ms: float = 30.0) -> Array:
    """Cut leading/trailing samples below ``threshold_db`` (keeps ``pad_ms`` of margin)."""
    if x.size == 0:
        return x
    thr = 10.0 ** (threshold_db / 20.0)
    env = np.max(np.abs(x), axis=0)
    idx = np.flatnonzero(env > thr)
    if idx.size == 0:
        return x
    pad = int(sr * pad_ms / 1000.0)
    start, end = max(0, int(idx[0]) - pad), min(x.shape[1], int(idx[-1]) + 1 + pad)
    return x[:, start:end]


def make_loopable(x: Array, sr: int, crossfade_seconds: float = 2.0) -> Array:
    """Return a clip whose end flows seamlessly back into its start.

    The first ``n`` samples are replaced by a blend that runs from the clip's
    own tail into its head, and the tail is removed: when the result repeats,
    sample ``L-1`` is followed by what used to follow sample ``L-n-1``.
    """
    length = x.shape[1]
    n = int(min(round(crossfade_seconds * sr), length // 3))
    if n < 16:
        return x
    g_out, g_in = _gains(n, "equal_power")
    blend = x[:, -n:] * g_out + x[:, :n] * g_in
    return np.concatenate([blend, x[:, n : length - n]], axis=1).astype(np.float32, copy=False)


def loop_to_duration(clip: Array, sr: int, total_seconds: float, crossfade_seconds: float = 3.0) -> Array:
    """Tile ``clip`` with equal-power crossfades until ``total_seconds`` is reached."""
    target = int(round(total_seconds * sr))
    if clip.shape[1] >= target:
        return clip[:, :target]
    n = int(min(round(crossfade_seconds * sr), clip.shape[1] // 3))
    out = clip
    while out.shape[1] < target:
        out = crossfade(out, clip, n, "equal_power")
    return out[:, :target]


def assemble_speech(segments: Sequence[Tuple[Array, float]], sr: int, channels: int = 1) -> Array:
    """Concatenate ``(audio, pause_after_seconds)`` pairs into one clip."""
    pieces: List[Array] = []
    for audio, pause in segments:
        pieces.append(audio)
        if pause > 0:
            pieces.append(silence(pause, sr, audio.shape[0] if audio.size else channels))
    if not pieces:
        return np.zeros((channels, 0), dtype=np.float32)
    return np.concatenate(pieces, axis=1)


def resample(x: Array, sr_in: int, sr_out: int) -> Array:
    """Resample with torchaudio (a dependency of the project)."""
    if sr_in == sr_out:
        return x
    try:
        import torch
        import torchaudio.functional as AF
    except Exception as exc:  # pragma: no cover
        raise DependencyError(f"resampling needs torch + torchaudio: {exc}") from exc
    t = torch.from_numpy(np.ascontiguousarray(x))
    return AF.resample(t, sr_in, sr_out).numpy().astype(np.float32)


def write_audio(path: Union[str, Path], x: Array, sr: int, bit_depth: str = "16") -> Path:
    """Write ``(channels, frames)`` audio; the format follows the file extension."""
    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover
        raise DependencyError(f"soundfile is required to write audio: {exc}") from exc
    if bit_depth not in BIT_DEPTHS:
        raise RequestError(f"bit depth must be one of {', '.join(BIT_DEPTHS)}")
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    data = np.ascontiguousarray(np.clip(x, -1.0, 1.0).T)  # (frames, channels)
    suffix = out.suffix.lower()
    kwargs = {}
    if suffix in ("", ".wav", ".wave"):
        kwargs["subtype"] = BIT_DEPTHS[bit_depth]
        if suffix == "":
            out = out.with_suffix(".wav")
    elif suffix == ".flac":
        kwargs["subtype"] = "PCM_24" if bit_depth != "16" else "PCM_16"
    sf.write(str(out), data, int(sr), **kwargs)
    return out
