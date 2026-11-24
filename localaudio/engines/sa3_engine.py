"""Stable Audio 3 Small engine (sound effects and music).

Wraps the official ``StableAudioModel`` with:

* the project's inference defaults (pingpong sampler, 8 steps, CFG 1.0),
* seamless extension beyond the model's 120 s limit, either by causal
  *continuation* (the official inpainting path: the tail of the audio so far is
  given as context and only the new region is generated) or by crossfaded
  *looping*,
* text-encoder offloading so a 4 GB GPU never holds the encoder and the
  diffusion transformer at the same time while sampling.
"""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np

from .. import config as C
from ..audio import crossfade, loop_to_duration, make_loopable
from ..device import cleanup_memory, cuda_memory_line, import_torch
from ..errors import ModelsMissingError
from ..jobs import SoundJob
from ..models import Sa3Assets
from .base import AudioResult, Engine
from .sa3_loader import build_stable_audio_model, quiet_stdout


class Sa3Engine(Engine):
    """One Stable Audio 3 Small variant (``small-sfx`` or ``small-music``)."""

    def __init__(self, paths, plan, variant: str, source: str = "auto", log=None, progress: bool = True):
        from .base import stderr_logger

        super().__init__(paths, plan, log or stderr_logger)
        if variant not in C.SA3_VARIANTS:
            raise KeyError(variant)
        self.variant = variant
        self.source = source
        self.key = C.SA3_VARIANTS[variant].mode
        self.display_name = f"Stable Audio 3 {variant}"
        self.progress = progress
        self._sa: Any = None
        self._assets: Optional[Sa3Assets] = None
        self._report = None

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        if self._loaded:
            return
        assets = self.paths.sa3_assets(self.variant, self.source)
        if assets is None:
            missing = self.paths.missing(self.key, self.source)
            raise ModelsMissingError(
                f"{self.display_name} files are missing:\n  " + "\n  ".join(missing or ["(unknown)"]) +
                f"\nRun: python scripts/setup_models.py --only {self.key}"
            )
        self.log(f"loading {self.display_name} ({assets.source} files) on {self.plan.short()} ...")
        self._sa, self._report = build_stable_audio_model(assets, self.plan, self.log)
        self._assets = assets
        self._loaded = True

    def unload(self) -> None:
        sa, self._sa = self._sa, None
        if sa is not None:
            try:  # return VRAM even if something still references the Python object
                self._text_encoder_to_cpu(sa)
                sa.model.to("cpu")
            except Exception:
                pass
        self._loaded = False

    # --------------------------------------------------------- text encoder
    @staticmethod
    def _text_encoder_to_cpu(sa: Any) -> None:
        cond = sa.model.conditioner.conditioners["prompt"]
        cond.model.to("cpu")
        cond._device_initialized = False  # make the next forward move it back to the device

    def _encode(self, prompt: str, seconds: float):
        torch = import_torch()
        cond = [{"prompt": prompt, "seconds_total": float(seconds)}]
        with torch.inference_mode():
            tensors = self._sa.model.conditioner(cond, self.plan.device)
        return cond, tensors

    # --------------------------------------------------------- single clip
    def _clip(self, prompt: str, negative: Optional[str], seconds: float, seed: int, steps: int, cfg: float,
              inpaint: Optional[tuple] = None) -> np.ndarray:
        """Generate ``seconds`` of stereo audio -> ``(2, frames)`` float32."""
        torch = import_torch()
        sa = self._sa
        cond, tensors = self._encode(prompt, seconds)
        kwargs = dict(
            conditioning=cond,
            conditioning_tensors=tensors,
            duration=float(seconds),
            steps=int(steps),
            cfg_scale=float(cfg),
            seed=int(seed),
            sampler_type=C.SA3_SAMPLER,
            disable_tqdm=not self.progress,
        )
        if negative and cfg > 1.0:  # negative prompts only act when CFG is on
            ncond, ntensors = self._encode(negative, seconds)
            kwargs.update(negative_conditioning=ncond, negative_conditioning_tensors=ntensors)
        if inpaint is not None:
            context, start_s, end_s = inpaint
            kwargs.update(
                inpaint_audio=(C.SA3_SAMPLE_RATE, torch.from_numpy(np.ascontiguousarray(context))),
                inpaint_mask_start_seconds=float(start_s),
                inpaint_mask_end_seconds=float(end_s),
            )
        if self.plan.is_cuda:
            self._text_encoder_to_cpu(sa)  # free VRAM before sampling
            cleanup_memory(torch)
        with quiet_stdout():
            out = sa.generate(**kwargs)  # [1, 2, T], float32 in [-1, 1]
        audio = out[0].detach().float().cpu().numpy()
        del out, tensors
        return audio

    # ------------------------------------------------------------ extension
    def _extend_by_continuation(self, job: SoundJob, prompt: str, seed: int) -> np.ndarray:
        sr, hop = C.SA3_SAMPLE_RATE, C.SA3_LATENT_HOP
        target = int(round(job.duration * sr))
        max_len = int(C.SA3_MAX_SECONDS * sr)
        xf_want = int(round(job.crossfade_seconds * sr))

        self.log(f"generating base clip ({C.SA3_MAX_SECONDS:.0f} s) ...")
        audio = self._clip(prompt, job.negative_prompt, C.SA3_MAX_SECONDS, seed, job.steps, job.cfg_scale)
        step = 0
        while audio.shape[1] < target:
            step += 1
            remaining = target - audio.shape[1]
            ctx = min(int(job.context_seconds * sr), audio.shape[1])
            ctx = (ctx // hop) * hop  # keep the context boundary on the latent grid
            new = min(max_len - ctx, remaining)
            total_s = (ctx + new) / sr
            self.log(f"extending: step {step}, +{new / sr:.1f} s from {ctx / sr:.0f} s of context "
                     f"({audio.shape[1] / sr:.0f} s of {job.duration:.0f} s so far)")
            context = audio[:, -ctx:]
            out = self._clip(
                prompt, job.negative_prompt, total_s, seed + step, job.steps, job.cfg_scale,
                inpaint=(context, ctx / sr, total_s),
            )
            xf = min(xf_want, ctx, out.shape[1] - ctx if out.shape[1] > ctx else 0)
            # `out` starts with a re-rendering of the context: blend old tail -> new rendering, then append
            audio = crossfade(audio, out[:, ctx - xf :], xf, curve="linear")
        return audio[:, :target]

    def _extend_by_looping(self, job: SoundJob, prompt: str, seed: int) -> np.ndarray:
        self.log(f"generating base clip ({C.SA3_MAX_SECONDS:.0f} s) to loop ...")
        base = self._clip(prompt, job.negative_prompt, C.SA3_MAX_SECONDS, seed, job.steps, job.cfg_scale)
        return loop_to_duration(base, C.SA3_SAMPLE_RATE, job.duration, max(job.crossfade_seconds, 1.0))

    # ------------------------------------------------------------- generate
    def generate(self, job: SoundJob) -> AudioResult:
        job.validate()
        if job.mode != self.key:
            raise ValueError(f"{self.display_name} cannot render a {job.mode!r} job")
        if not self._loaded:
            self.load()
        seed = int(job.seed) if job.seed is not None else random.randint(0, 2**31 - 1)
        prompt = job.prompt.strip()
        method = None
        if not job.needs_extension:
            audio = self._clip(prompt, job.negative_prompt, job.duration, seed, job.steps, job.cfg_scale)
        else:
            method = "continue" if job.extend_method == "auto" else job.extend_method
            if method == "continue":
                audio = self._extend_by_continuation(job, prompt, seed)
            else:
                audio = self._extend_by_looping(job, prompt, seed)
        if job.loopable:
            audio = make_loopable(audio, C.SA3_SAMPLE_RATE, job.crossfade_seconds)
        mem = cuda_memory_line()
        if mem:
            self.log(mem)
        info = {
            "engine": self.display_name,
            "weights": self._assets.source if self._assets else None,
            "prompt": prompt,
            "negative_prompt": job.negative_prompt if job.cfg_scale > 1.0 else None,
            "duration_requested": job.duration,
            "steps": job.steps,
            "cfg_scale": job.cfg_scale,
            "sampler": C.SA3_SAMPLER,
            "seed": seed,
            "extension": method,
            "loopable": job.loopable,
            "device": self.plan.to_dict(),
        }
        return AudioResult(audio.astype(np.float32, copy=False), C.SA3_SAMPLE_RATE, info)
