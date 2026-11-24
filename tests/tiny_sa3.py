"""A miniature, random-weight Stable Audio 3 for integration tests.

It is built by the *real* upstream code from a scaled-down copy of the vendored
config, so it exercises the same ``generate()`` / inpainting / conditioning
paths as the real model - just with a few thousand parameters and a stand-in
text encoder (the real T5Gemma needs the 1 GB weight file).
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

from localaudio import config as C


def tiny_config() -> Dict[str, Any]:
    cfg = json.loads((C.ASSETS_DIR / "sa3_small_sfx_config.json").read_text(encoding="utf-8"))
    pre = cfg["model"]["pretransform"]["config"]
    for part in ("encoder", "decoder"):
        sub = pre[part]["config"]
        sub["channels"] = 16
        sub["c_mults"] = [4]
        sub["transformer_depths"] = [1]
        sub["latent_dim"] = 16
    pre["bottleneck"]["config"]["dim"] = 16
    pre["latent_dim"] = 16
    cfg["model"]["io_channels"] = 16
    cfg["model"]["conditioning"]["cond_dim"] = 32
    dit = cfg["model"]["diffusion"]["config"]
    dit.update(io_channels=16, embed_dim=64, depth=2, num_heads=2, cond_token_dim=32,
               global_cond_dim=32, local_add_cond_dim=17, num_memory_tokens=4)
    return cfg


def _fake_t5_class():
    import torch
    from stable_audio_3.models.conditioners import Conditioner

    class FakeT5Conditioner(Conditioner):
        """Deterministic stand-in for T5GemmaConditioner (no weights, no tokenizer)."""

        def __init__(self, output_dim: int, max_length: int = 256, padding_mode: str = "learned", **_: Any):
            super().__init__(output_dim, output_dim, padding_mode=padding_mode)
            self.max_length = max_length
            self.__dict__["model"] = torch.nn.Identity()  # mimic the unregistered encoder
            self._device_initialized = False
            self.calls = []

        def forward(self, inputs, device):
            self.calls.append(list(inputs))
            self._device_initialized = True
            embs, masks = [], []
            for text in inputs:
                seed = int(hashlib.sha256(str(text).encode()).hexdigest()[:8], 16)
                gen = torch.Generator().manual_seed(seed)
                n = max(1, min(len(str(text).split()), self.max_length))
                emb = torch.randn(self.max_length, self.output_dim, generator=gen)
                mask = torch.zeros(self.max_length, dtype=torch.bool)
                mask[:n] = True
                embs.append(emb)
                masks.append(mask)
            embeddings = torch.stack(embs).to(device)
            attention_mask = torch.stack(masks).to(device)
            embeddings = self.apply_padding(embeddings, attention_mask)
            return embeddings, attention_mask

    return FakeT5Conditioner


@contextlib.contextmanager
def patched_t5() -> Iterator[None]:
    from unittest import mock

    with mock.patch("stable_audio_3.factory.T5GemmaConditioner", _fake_t5_class()):
        yield


def build_tiny_model(device: str = "cpu") -> Tuple[Any, Dict[str, Any]]:
    """Return ``(StableAudioModel, config)`` with random weights (seeded)."""
    import torch
    from stable_audio_3.factory import create_diffusion_cond_from_config
    from stable_audio_3.model import StableAudioModel

    cfg = tiny_config()
    torch.manual_seed(0)
    with patched_t5():
        model = create_diffusion_cond_from_config(copy.deepcopy(cfg))
    model.eval().requires_grad_(False)
    model.use_lora = False
    model.lora_names = []
    model.to(device)
    return StableAudioModel(model, cfg, device, False), cfg


def save_tiny_checkpoint(model: Any, path: Path) -> Path:
    """Write the tiny network's state dict as safetensors (same layout as the real file)."""
    from safetensors.torch import save_file

    state = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path))
    return path
