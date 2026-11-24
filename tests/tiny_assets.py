"""Miniature but *complete* Stable Audio 3 assets in the un-gated ComfyUI layout.

Everything the production loader consumes is generated here with random weights:

* ``model.safetensors`` for a scaled-down SA3 (same upstream architecture code),
* a single-file T5Gemma encoder (hidden size 768 like the real one, 1 layer,
  tiny vocabulary) using the ``model.encoder.*`` key layout,
* a SentencePiece tokenizer trained on a toy corpus,
* the matching config files.

With these, ``build_stable_audio_model`` runs its real code path end to end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

from localaudio import config as C
from localaudio.device import DevicePlan
from localaudio.models import Sa3Assets

from tiny_sa3 import tiny_config

TINY_VOCAB = 128


def tiny_t5_config_dict() -> Dict[str, Any]:
    cfg = json.loads((C.ASSETS_DIR / C.T5_CONFIG_ASSET).read_text(encoding="utf-8"))
    for part in ("encoder", "decoder"):
        cfg[part]["num_hidden_layers"] = 1
        cfg[part]["layer_types"] = ["sliding_attention"]
        cfg[part]["vocab_size"] = TINY_VOCAB
    return cfg


def write_tiny_t5(tmp: Path, prefix: str = "model.encoder.", extra_decoder: bool = False) -> Tuple[Path, Path]:
    """Return ``(weights.safetensors, config.json)`` for the tiny encoder."""
    import torch
    import transformers
    from safetensors.torch import save_file

    cfg_path = tmp / "t5_config.json"
    cfg_path.write_text(json.dumps(tiny_t5_config_dict()), encoding="utf-8")
    cfg = transformers.T5GemmaConfig.from_json_file(str(cfg_path))
    cfg.is_encoder_decoder = False
    torch.manual_seed(1)
    enc = transformers.T5GemmaEncoderModel(cfg)
    state = {}
    for key, value in enc.state_dict().items():  # keys look like "encoder.layers.0...."
        state[prefix + key[len("encoder."):] if prefix else key[len("encoder."):]] = value.clone().contiguous()
    if extra_decoder:
        state["model.decoder.layers.0.self_attn.q_proj.weight"] = torch.zeros(4, 4)
        state["lm_head.weight"] = torch.zeros(4, 4)
    weights = tmp / "t5gemma_tiny.safetensors"
    save_file(state, str(weights))
    return weights, cfg_path


def write_tiny_tokenizer(tmp: Path) -> Path:
    import sentencepiece as spm

    corpus = tmp / "corpus.txt"
    words = ("a dog barking rain on tin roof footsteps gravel door slam laser blast calm acoustic guitar "
             "TrackType SFX Music no music binaural outdoor field recording studio close mic foley").split()
    corpus.write_text("\n".join(" ".join(words[i:] + words[:i]) for i in range(len(words))), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(tmp / "tiny_spm"), vocab_size=TINY_VOCAB - 1, model_type="unigram",
        pad_id=0, unk_id=1, bos_id=-1, eos_id=-1, hard_vocab_limit=False, minloglevel=2,
    )
    return tmp / "tiny_spm.model"


def build_tiny_assets(tmp: Path, variant: str = "small-sfx") -> Tuple[Sa3Assets, Any]:
    """Create tiny assets in ``tmp`` -> ``(Sa3Assets, reference_model)``.

    The reference model is built through the production patches (real
    ``T5GemmaConditioner`` + tiny encoder) so its state dict matches what
    ``build_stable_audio_model`` will expect.
    """
    import torch
    from stable_audio_3.factory import create_diffusion_cond_from_config

    from localaudio.engines.sa3_loader import comfy_t5_patches, load_model_config
    from tiny_sa3 import save_tiny_checkpoint

    t5_weights, t5_config = write_tiny_t5(tmp)
    tokenizer = write_tiny_tokenizer(tmp)
    cfg_path = tmp / "sa3_config.json"
    cfg = tiny_config()
    cfg["model"]["conditioning"]["configs"][0]["config"].update(
        {"model_name": "google/t5gemma-b-b-ul2", "max_length": 32}
    )
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    assets_stub = Sa3Assets(variant, "comfy", tmp / "model.safetensors", cfg_path,
                            t5_weights=t5_weights, t5_config=t5_config, tokenizer_model=tokenizer)
    torch.manual_seed(7)
    with comfy_t5_patches(assets_stub, torch.float32):
        model = create_diffusion_cond_from_config(load_model_config(assets_stub))
    model.eval().requires_grad_(False)
    save_tiny_checkpoint(model, assets_stub.checkpoint)
    return assets_stub, model


CPU_PLAN = DevicePlan("cpu", "float32", "test")
