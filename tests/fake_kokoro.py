"""A random-weight Kokoro-82M with the *real* architecture (same size as the real 327 MB file).

Used by the slow integration test to exercise the real kokoro/misaki/spaCy/espeak code paths.
"""
import json, sys
from pathlib import Path
from unittest import mock
import torch

def make(models_dir: Path):
    from kokoro import KModel
    from misaki import en
    chars = set()
    for n in dir(en):
        v = getattr(en, n)
        if n.endswith("_VOCAB") and isinstance(v, (str, frozenset, set)):
            chars |= set("".join(v)) if isinstance(v, str) else set(v)
    chars |= set(";:,.!?—…\"() “” abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    vocab_chars = sorted(chars)
    vocab = {ch: i + 1 for i, ch in enumerate(vocab_chars[:170])}
    config = {
        "istftnet": {"upsample_kernel_sizes": [20, 12], "upsample_rates": [10, 6], "gen_istft_hop_size": 5,
                     "gen_istft_n_fft": 20, "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                     "resblock_kernel_sizes": [3, 7, 11], "upsample_initial_channel": 512},
        "dim_in": 64, "dropout": 0.2, "hidden_dim": 512, "max_conv_dim": 512, "max_dur": 50, "multispeaker": True,
        "n_layer": 3, "n_mels": 80, "n_token": 178, "style_dim": 128, "text_encoder_kernel_size": 5,
        "plbert": {"hidden_size": 768, "num_attention_heads": 12, "intermediate_size": 2048,
                   "max_position_embeddings": 512, "num_hidden_layers": 12, "dropout": 0.1},
        "vocab": vocab,
    }
    kdir = models_dir / "kokoro"; (kdir / "voices").mkdir(parents=True, exist_ok=True)
    (kdir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    torch.manual_seed(0)
    with mock.patch("torch.load", return_value={}):
        m = KModel(repo_id="hexgrad/Kokoro-82M", config=config, model="ignored")
    state = {k: getattr(m, k).state_dict() for k in ["bert", "bert_encoder", "predictor", "text_encoder", "decoder"]}
    torch.save(state, kdir / "kokoro-v1_0.pth")
    for name in ("af_heart", "af_bella", "hf_alpha"):
        torch.save(torch.randn(510, 1, 256) * 0.1, kdir / "voices" / f"{name}.pt")
    return kdir

if __name__ == "__main__":
    out = make(Path(sys.argv[1]))
    print("fake kokoro at", out, sorted(p.name for p in out.iterdir()))
