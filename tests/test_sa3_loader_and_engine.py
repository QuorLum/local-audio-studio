"""Needs torch + the official stable_audio_3 package (skipped otherwise)."""

import numpy as np
import pytest

from localaudio import config as C
from localaudio.device import DevicePlan
from localaudio.errors import ModelFormatError, ModelsMissingError
from localaudio.jobs import SoundJob
from localaudio.models import ModelPaths

pytestmark = pytest.mark.usefixtures("torch_stack")

CPU = DevicePlan("cpu", "float32", "test")


@pytest.fixture
def L():
    from localaudio.engines import sa3_loader

    return sa3_loader


@pytest.fixture(scope="module")
def tiny_assets(tmp_path_factory, torch_stack):
    from tiny_assets import build_tiny_assets

    tmp = tmp_path_factory.mktemp("tiny_assets")
    assets, reference = build_tiny_assets(tmp)
    return assets, reference


# ------------------------------------------------------------- tokenizer
def test_sentencepiece_shim_shapes_padding_and_truncation(tiny_assets, torch_stack):
    from localaudio.engines.sa3_loader import SentencePieceTokenizer

    assets, _ = tiny_assets
    tok = SentencePieceTokenizer(assets.tokenizer_model, max_length=8)
    enc = tok(["a dog barking", ""], truncation=True, max_length=8, padding="max_length", return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    assert tuple(ids.shape) == (2, 8) == tuple(mask.shape)
    assert mask[0].sum() > 0 and mask[1].sum() == 0  # empty prompt -> fully masked
    assert int(ids[0, -1]) == 0 and int(mask[0, -1]) == 0  # right-padded with pad id 0
    long = tok("dog " * 100, max_length=8)
    assert int(long["attention_mask"].sum()) == 8  # truncated to max_length
    assert tok.encode("dog " * 1000).__len__() == 8
    assert isinstance(tok("a dog")["input_ids"], torch_stack.Tensor)


# ------------------------------------------------------------ strict loader
def _fresh_tiny(assets):
    import torch
    from stable_audio_3.factory import create_diffusion_cond_from_config

    from localaudio.engines.sa3_loader import comfy_t5_patches, load_model_config

    torch.manual_seed(99)  # different random weights than the reference
    with comfy_t5_patches(assets, torch.float32):
        model = create_diffusion_cond_from_config(load_model_config(assets))
    return model.eval().requires_grad_(False)


def test_strict_loader_copies_every_tensor(L, tiny_assets, torch_stack):
    assets, reference = tiny_assets
    fresh = _fresh_tiny(assets)
    key = next(k for k in reference.state_dict() if k.endswith("weight") and "model.model" in k)
    assert not torch_stack.equal(fresh.state_dict()[key], reference.state_dict()[key])
    report = L.load_weights_strict(fresh, assets.checkpoint)
    assert report.matched == report.total_expected and not report.missing and not report.unexpected
    assert report.coverage == pytest.approx(1.0)
    for k, v in reference.state_dict().items():
        assert torch_stack.equal(fresh.state_dict()[k], v), k


def test_header_only_check_does_not_touch_weights(L, tiny_assets, torch_stack):
    assets, reference = tiny_assets
    fresh = _fresh_tiny(assets)
    before = {k: v.clone() for k, v in fresh.state_dict().items()}
    report = L.load_weights_strict(fresh, assets.checkpoint, copy=False)
    assert report.matched == report.total_expected
    assert all(torch_stack.equal(before[k], v) for k, v in fresh.state_dict().items())


def _rewrite(assets, tmp_path, mutate):
    from safetensors.torch import load_file, save_file

    state = load_file(str(assets.checkpoint))
    mutate(state)
    path = tmp_path / "mutated.safetensors"
    save_file(state, str(path))
    return path


def test_wrong_shape_is_rejected(L, tiny_assets, tmp_path):
    import torch

    assets, _ = tiny_assets

    def mutate(state):
        key = next(k for k in state if k.endswith(".weight") and state[k].ndim == 2)
        state[key] = torch.zeros(state[key].shape[0] + 1, state[key].shape[1])

    with pytest.raises(ModelFormatError, match="different shape"):
        L.load_weights_strict(_fresh_tiny(assets), _rewrite(assets, tmp_path, mutate))


def test_large_missing_fraction_is_rejected(L, tiny_assets, tmp_path):
    assets, _ = tiny_assets

    def mutate(state):
        for key in [k for k in state if ".layers.1." in k and k.startswith("model.model")]:
            del state[key]

    with pytest.raises(ModelFormatError, match="absent"):
        L.load_weights_strict(_fresh_tiny(assets), _rewrite(assets, tmp_path, mutate))


def test_tiny_missing_tensor_and_extras_are_tolerated(L, tiny_assets, tmp_path):
    import torch

    assets, _ = tiny_assets
    notes = []

    def mutate(state):
        smallest = min(state, key=lambda k: state[k].numel())
        del state[smallest]
        state["some.extra.tensor"] = torch.zeros(3)

    report = L.load_weights_strict(_fresh_tiny(assets), _rewrite(assets, tmp_path, mutate), log=notes.append)
    assert len(report.missing) == 1 and report.unexpected == ["some.extra.tensor"]
    assert any("absent" in n for n in notes) and any("unused" in n for n in notes)


def test_upstream_style_key_remap(L):
    expected = {"model.model.layers.0.weight": 1}
    assert L._remap_like_upstream("model.model.layers.0.weight", expected) == "model.model.layers.0.weight"
    # like upstream, exactly one extra path segment is tolerated ...
    assert L._remap_like_upstream("model.EXTRA.model.layers.0.weight", expected) == "model.model.layers.0.weight"
    # ... two are not
    assert L._remap_like_upstream("model.A.B.model.layers.0.weight", expected) is None
    assert L._remap_like_upstream("model.x.layers.0.weight", {"model.layers.0.weight": 1}) == "model.layers.0.weight"
    assert L._remap_like_upstream("totally.unrelated", expected) is None


# ------------------------------------------------------------ T5 single file
@pytest.mark.parametrize("prefix, extra", [("model.encoder.", False), ("encoder.", False), ("", False),
                                           ("model.encoder.", True)])
def test_t5_single_file_key_layouts(L, tmp_path, prefix, extra, torch_stack):
    from tiny_assets import write_tiny_t5

    weights, cfg = write_tiny_t5(tmp_path, prefix=prefix, extra_decoder=extra)
    enc = L.build_t5_encoder_from_single_file(weights, cfg, torch_stack.float32)
    ids = torch_stack.tensor([[3, 4, 5, 0]])
    out = enc(input_ids=ids, attention_mask=torch_stack.tensor([[1, 1, 1, 0]]).bool())["last_hidden_state"]
    assert tuple(out.shape) == (1, 4, 768) and torch_stack.isfinite(out).all()
    assert not any(p.requires_grad for p in enc.parameters())


def test_t5_single_file_rejects_wrong_files(L, tmp_path, torch_stack):
    from safetensors.torch import load_file, save_file

    from tiny_assets import write_tiny_t5

    weights, cfg = write_tiny_t5(tmp_path)
    state = load_file(str(weights))
    broken = {k: v for k, v in state.items() if "layers.0.mlp" not in k}
    save_file(broken, str(tmp_path / "missing.safetensors"))
    with pytest.raises(ModelFormatError, match="missing"):
        L.build_t5_encoder_from_single_file(tmp_path / "missing.safetensors", cfg, torch_stack.float32)
    wrong = dict(state)
    wrong["model.encoder.embed_tokens.weight"] = torch_stack.zeros(5, 768)
    save_file(wrong, str(tmp_path / "wrong.safetensors"))
    with pytest.raises(ModelFormatError, match="shape"):
        L.build_t5_encoder_from_single_file(tmp_path / "wrong.safetensors", cfg, torch_stack.float32)


def test_patches_are_scoped_and_use_the_real_conditioner(L, tiny_assets, torch_stack):
    import transformers

    assets, _ = tiny_assets
    before = (transformers.AutoTokenizer.__dict__["from_pretrained"],
              transformers.AutoConfig.__dict__["from_pretrained"])
    model = _fresh_tiny(assets)  # built inside comfy_t5_patches with the REAL T5GemmaConditioner
    after = (transformers.AutoTokenizer.__dict__["from_pretrained"],
             transformers.AutoConfig.__dict__["from_pretrained"])
    assert before == after  # monkeypatches removed again
    assert "from_pretrained" not in transformers.T5GemmaEncoderModel.__dict__
    cond = model.conditioner.conditioners["prompt"]
    assert type(cond).__name__ == "T5GemmaConditioner"
    out = model.conditioner([{"prompt": "a dog barking", "seconds_total": 3.0}], "cpu")
    emb, mask = out["prompt"]
    assert tuple(emb.shape) == (1, 32, 32) and tuple(mask.shape) == (1, 32)  # max_length=32, cond_dim=32
    assert 0 < int(mask.sum()) < 32  # real tokens + masked padding
    assert "seconds_total" in out


def test_verify_assets_header_only_on_meta_device(L, tiny_assets):
    assets, _ = tiny_assets
    report = L.verify_assets(assets)
    assert report.matched == report.total_expected and report.coverage == pytest.approx(1.0)


def test_verify_assets_detects_foreign_checkpoint(L, tiny_assets, tmp_path):
    import dataclasses

    import torch
    from safetensors.torch import save_file

    assets, _ = tiny_assets
    bad = tmp_path / "foreign.safetensors"
    save_file({"model.model.something": torch.zeros(4)}, str(bad))
    with pytest.raises(ModelFormatError):
        L.verify_assets(dataclasses.replace(assets, checkpoint=bad))


def test_build_stable_audio_model_end_to_end(L, tiny_assets, torch_stack):
    assets, reference = tiny_assets
    sa, report = L.build_stable_audio_model(assets, CPU)
    assert report.matched == report.total_expected
    for k, v in reference.state_dict().items():
        assert torch_stack.equal(sa.model.state_dict()[k], v), k
    audio = sa.generate(prompt="a dog barking", duration=2, steps=2, seed=3, sampler_type="pingpong",
                        disable_tqdm=True)
    assert tuple(audio.shape) == (1, 2, 88200) and torch_stack.isfinite(audio).all()


def test_stdout_stays_clean(L, tiny_assets, capsys):
    L.build_stable_audio_model(tiny_assets[0], CPU)
    assert capsys.readouterr().out == ""  # upstream prints go to stderr


# ------------------------------------------------------------------ engine
def make_engine(tiny_model, log=None):
    from localaudio.engines.sa3_engine import Sa3Engine

    sa, _cfg = tiny_model
    engine = Sa3Engine(ModelPaths("/nonexistent"), CPU, "small-sfx", log=log or (lambda m: None), progress=False)
    engine._sa, engine._loaded = sa, True
    return engine


def test_engine_single_clip_and_seeds(tiny_model):
    engine = make_engine(tiny_model)
    job = SoundJob("door slam", "sfx", duration=3.0, steps=3, seed=5)
    a, b = engine.generate(job), engine.generate(job)
    assert a.audio.shape == (2, 132300) and a.sample_rate == 44100 and a.info["seed"] == 5
    assert a.info["sampler"] == "pingpong" and a.info["steps"] == 3
    np.testing.assert_allclose(a.audio, b.audio)
    other = engine.generate(SoundJob("door slam", "sfx", duration=3.0, steps=3, seed=6))
    assert not np.allclose(a.audio, other.audio)
    auto = engine.generate(SoundJob("door slam", "sfx", duration=1.0, steps=2))
    assert isinstance(auto.info["seed"], int)  # random seed is reported so it can be reproduced


def test_engine_defaults_match_spec(tiny_model):
    job = SoundJob("x", "music")
    assert (job.steps, job.cfg_scale) == (8, 1.0) and C.SA3_SAMPLER == "pingpong"
    assert C.SA3_MAX_SECONDS == 120.0 and C.MIN_VRAM_GB_SA3 == 3.5


def test_engine_cfg_and_negative_prompt(tiny_model):
    engine = make_engine(tiny_model)
    r = engine.generate(SoundJob("door", "sfx", duration=2, steps=2, seed=1, cfg_scale=3.0, negative_prompt="music"))
    assert r.info["negative_prompt"] == "music" and np.isfinite(r.audio).all()
    ignored = engine.generate(SoundJob("door", "sfx", duration=2, steps=2, seed=1, negative_prompt="music"))
    assert ignored.info["negative_prompt"] is None  # no effect at CFG 1.0, so it is not claimed


def test_engine_loopable_trims_crossfade(tiny_model):
    engine = make_engine(tiny_model)
    r = engine.generate(SoundJob("rain", "sfx", duration=6.0, steps=2, seed=2, loopable=True, crossfade_seconds=1.0))
    assert r.audio.shape[1] == int(5.0 * 44100) and r.info["loopable"] is True


def test_engine_rejects_wrong_mode(tiny_model):
    engine = make_engine(tiny_model)
    with pytest.raises(ValueError):
        engine.generate(SoundJob("lofi", "music", duration=2))


def test_extension_by_continuation_uses_inpainting_correctly(tiny_model):
    engine = make_engine(tiny_model)
    calls = []
    real = engine._sa.generate

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    engine._sa.generate = spy
    try:
        job = SoundJob("pad", "sfx", duration=250.0, steps=2, seed=3, context_seconds=30, crossfade_seconds=2.0)
        r = engine.generate(job)
    finally:
        engine._sa.generate = real
    assert r.audio.shape[1] == int(250 * 44100) and r.info["extension"] == "continue"
    assert np.isfinite(r.audio).all()
    assert len(calls) == 3  # 120 s base + two continuations (90 s each, 30 s context)
    assert "inpaint_audio" not in calls[0] and calls[0]["duration"] == 120.0
    for step, kw in enumerate(calls[1:], 1):
        sr, context = kw["inpaint_audio"]
        assert sr == 44100 and tuple(context.shape)[0] == 2
        assert context.shape[1] % C.SA3_LATENT_HOP == 0  # context boundary sits on the latent grid
        assert kw["inpaint_mask_start_seconds"] == pytest.approx(context.shape[1] / 44100)
        assert kw["duration"] <= 120.0 + 1e-6
        assert kw["inpaint_mask_end_seconds"] == pytest.approx(kw["duration"])
        assert kw["seed"] == 3 + step  # each continuation has its own deterministic seed
        assert kw["sampler_type"] == "pingpong" and kw["cfg_scale"] == 1.0 and kw["steps"] == 2
    assert calls[1]["inpaint_mask_start_seconds"] == pytest.approx(29.99, abs=0.1)


def test_extension_by_looping(tiny_model):
    engine = make_engine(tiny_model)
    r = engine.generate(SoundJob("pad", "sfx", duration=130.0, steps=2, seed=3, extend_method="loop"))
    assert r.audio.shape[1] == int(130 * 44100) and r.info["extension"] == "loop"


def test_unload_returns_vram_and_resets_text_encoder_flag(tiny_model):
    engine = make_engine(tiny_model)
    engine.generate(SoundJob("a", "sfx", duration=1, steps=2, seed=1))
    cond = engine._sa.model.conditioner.conditioners["prompt"]
    assert cond._device_initialized is True
    engine._text_encoder_to_cpu(engine._sa)
    assert cond._device_initialized is False  # next forward re-moves it to the compute device
    sa = engine._sa
    engine.unload()
    assert engine._sa is None and not engine.loaded
    assert next(sa.model.parameters()).device.type == "cpu"
    engine._sa, engine._loaded = sa, True  # the fixture is shared; restore


def test_engine_load_reports_missing_files(tmp_path):
    from localaudio.engines.sa3_engine import Sa3Engine

    engine = Sa3Engine(ModelPaths(tmp_path), CPU, "small-music", log=lambda m: None)
    with pytest.raises(ModelsMissingError, match="setup_models.py --only music"):
        engine.load()


def test_engine_load_from_assets_end_to_end(tiny_assets, monkeypatch):
    from localaudio.engines.sa3_engine import Sa3Engine

    assets, _ = tiny_assets
    paths = ModelPaths("/nonexistent")
    monkeypatch.setattr(ModelPaths, "sa3_assets", lambda self, variant, source="auto": assets)
    engine = Sa3Engine(paths, CPU, "small-sfx", log=lambda m: None, progress=False)
    engine.load()
    r = engine.generate(SoundJob("a dog barking", "sfx", duration=2.0, steps=2, seed=1))
    assert r.audio.shape == (2, 88200) and r.info["weights"] == "comfy"
    engine.unload()
