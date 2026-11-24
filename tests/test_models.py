from pathlib import Path

import pytest

from localaudio import config as C
from localaudio.models import ModelPaths


def touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def install_comfy(paths: ModelPaths, *variants: str):
    for v in variants:
        touch(paths.comfy_checkpoint(v))
    touch(paths.comfy_t5_weights)
    touch(paths.tokenizer_model)


def install_official(paths: ModelPaths, variant: str):
    d = paths.official_dir(variant)
    touch(d / "model_config.json", b"{}")
    touch(d / "model.safetensors")
    touch(d / "t5gemma-b-b-ul2" / "config.json", b"{}")


def test_default_models_dir_priority(monkeypatch, tmp_path):
    monkeypatch.setenv(C.ENV_MODELS_DIR, str(tmp_path / "env"))
    assert C.default_models_dir() == (tmp_path / "env").resolve()
    assert C.default_models_dir(str(tmp_path / "arg")) == (tmp_path / "arg").resolve()
    monkeypatch.delenv(C.ENV_MODELS_DIR)
    assert C.default_models_dir().name == "models"  # source checkout -> <repo>/models


def test_layout(tmp_path):
    p = ModelPaths(tmp_path)
    assert p.kokoro_weights == tmp_path / "kokoro" / "kokoro-v1_0.pth"
    assert p.comfy_checkpoint("small-sfx").name == "stable_audio_3_small_sfx.safetensors"
    assert p.comfy_t5_weights == tmp_path / "comfy" / "text_encoders" / "t5gemma_b_b_ul2.safetensors"
    assert p.voice_path("af_heart").name == "af_heart.pt"


def test_empty_dir_reports_everything_missing(tmp_path):
    p = ModelPaths(tmp_path)
    status = p.status()
    assert not any(status[m]["ready"] for m in C.MODES)
    assert len(p.missing("sfx")) == 3  # checkpoint, text encoder, tokenizer
    assert any("voices" in m for m in p.missing("tts"))


def test_tts_ready_and_voices(tmp_path):
    p = ModelPaths(tmp_path)
    touch(p.kokoro_config)
    touch(p.kokoro_weights)
    touch(p.voice_path("af_heart"))
    touch(p.voice_path("bm_george"))
    assert p.missing("tts") == []
    assert p.installed_voices() == ["af_heart", "bm_george"]


def test_comfy_assets_need_all_three_files(tmp_path):
    p = ModelPaths(tmp_path)
    touch(p.comfy_checkpoint("small-sfx"))
    assert p.sa3_assets("small-sfx") is None
    touch(p.comfy_t5_weights)
    assert p.sa3_assets("small-sfx") is None
    touch(p.tokenizer_model)
    a = p.sa3_assets("small-sfx")
    assert a and a.source == "comfy" and a.t5_weights == p.comfy_t5_weights
    assert a.config_json.name == "sa3_small_sfx_config.json" and a.config_json.is_file()
    assert a.t5_config.is_file()
    assert p.sa3_assets("small-music") is None  # other variant not installed


def test_music_and_sfx_share_encoder_but_not_checkpoints(tmp_path):
    p = ModelPaths(tmp_path)
    install_comfy(p, "small-sfx", "small-music")
    assert p.sa3_assets("small-sfx").checkpoint != p.sa3_assets("small-music").checkpoint
    assert p.sa3_assets("small-sfx").t5_weights == p.sa3_assets("small-music").t5_weights


def test_official_bundle_is_preferred_in_auto(tmp_path):
    p = ModelPaths(tmp_path)
    install_comfy(p, "small-music")
    install_official(p, "small-music")
    assert p.sa3_assets("small-music", "auto").source == "official"
    assert p.sa3_assets("small-music", "comfy").source == "comfy"
    assert p.sa3_assets("small-music", "official").source == "official"
    assert p.sa3_assets("small-sfx", "official") is None


def test_status_reports_source(tmp_path):
    p = ModelPaths(tmp_path)
    install_comfy(p, "small-sfx")
    st = p.status()
    assert st["sfx"] == {"ready": True, "missing": [], "source": "comfy"}
    assert st["music"]["ready"] is False and st["music"]["source"] is None


def test_unknown_variant(tmp_path):
    with pytest.raises(KeyError):
        ModelPaths(tmp_path).sa3_assets("medium")


def test_lang_for_voice():
    assert C.lang_for_voice("af_heart") == "a"
    assert C.lang_for_voice("hf_alpha") == "h"
    assert C.lang_for_voice("/x/y/zm_yunxi.pt") == "z"


def test_catalog_integrity():
    assert len(C.KOKORO_VOICES) == 54 and len(set(C.KOKORO_VOICES)) == 54
    assert C.DEFAULT_VOICE in C.KOKORO_VOICES
    assert {v[0] for v in C.KOKORO_VOICES} <= set(C.KOKORO_LANGS)
    assert all(v[1] in "fm" and v[2] == "_" for v in C.KOKORO_VOICES)
    assert (C.ASSETS_DIR / C.T5_CONFIG_ASSET).is_file()
    assert all((C.ASSETS_DIR / v.config_asset).is_file() for v in C.SA3_VARIANTS.values())
