import numpy as np
import pytest

from localaudio.engines.kokoro_engine import parse_voice_spec, resolve_voice_files
from localaudio.errors import RequestError


def test_parse_voice_spec():
    assert parse_voice_spec("af_heart") == [("af_heart", 1.0)]
    assert parse_voice_spec("af_heart:0.7, af_bella:0.3") == [("af_heart", 0.7), ("af_bella", 0.3)]
    assert parse_voice_spec("af_heart,af_bella") == [("af_heart", 1.0), ("af_bella", 1.0)]
    assert parse_voice_spec("/some/dir/custom.pt") == [("/some/dir/custom.pt", 1.0)]


@pytest.mark.parametrize("bad", ["", "  ", "../../etc/passwd", "af heart", "af_heart:abc", "af_heart:0", "af_heart:-1"])
def test_parse_voice_spec_rejects_bad_input(bad):
    with pytest.raises(RequestError):
        parse_voice_spec(bad)


def test_resolve_voice_files_and_suggestions(tmp_path):
    (tmp_path / "af_heart.pt").write_bytes(b"x")
    (tmp_path / "af_bella.pt").write_bytes(b"x")
    files = resolve_voice_files("af_heart:2,af_bella", tmp_path, ["af_heart", "af_bella"])
    assert [(p.name, w) for p, w in files] == [("af_heart.pt", 2.0), ("af_bella.pt", 1.0)]
    with pytest.raises(RequestError, match="Did you mean: af_heart"):
        resolve_voice_files("af_hart", tmp_path, ["af_heart", "af_bella"])
    custom = tmp_path / "mine.pt"
    custom.write_bytes(b"x")
    assert resolve_voice_files(str(custom), tmp_path)[0][0] == custom


def test_engine_reports_missing_models(tmp_path):
    from localaudio.device import DevicePlan
    from localaudio.engines.kokoro_engine import KokoroEngine
    from localaudio.errors import ModelsMissingError
    from localaudio.models import ModelPaths

    engine = KokoroEngine(ModelPaths(tmp_path), DevicePlan("cpu", "float32", "t"), log=lambda m: None)
    with pytest.raises(ModelsMissingError, match="setup_models.py --only tts"):
        engine.load()


@pytest.mark.slow
def test_end_to_end_with_random_weights(tmp_path):
    """Real kokoro + misaki + spaCy + espeak code, random (but correctly shaped) weights."""
    pytest.importorskip("torch")
    pytest.importorskip("kokoro")
    spacy_util = pytest.importorskip("spacy.util")
    if not spacy_util.is_package("en_core_web_sm"):
        pytest.skip("spaCy model en_core_web_sm not installed")
    sf = pytest.importorskip("soundfile")
    import fake_kokoro

    from localaudio.studio import Studio

    fake_kokoro.make(tmp_path)
    with Studio(str(tmp_path), device="cpu", quiet=True) as studio:
        r = studio.tts("Hello there.", voice="af_heart", seed=1, output=str(tmp_path / "a.wav"))
        assert r.sample_rate == 24000 and r.channels == 1 and r.duration > 0 and r.info["lang"] == "a"
        data, sr = sf.read(r.path)
        assert sr == 24000 and np.isfinite(data).all() and np.abs(data).max() == pytest.approx(0.891, abs=0.01)

        blend = studio.tts("Hi.", voice="af_heart:0.7,af_bella:0.3", speed=1.25, output=str(tmp_path / "b.wav"))
        assert blend.info["speed"] == 1.25 and blend.duration > 0

        # two paragraphs -> several chunks; pauses make the file longer than the speech alone
        text = "First paragraph here.\n\nSecond paragraph here."
        spaced = studio.tts(text, voice="af_heart", paragraph_pause=2.0, output=str(tmp_path / "c.wav"))
        tight = studio.tts(text, voice="af_heart", paragraph_pause=0.0, output=str(tmp_path / "d.wav"))
        assert spaced.info["chunks"] == 2
        assert spaced.duration - tight.duration == pytest.approx(2.0, abs=0.05)

        hindi = studio.tts("नमस्ते।", voice="hf_alpha", output=str(tmp_path / "h.wav"))
        assert hindi.info["lang"] == "h" and hindi.duration > 0

        with pytest.raises(RequestError, match="not found"):
            studio.tts("x", voice="af_nope", output=str(tmp_path / "e.wav"))
