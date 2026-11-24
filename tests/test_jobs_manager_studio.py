import json

import numpy as np
import pytest

from localaudio import config as C
from localaudio.engines.base import AudioResult, Engine
from localaudio.errors import ModelsMissingError, RequestError
from localaudio.jobs import SoundJob, SpeechJob, job_from_dict
from localaudio.manager import ModelManager
from localaudio.models import ModelPaths
from localaudio.studio import PostOptions, Studio, apply_post, read_batch_file

SR = 24000


# ------------------------------------------------------------------ jobs
def test_speech_job_validation():
    assert SpeechJob("hello").validate()
    for bad in (SpeechJob("  "), SpeechJob("hi", speed=0.1), SpeechJob("hi", speed=3.0),
                SpeechJob("hi", max_chars=5), SpeechJob("hi", sentence_pause=-1)):
        with pytest.raises(RequestError):
            bad.validate()


def test_sound_job_validation():
    assert SoundJob("rain", "sfx", duration=5).validate()
    bad = [
        SoundJob("", "sfx"), SoundJob("x", "tts"), SoundJob("x", "sfx", duration=0.1),
        SoundJob("x", "sfx", steps=0), SoundJob("x", "sfx", steps=13), SoundJob("x", "sfx", cfg_scale=-1),
        SoundJob("x", "sfx", extend_method="warp"), SoundJob("x", "sfx", context_seconds=1),
        SoundJob("x", "sfx", crossfade_seconds=-1), SoundJob("x", "sfx", duration=99999),
    ]
    for job in bad:
        with pytest.raises(RequestError):
            job.validate()


def test_steps_up_to_twelve_allowed_and_default_is_eight():
    assert SoundJob("x", "music").steps == 8
    assert SoundJob("x", "music", steps=12).validate().steps == 12


def test_extension_threshold_is_120_seconds():
    assert not SoundJob("x", "music", duration=120).needs_extension
    assert SoundJob("x", "music", duration=120.5).needs_extension


def test_job_from_dict_aliases_and_unknown_keys():
    j = job_from_dict("sfx", {"text": "rain", "duration": 4, "output": "a.wav", "enrich": False})
    assert j.prompt == "rain" and j.duration == 4 and j.mode == "sfx"
    t = job_from_dict("tts", {"prompt": "hello", "voice": "af_bella"})
    assert t.text == "hello" and t.voice == "af_bella"
    with pytest.raises(RequestError):
        job_from_dict("sfx", {"prompt": "x", "bogus": 1})


# ------------------------------------------------------------------ manager
class FakeEngine(Engine):
    events = []

    def __init__(self, key, plan, fail_load=False):
        super().__init__(ModelPaths("/tmp/none"), plan, log=lambda m: None)
        self.key = key
        self.display_name = f"fake-{key}"
        self.fail_load = fail_load

    def load(self):
        FakeEngine.events.append(f"load:{self.key}")
        if self.fail_load:
            raise ModelsMissingError("missing")
        self._loaded = True

    def unload(self):
        FakeEngine.events.append(f"unload:{self.key}")
        self._loaded = False

    def generate(self, job):
        t = np.linspace(0, 1, SR if self.key == "tts" else 44100, dtype=np.float32)
        wave = (0.2 * np.sin(2 * np.pi * 220 * t))[None, :]
        if self.key != "tts":
            wave = np.vstack([wave, wave])
        return AudioResult(wave, SR if self.key == "tts" else 44100, {"engine": self.display_name})


@pytest.fixture
def manager(fake_torch):
    FakeEngine.events = []
    fail = {"music": False}

    def factory(key, plan):
        return FakeEngine(key, plan, fail_load=fail.get(key, False))

    mgr = ModelManager(ModelPaths("/tmp/none"), device="cpu", engine_factory=factory,
                       log=lambda m: None, cleanup=lambda: FakeEngine.events.append("cleanup"))
    mgr._fail = fail
    return mgr


def test_switching_unloads_then_cleans_then_loads(manager):
    manager.use("tts")
    manager.use("sfx")
    assert FakeEngine.events == ["load:tts", "unload:tts", "cleanup", "load:sfx"]
    assert manager.current == "sfx"


def test_same_model_is_reused(manager):
    a = manager.use("sfx")
    b = manager.use("sfx")
    assert a is b and FakeEngine.events == ["load:sfx"]


def test_release_cleans_up_and_is_idempotent(manager):
    manager.use("music")
    manager.release()
    manager.release()
    assert FakeEngine.events == ["load:music", "unload:music", "cleanup"]
    assert manager.current is None


def test_failed_load_still_cleans_up(manager):
    manager._fail["music"] = True
    with pytest.raises(ModelsMissingError):
        manager.use("music")
    assert FakeEngine.events[-2:] == ["unload:music", "cleanup"] and manager.current is None


def test_context_manager_releases(manager):
    with manager as m:
        m.use("tts")
    assert FakeEngine.events[-2:] == ["unload:tts", "cleanup"]


def test_unknown_mode_rejected(manager):
    with pytest.raises(RequestError):
        manager.use("video")


def test_plan_for_uses_per_engine_policy(fake_torch, monkeypatch):
    import localaudio.device as D

    monkeypatch.setattr(D, "import_torch", lambda: fake_torch(total_gib=4.0, free_gib=3.9))
    mgr = ModelManager(ModelPaths("/tmp/none"), device="auto", log=lambda m: None)
    assert mgr.plan_for("sfx").dtype == "float16"  # Stable Audio: fp16 on CUDA
    assert mgr.plan_for("tts").dtype == "float32"  # Kokoro always fp32
    assert mgr.plan_for("music").device == "cuda"


# ------------------------------------------------------------------ studio
@pytest.fixture
def studio(manager, tmp_path):
    return Studio(str(tmp_path / "models"), manager=manager, quiet=True)


def test_post_processing_pipeline():
    x = np.full((2, 44100), 0.1, dtype=np.float32)
    y, sr = apply_post(x, 44100, PostOptions(normalize_db=-6.0, edge_fade_ms=5), "sfx")
    assert sr == 44100 and np.abs(y).max() == pytest.approx(10 ** (-6 / 20), rel=1e-3)
    assert y[0, 0] == pytest.approx(0.0, abs=1e-6)  # click guard fade-in
    y2, _ = apply_post(x, 44100, PostOptions(normalize_db=None, edge_fade_ms=5), "sfx", loopable=True)
    assert y2[0, 0] == pytest.approx(0.1)  # loopable audio must not be faded at the edges
    y3, _ = apply_post(x, 44100, PostOptions(normalize_db=None), "tts")
    assert y3[0, 0] == pytest.approx(0.1)  # speech has no edge fade


def test_studio_writes_files_and_reports(studio, tmp_path):
    sf = pytest.importorskip("soundfile")
    r = studio.sfx("door slam", duration=1, output=str(tmp_path / "out" / "slam.wav"))
    assert r.path.exists() and r.channels == 2 and r.sample_rate == 44100
    assert r.info["enrichment"]["category"] in ("impact_cinematic", "foley", "general")
    assert "TrackType: SFX" in r.info["enrichment"]["prompt"]
    data, sr = sf.read(r.path)
    assert sr == 44100 and data.shape[1] == 2


def test_studio_no_enrich_keeps_prompt_exact(studio, tmp_path):
    r = studio.music("my exact words", duration=1, enrich=False, output=str(tmp_path / "m.wav"))
    assert "enrichment" not in r.info


def test_batch_groups_by_model_and_keeps_order(studio, tmp_path):
    specs = [
        {"mode": "music", "prompt": "lofi", "duration": 1, "output": str(tmp_path / "m.wav")},
        {"mode": "tts", "text": "hello", "output": str(tmp_path / "t.wav")},
        {"mode": "sfx", "prompt": "rain", "duration": 1, "output": str(tmp_path / "s.wav")},
        {"mode": "tts", "text": "again", "output": str(tmp_path / "t2.wav")},
    ]
    results = studio.batch(specs)
    assert [r["index"] for r in results] == [0, 1, 2, 3]
    assert all(r["ok"] for r in results)
    loads = [e for e in FakeEngine.events if e.startswith("load")]
    assert loads == ["load:tts", "load:sfx", "load:music"]  # one load per model, tts first
    assert results[1]["result"]["path"].endswith("t.wav")


def test_batch_reports_errors_per_job(studio, tmp_path):
    specs = [
        {"mode": "tts", "text": "", "output": str(tmp_path / "bad.wav")},
        {"mode": "tts", "text": "fine", "output": str(tmp_path / "ok.wav")},
    ]
    results = studio.batch(specs)
    assert results[0]["ok"] is False and "empty" in results[0]["error"]
    assert results[1]["ok"] is True
    with pytest.raises(RequestError):
        studio.batch([{"mode": "video"}])


def test_read_batch_file_formats(tmp_path):
    jobs = [{"mode": "tts", "text": "a"}, {"mode": "sfx", "prompt": "b"}]
    (tmp_path / "list.json").write_text(json.dumps(jobs))
    (tmp_path / "obj.json").write_text(json.dumps({"jobs": jobs}))
    (tmp_path / "lines.jsonl").write_text("\n".join(json.dumps(j) for j in jobs))
    for name in ("list.json", "obj.json", "lines.jsonl"):
        assert read_batch_file(str(tmp_path / name)) == jobs
    (tmp_path / "bad.json").write_text("not json at all {")
    with pytest.raises(RequestError):
        read_batch_file(str(tmp_path / "bad.json"))
    (tmp_path / "empty.json").write_text("")
    with pytest.raises(RequestError):
        read_batch_file(str(tmp_path / "empty.json"))


def test_plan_describes_without_loading(studio):
    job = SoundJob("rain", "sfx", duration=300, steps=8, context_seconds=30).validate()
    plan = studio.plan("sfx", job)
    assert plan["engine"] == "Stable Audio 3 small-sfx"
    assert plan["extension"]["method"] == "continue" and plan["extension"]["model_calls"] == 1 + 2
    assert plan["sound"]["sampler"] == "pingpong" and plan["sound"]["sample_rate"] == 44100
    assert FakeEngine.events == []  # nothing was loaded
    tts = studio.plan("tts", SpeechJob("One sentence. Another one.", voice="af_heart"))
    assert tts["speech"]["lang"] == "a" and tts["speech"]["chunks"] >= 1


def test_batch_accepts_enrichment_options_at_top_level_and_survives_bad_jobs(studio, tmp_path):
    specs = [
        {"mode": "music", "prompt": "soft jazz", "duration": 1, "bpm": 78, "instruments": "felt piano",
         "output": str(tmp_path / "ok.wav")},
        {"mode": "sfx", "prompt": "rain", "duration": 1, "not_a_real_option": 5, "output": str(tmp_path / "bad.wav")},
        {"mode": "tts", "text": "fine", "output": str(tmp_path / "t.wav")},
    ]
    results = studio.batch(specs)
    assert results[0]["ok"] and "78 BPM" in results[0]["result"]["info"]["enrichment"]["prompt"]
    assert results[1]["ok"] is False and "not_a_real_option" in results[1]["error"]
    assert results[2]["ok"] is True  # one malformed job never aborts the rest


def test_batch_dry_run_loads_nothing(studio, tmp_path):
    specs = [{"mode": "sfx", "prompt": "rain on a tin roof", "duration": 3},
             {"mode": "tts", "text": "Hello there."}]
    results = studio.batch(specs, dry_run=True)
    assert all(r["ok"] for r in results) and FakeEngine.events == []
    assert results[0]["plan"]["enrichment"]["category"] == "ambience_nature"
    assert results[1]["plan"]["speech"]["voice"] == C.DEFAULT_VOICE
