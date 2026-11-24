import json
from pathlib import Path

import pytest

from localaudio import cli
from localaudio import config as C
from localaudio import installer as I
from localaudio.errors import ModelFormatError, RequestError
from localaudio.models import ModelPaths


def run_cli(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


# ------------------------------------------------------------------- CLI
def test_help_lists_all_spec_flags(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    text = capsys.readouterr().out
    for flag in ("--mode", "--prompt", "--text", "--voice", "--speed", "--duration", "--steps", "--seed",
                 "--device", "--output", "--dry-run", "--list-voices", "--batch", "--loopable"):
        assert flag in text


def test_dry_run_sfx_shows_enriched_prompt_and_routing(capsys, tmp_path):
    code, out, _ = run_cli(capsys, "--mode", "sfx", "--prompt", "rain on a tin roof", "--duration", "8",
                           "--dry-run", "--device", "cpu", "--models-dir", str(tmp_path), "-o", "x.wav")
    assert code == 0
    plan = json.loads(out)
    assert plan["engine"] == "Stable Audio 3 small-sfx"
    assert plan["sound"]["sampler"] == "pingpong" and plan["sound"]["steps"] == 8 and plan["sound"]["cfg_scale"] == 1.0
    assert plan["sound"]["prompt"].startswith("TrackType: SFX, rain on a tin roof")
    assert plan["enrichment"]["category"] == "ambience_nature"
    assert plan["device"]["device"] == "cpu" and plan["device"]["dtype"] == "float32"
    assert plan["models"]["ready"] is False


def test_dry_run_music_default_duration_and_no_enrich(capsys, tmp_path):
    code, out, _ = run_cli(capsys, "--mode", "music", "--prompt", "lofi beat", "--no-enrich", "--dry-run",
                           "--device", "cpu", "--models-dir", str(tmp_path))
    plan = json.loads(out)
    assert code == 0 and plan["sound"]["duration"] == 30.0
    assert plan["sound"]["prompt"] == "lofi beat" and "enrichment" not in plan


def test_dry_run_music_extension_plan(capsys, tmp_path):
    code, out, _ = run_cli(capsys, "--mode", "music", "--prompt", "score", "--duration", "400", "--dry-run",
                           "--device", "cpu", "--models-dir", str(tmp_path))
    plan = json.loads(out)
    assert plan["extension"]["method"] == "continue" and plan["extension"]["model_calls"] >= 4


def test_dry_run_tts_and_text_file(capsys, tmp_path):
    script = tmp_path / "script.txt"
    script.write_text("First sentence. Second sentence.\n\nA new paragraph.", encoding="utf-8")
    code, out, _ = run_cli(capsys, "--mode", "tts", "--text-file", str(script), "--voice", "hf_alpha",
                           "--dry-run", "--device", "cpu", "--models-dir", str(tmp_path))
    plan = json.loads(out)
    assert code == 0 and plan["speech"]["lang"] == "h" and plan["speech"]["chunks"] >= 2
    assert plan["speech"]["sample_rate"] == 24000


@pytest.mark.parametrize(
    "argv, code",
    [
        ([], 2),  # no mode
        (["--mode", "tts"], 2),  # empty text
        (["--mode", "sfx"], 2),  # no prompt
        (["--mode", "sfx", "--prompt", "x", "--steps", "13"], 2),
        (["--mode", "sfx", "--prompt", "x", "--steps", "0"], 2),
        (["--mode", "tts", "--text", "hi", "--speed", "9"], 2),
        (["--mode", "music", "--prompt", "x", "--duration", "0.1"], 2),
    ],
)
def test_invalid_requests_exit_2(capsys, tmp_path, argv, code):
    got, _, err = run_cli(capsys, *argv, "--device", "cpu", "--models-dir", str(tmp_path), "--dry-run")
    assert got == code and err.startswith("error:")


def test_missing_models_gives_actionable_error(capsys, tmp_path):
    got, out, err = run_cli(capsys, "--mode", "sfx", "--prompt", "door", "--device", "cpu",
                            "--models-dir", str(tmp_path), "--quiet")
    assert got == 3 and out == ""
    assert "setup_models.py --only sfx" in err


def test_list_voices(capsys, tmp_path):
    code, out, _ = run_cli(capsys, "--list-voices", "--json", "--models-dir", str(tmp_path))
    rows = json.loads(out)
    assert code == 0 and len(rows) == 54
    assert next(r for r in rows if r["voice"] == "hf_alpha")["language"] == "Hindi"
    assert not any(r["installed"] for r in rows)


def test_seed_negative_means_random(capsys, tmp_path):
    code, out, _ = run_cli(capsys, "--mode", "sfx", "--prompt", "x", "--seed", "-1", "--dry-run",
                           "--device", "cpu", "--models-dir", str(tmp_path))
    assert json.loads(out)["sound"]["seed"] == "random"
    code, out, _ = run_cli(capsys, "--mode", "sfx", "--prompt", "x", "--seed", "42", "--dry-run",
                           "--device", "cpu", "--models-dir", str(tmp_path))
    assert json.loads(out)["sound"]["seed"] == 42


def test_doctor_json(capsys, tmp_path):
    code, out, _ = run_cli(capsys, "--doctor", "--json", "--models-dir", str(tmp_path))
    report = json.loads(out)
    names = {c["name"] for c in report["checks"]}
    assert {"python", "models:tts", "models:sfx", "models:music"} <= names
    assert code in (0, 1)


# -------------------------------------------------------------- installer
def test_parse_only():
    assert I.parse_only(None) == ["tts", "sfx", "music"]
    assert I.parse_only(["all"]) == ["tts", "sfx", "music"]
    assert I.parse_only(["sfx,tts"]) == ["sfx", "tts"]
    assert I.parse_only(["music", "music"]) == ["music"]
    assert I.parse_only(["speech", "foley"]) == ["tts", "sfx"]
    with pytest.raises(RequestError):
        I.parse_only(["video"])


def test_plan_reflects_installed_files(tmp_path):
    paths = ModelPaths(tmp_path)
    rows = I.build_plan(["sfx", "music"], "comfy", paths)
    assert [r.what.split(" (")[0] for r in rows][:2] == ["Stable Audio 3 small-sfx checkpoint",
                                                           "Stable Audio 3 small-music checkpoint"]
    assert not any(r.installed for r in rows) and any("T5Gemma" in r.what for r in rows)
    paths.comfy_checkpoint("small-sfx").parent.mkdir(parents=True)
    paths.comfy_checkpoint("small-sfx").write_bytes(b"x")
    rows = I.build_plan(["sfx"], "comfy", paths)
    assert rows[0].installed and not rows[1].installed
    official = I.build_plan(["music"], "official", paths)
    assert len(official) == 1 and "gated" in official[0].what


def test_dry_run_prints_total(capsys, tmp_path):
    code = I.main(["--dry-run", "--models-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0 and "still to download" in out and "free disk space" in out


def test_human_sizes():
    assert I.human(500) == "500 B" and I.human(5_000) == "5 kB" and I.human(327_000_000) == "327 MB"
    assert I.human(2_270_000_000) == "2.27 GB"


def test_prompt_groups_menu():
    assert I.prompt_groups([], input_fn=lambda _p: "") == ["tts", "sfx", "music"]
    assert I.prompt_groups([], input_fn=lambda _p: "t, s") == ["tts", "sfx"]
    assert I.prompt_groups([], input_fn=lambda _p: "m") == ["music"]
    assert I.prompt_groups([], input_fn=lambda _p: "q") == []


def test_download_url_verifies_hash(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"hello tokenizer")
    digest = I.sha256_file(src)
    dest = I.download_url(src.as_uri(), tmp_path / "dl" / "tok.model", digest, log=lambda m: None)
    assert dest.read_bytes() == b"hello tokenizer"
    with pytest.raises(ModelFormatError):
        I.download_url(src.as_uri(), tmp_path / "dl" / "bad.model", "0" * 64, log=lambda m: None)
    assert not (tmp_path / "dl" / "bad.model").exists() and not list((tmp_path / "dl").glob("*.part"))


def test_pinned_tokenizer_constants_are_sane():
    assert len(C.TOKENIZER_SHA256) == 64 and C.SA3_COMMIT in C.TOKENIZER_URL
    assert C.TOKENIZER_URL.startswith("https://raw.githubusercontent.com/Stability-AI/stable-audio-3/")


class FakeHub:
    """Records downloads and materialises files so the rest of the flow can proceed."""

    def __init__(self, monkeypatch):
        self.calls = []
        monkeypatch.setattr(I, "hub_snapshot", self.snapshot)
        monkeypatch.setattr(I, "hub_download", self.download)
        monkeypatch.setattr(I, "download_url", self.url)
        monkeypatch.setattr(I, "ensure_spacy_model", lambda log=print: True)

    def snapshot(self, repo, dest_dir, revision=None, token=None, allow_patterns=None):
        self.calls.append(("snapshot", repo, revision, token, tuple(allow_patterns or ())))
        (Path(dest_dir) / "voices").mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "kokoro-v1_0.pth", "voices/af_heart.pt"):
            (Path(dest_dir) / name).write_bytes(b"x")
        return Path(dest_dir)

    def download(self, repo, filename, dest_dir, revision=None, token=None):
        self.calls.append(("file", repo, filename, token))
        target = Path(dest_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        return target

    def url(self, url, dest, sha256=None, log=print):
        self.calls.append(("url", url, sha256))
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(b"x")
        return Path(dest)


def test_install_tts_only_pulls_kokoro(monkeypatch, tmp_path):
    hub = FakeHub(monkeypatch)
    code = I.main(["--only", "tts", "--models-dir", str(tmp_path), "--yes", "--no-verify"], log=lambda m: None)
    assert code == 0
    kind, repo, revision, token, patterns = hub.calls[0]
    assert (kind, repo, revision, token) == ("snapshot", C.KOKORO_REPO, C.KOKORO_REVISION, None)
    assert set(patterns) == {"config.json", "kokoro-v1_0.pth", "voices/*.pt"}
    assert len(hub.calls) == 1
    assert ModelPaths(tmp_path).missing("tts") == []


def test_install_sfx_uses_comfy_files_without_any_token(monkeypatch, tmp_path):
    hub = FakeHub(monkeypatch)
    code = I.main(["--only", "sfx", "--models-dir", str(tmp_path), "--yes", "--no-verify",
                   "--hf-token", "hf_should_not_be_used"], log=lambda m: None)
    assert code == 0
    files = [c for c in hub.calls if c[0] == "file"]
    assert [c[2] for c in files] == ["checkpoints/stable_audio_3_small_sfx.safetensors",
                                     "text_encoders/t5gemma_b_b_ul2.safetensors"]
    assert all(c[1] == C.COMFY_REPO and c[3] is None for c in files)  # ungated: token is never sent
    assert ("url", C.TOKENIZER_URL, C.TOKENIZER_SHA256) in hub.calls
    assert not any("music" in str(c) for c in hub.calls)  # selective download
    assert ModelPaths(tmp_path).missing("sfx") == []


def test_install_all_downloads_shared_encoder_once(monkeypatch, tmp_path):
    hub = FakeHub(monkeypatch)
    assert I.main(["--models-dir", str(tmp_path), "--yes", "--no-verify"], log=lambda m: None) == 0
    names = [c[2] for c in hub.calls if c[0] == "file"]
    assert names.count("text_encoders/t5gemma_b_b_ul2.safetensors") == 1
    assert len(names) == 3
    paths = ModelPaths(tmp_path)
    assert all(not paths.missing(m) for m in C.MODES)


def test_install_skips_existing_files(monkeypatch, tmp_path):
    hub = FakeHub(monkeypatch)
    args = ["--only", "music", "--models-dir", str(tmp_path), "--yes", "--no-verify"]
    I.main(args, log=lambda m: None)
    hub.calls.clear()
    I.main(args, log=lambda m: None)
    assert hub.calls == []
    I.main(args + ["--force"], log=lambda m: None)
    assert len([c for c in hub.calls if c[0] == "file"]) == 2


def test_official_source_passes_token(monkeypatch, tmp_path):
    hub = FakeHub(monkeypatch)
    I.main(["--only", "music", "--source", "official", "--hf-token", "hf_abc", "--models-dir", str(tmp_path),
            "--yes", "--no-verify"], log=lambda m: None)
    kind, repo, revision, token, _ = hub.calls[0]
    assert repo == "stabilityai/stable-audio-3-small-music" and token == "hf_abc"


def test_gated_error_is_translated():
    class GatedRepoError(Exception):
        pass

    err = I._translate_hub_errors(GatedRepoError("401 Client Error"), "stabilityai/x")
    assert "gated" in str(err) and "--hf-token" in str(err)
    assert "not found" in str(I._translate_hub_errors(Exception("404 not found"), "a/b"))


def test_check_installation(tmp_path):
    lines = []
    assert I.check_installation(ModelPaths(tmp_path), log=lines.append, deep=False) is False
    assert any("[miss] sfx" in line for line in lines)
