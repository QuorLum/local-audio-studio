"""Keep the documentation honest: examples, tables, flags and pins must match the code."""

import re
import tomllib
from pathlib import Path

import pytest

from localaudio import cli
from localaudio import config as C
from localaudio import prompts as P

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


# ------------------------------------------------------------- prompt examples in the docs
SFX_EXAMPLES = {
    "footsteps on wet gravel": "TrackType: SFX, footsteps on wet gravel, close-mic foley recording, dry studio acoustics, "
                               "clean and detailed, no background noise, no music",
    "rain on a tin roof": "TrackType: SFX, rain on a tin roof, binaural outdoor field recording, open-air natural "
                          "acoustics, wide stereo ambience with natural dynamics, no music",
    "laser blast": "TrackType: SFX, laser blast, clean designed sound effect with precise stereo placement, dry, no room "
                   "reverb, crisp high-fidelity studio sound design, no music",
}
MUSIC_EXAMPLES = {
    "calm acoustic background music for a travel vlog":
        "TrackType: Music, VocalType: Instrumental, calm acoustic background music for a travel vlog, warm fingerpicked "
        "acoustic guitar, upright bass, brushed snare, clean studio mix, no vocals, 95 BPM",
    "lofi study beat":
        "TrackType: Music, VocalType: Instrumental, lofi study beat, dusty Rhodes electric piano, mellow upright bass, "
        "lazy boom-bap drums with soft vinyl crackle, relaxed and nostalgic mood, no vocals, 80 BPM",
}


@pytest.mark.parametrize("prompt, expected", SFX_EXAMPLES.items())
def test_sfx_examples_match_code_and_docs(prompt, expected):
    assert P.enrich_sfx(prompt).prompt == expected
    assert expected in read("references/prompting-sfx.md")


@pytest.mark.parametrize("prompt, expected", MUSIC_EXAMPLES.items())
def test_music_examples_match_code_and_docs(prompt, expected):
    assert P.enrich_music(prompt).prompt == expected
    assert expected in read("references/prompting-music.md")


def test_readme_examples_match_code():
    readme = read("README.md")
    for expected in (SFX_EXAMPLES["footsteps on wet gravel"], SFX_EXAMPLES["rain on a tin roof"],
                     MUSIC_EXAMPLES["calm acoustic background music for a travel vlog"]):
        assert expected in readme


def test_every_category_and_genre_is_documented():
    sfx_doc, music_doc = read("references/prompting-sfx.md"), read("references/prompting-music.md")
    for category in list(P._SFX_CATEGORIES) + [P._SFX_DEFAULT]:
        assert f"`{category.name}`" in sfx_doc, category.name
        assert category.mic in sfx_doc and category.room in sfx_doc, category.name
    for genre in list(P._GENRES) + [P._GENRE_DEFAULT]:
        assert f"`{genre.name}`" in music_doc, genre.name
        assert genre.instruments in music_doc, genre.name


def test_every_kokoro_voice_is_in_the_voice_reference():
    text = read("references/tts-voices.md")
    for voice in C.KOKORO_VOICES:
        lang, name = voice.split("_")
        assert name in text, voice


# ------------------------------------------------------------- SKILL.md
def test_skill_frontmatter_and_size():
    text = read("SKILL.md")
    assert text.startswith("---\n")
    front = text.split("---\n")[1]
    assert re.search(r"^name: local-audio-studio$", front, re.M)
    description = re.search(r"^description: (.+)$", front, re.M).group(1)
    assert 200 < len(description) < 1024
    for trigger in ("ElevenLabs", "TTS", "foley", "background music", "voiceover", "offline"):
        assert trigger.lower() in description.lower(), trigger
    assert len(text.splitlines()) < 500


def test_skill_references_existing_files():
    text = read("SKILL.md")
    for rel in set(re.findall(r"(?:references|examples|scripts)/[\w./-]+\.(?:md|json|py|txt)", text)):
        assert (ROOT / rel).is_file(), rel


def test_skill_mentions_required_topics():
    text = read("SKILL.md")
    for needle in ("--mode", "--dry-run", "--doctor", "setup_models.py", "120 s", "gc.collect()",
                   "torch.cuda.empty_cache()", "Powered by Stability AI", "3.5", "tts", "sfx", "music"):
        assert needle in text, needle


# ------------------------------------------------------------- CLI flags in docs exist
def parser_flags():
    flags = set()
    for action in cli.build_parser()._actions:
        flags.update(f for f in action.option_strings if f.startswith("--"))
    return flags


def test_readme_cli_table_flags_exist():
    readme = read("README.md")
    table = readme.split("## CLI reference")[1].split("\n## ")[0]
    mentioned = set(re.findall(r"`(--[a-z][a-z0-9-]*)", table))
    assert mentioned, "no flags found"
    missing = sorted(f for f in mentioned if f not in parser_flags())
    assert not missing, missing


def test_skill_flags_exist():
    text = read("SKILL.md").split("## 3. Generate")[1].split("## 4.")[0]
    mentioned = set(re.findall(r"`(--[a-z][a-z0-9-]*)", text))
    missing = sorted(f for f in mentioned if f not in parser_flags())
    assert not missing, missing


def test_spec_flags_all_present():
    for flag in ("--mode", "--prompt", "--text", "--voice", "--speed", "--duration", "--steps", "--seed",
                 "--device", "--output"):
        assert flag in parser_flags(), flag


# ------------------------------------------------------------- packaging consistency
def test_requirements_and_pyproject_agree_and_pins_match():
    pyproject = tomllib.loads(read("pyproject.toml"))
    deps = {d.split("#")[0].strip() for d in pyproject["project"]["dependencies"]}
    reqs = {line.split("#")[0].strip() for line in read("requirements.txt").splitlines()
            if line.strip() and not line.strip().startswith("#")}
    assert deps == reqs
    assert any(C.SA3_COMMIT in d for d in deps)
    assert C.SA3_COMMIT in read("references/troubleshooting.md")
    assert pyproject["project"]["requires-python"] == ">=3.10,<3.13"


def test_gitignore_blocks_weights_and_outputs():
    ignore = read(".gitignore")
    for pattern in ("/models/", "*.safetensors", "*.pth", "*.pt", "/output/"):
        assert pattern in ignore


def test_license_files_present_and_notice_complete():
    notice = read("NOTICE")
    assert "This Stability AI Model is licensed under the Stability AI Community License" in notice
    assert "Powered by Stability AI" in notice and "Gemma Terms of Use" in notice
    assert "Apache" in notice and "MIT" in notice
    assert "STABILITY AI COMMUNITY LICENSE AGREEMENT" in read("licenses/STABILITY_AI_COMMUNITY_LICENSE.md")
    assert read("LICENSE").startswith("MIT License")
    assert "Powered by Stability AI" in read("README.md") and "Powered by Stability AI" in read("SKILL.md")
