import pytest

from localaudio import prompts as P


# ------------------------------------------------------------------ SFX
@pytest.mark.parametrize(
    "prompt, category",
    [
        ("footsteps on gravel", "foley"),
        ("rain on a tin roof", "ambience_nature"),
        ("busy cafe with chatter", "ambience_urban"),
        ("laser blast", "ui_synth"),
        ("big explosion", "impact_cinematic"),
        ("dog barking", "animal"),
        ("diesel truck engine idling", "vehicle_machine"),
        ("a person coughing", "human_vocal"),
        ("magical sparkle shimmer", "magic_fantasy"),
        ("something completely unusual", "general"),
    ],
)
def test_sfx_category_detection(prompt, category):
    assert P.enrich_sfx(prompt).category == category


def test_sfx_injects_mic_room_texture_and_no_music():
    e = P.enrich_sfx("footsteps on wet gravel")
    assert e.prompt.startswith("TrackType: SFX, footsteps on wet gravel")
    assert "close-mic foley recording" in e.prompt
    assert "dry studio acoustics" in e.prompt
    assert e.prompt.endswith("no music")
    assert "no music" in e.added


def test_sfx_is_conflict_aware():
    e = P.enrich_sfx("a heavy door creaking open in a large cathedral with long reverb")
    assert "dry studio acoustics" not in e.prompt  # user already described the room
    assert any("room" in n for n in e.notes)
    e2 = P.enrich_sfx("rain, lo-fi tape saturated")
    assert "wide stereo ambience" not in e2.prompt  # texture already given


def test_sfx_explicit_slots_win():
    e = P.enrich_sfx("door slam", mic="contact mic", room="small bathroom", texture="gritty")
    assert "contact mic" in e.prompt and "small bathroom" in e.prompt and "gritty" in e.prompt


def test_sfx_music_words_suppress_no_music():
    assert "no music" not in P.enrich_sfx("music box winding up").prompt
    assert P.enrich_sfx("rain, no music").prompt.count("no music") == 1


def test_sfx_tags_optional_and_not_duplicated():
    assert "TrackType" not in P.enrich_sfx("rain", tags=False).prompt
    once = P.enrich_sfx("TrackType: SFX, rain").prompt
    assert once.count("TrackType") == 1


def test_sfx_stays_within_budget():
    long_prompt = "a " + "very " * 100 + "long door slam"
    e = P.enrich_sfx(long_prompt)
    assert len(e.prompt) <= len(long_prompt) + 200 or len(e.prompt) <= P.MAX_PROMPT_CHARS + len(long_prompt)
    short_enough = P.enrich_sfx("x" * 600 + " door")
    assert any("dropped" in n for n in short_enough.notes)


def test_empty_prompt_rejected():
    with pytest.raises(ValueError):
        P.enrich_sfx("   ")
    with pytest.raises(ValueError):
        P.enrich_music("")


# ------------------------------------------------------------------ music
def test_music_spec_example_shape():
    e = P.enrich_music("calm acoustic background music for a travel vlog")
    assert e.prompt.startswith("TrackType: Music, VocalType: Instrumental, calm acoustic background music")
    assert "95 BPM" in e.prompt
    assert "fingerpicked acoustic guitar" in e.prompt
    assert "no vocals" in e.prompt
    assert e.category == "acoustic"


def test_music_keeps_user_bpm_instruments_mood_mix():
    e = P.enrich_music("epic cinematic trailer with taiko drums and a punchy mix at 140 BPM")
    assert e.prompt.count("BPM") == 1 and "140 BPM" in e.prompt
    assert "string orchestra" not in e.prompt  # user named instruments
    assert "epic and emotional" not in e.prompt  # user gave a mood
    assert "wide cinematic mix" not in e.prompt  # user gave a mix


def test_music_explicit_values_win():
    e = P.enrich_music("study music", bpm=72, instruments="felt piano and cello", mood="tender", mix="dry close-mic mix")
    assert "72 BPM" in e.prompt and "felt piano and cello" in e.prompt
    assert "tender mood" in e.prompt and "dry close-mic mix" in e.prompt


def test_music_ambient_has_free_tempo():
    e = P.enrich_music("dark ambient drone")
    assert "BPM" not in e.prompt
    assert any("free tempo" in n for n in e.notes)


def test_music_vocals_allowed():
    e = P.enrich_music("upbeat pop song with female vocals")
    assert "no vocals" not in e.prompt
    assert "VocalType: Instrumental" not in e.prompt
    forced = P.enrich_music("indie rock", vocals=True)
    assert "no vocals" not in forced.prompt


def test_music_instrumental_word_is_respected():
    e = P.enrich_music("instrumental lofi hip hop")
    assert e.prompt.count("no vocals") == 0  # already says instrumental
    assert "VocalType: Instrumental" in e.prompt


@pytest.mark.parametrize(
    "prompt, genre",
    [("lofi study beat", "lofi"), ("drum and bass banger", "drum_and_bass"), ("deep house groove", "house"),
     ("jazz trio", "jazz"), ("heavy metal riffs", "rock"), ("solo piano nocturne", "piano"),
     ("tense horror underscore", "tension"), ("synthwave night drive", "synthwave"), ("hello", "general")],
)
def test_music_genre_detection(prompt, genre):
    assert P.enrich_music(prompt).category == genre


def test_dispatch():
    assert P.enrich("sfx", "rain").mode == "sfx"
    assert P.enrich("music", "lofi", bpm=70).prompt.count("70 BPM") == 1
    assert P.enrich("sfx", "rain", bpm=70, mic="x").prompt  # irrelevant kwargs are ignored
    with pytest.raises(ValueError):
        P.enrich("tts", "hello")


# ------------------------------------------------------------------ stems / solo instruments
def test_stem_uses_instrument_tag_and_no_band_lineup():
    e = P.enrich_music("solo cello, slow and mournful")
    assert e.prompt.startswith("TrackType: Instrument, solo cello")
    assert "no vocals" not in e.prompt and "VocalType" not in e.prompt
    assert "string orchestra" not in e.prompt and any("instrumentation" in n for n in e.notes)


@pytest.mark.parametrize("prompt", ["isolated bass stem", "a solo electric guitar track", "solo piano nocturne"])
def test_stem_detection(prompt):
    assert P.enrich_music(prompt).prompt.startswith("TrackType: Instrument")


def test_full_band_with_a_solo_is_not_a_stem():
    e = P.enrich_music("funk groove with a guitar solo in the bridge")
    assert e.prompt.startswith("TrackType: Music, VocalType: Instrumental")


def test_stem_flag_forces_the_tag():
    assert P.enrich_music("tight kick and snare groove", stem=True).prompt.startswith("TrackType: Instrument")
    assert not P.enrich_music("solo cello", stem=False).prompt.startswith("TrackType: Instrument")


def test_no_arbitrary_mood_when_genre_unknown():
    e = P.enrich_music("solo cello, slow and mournful")
    assert "calm and positive" not in e.prompt  # user said mournful; never contradict it
    assert "72 BPM" in e.prompt  # 'slow' lowers the default tempo
    assert "mood" not in e.prompt or "mournful" in e.prompt
    assert P.enrich_music("hello").added  # still enriched, just without a made-up mood
    assert "mood" not in P.enrich_music("hello").prompt


def test_tempo_hints_adjust_default_bpm():
    assert "72 BPM" in P.enrich_music("slow lofi beat").prompt
    assert "128 BPM" in P.enrich_music("fast chase cue for a thriller").prompt or "BPM" in P.enrich_music("fast chase cue").prompt
    fast = P.enrich_music("fast-paced acoustic strumming")
    assert any(int(tok) >= 128 for tok in fast.prompt.split() if tok.isdigit())
