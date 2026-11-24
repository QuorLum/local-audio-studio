"""Acoustic prompt enrichment for Stable Audio 3.

Stable Audio 3 was trained on Freesound and AudioSparx recordings with
descriptive metadata. Short, abstract prompts ("rain") come out sounding like
generic synthetic audio; prompts that read like *a description of a real
recording* ("binaural outdoor field recording of steady rain on leaves ...")
come out far more natural. This module turns a short request into such a
prompt without calling any LLM:

* **SFX** - injects microphone perspective, room acoustics and recording
  texture (only the slots the user has not already specified), and ``no music``.
* **Music** - injects BPM, concrete instrumentation, mood and mix style, and
  ``no vocals`` for instrumentals.

Both optionally prepend the AudioSparx style tags the upstream prompting guide
recommends (``TrackType: SFX`` / ``TrackType: Music, VocalType: Instrumental``).

The enrichment is *conflict-aware*: if the prompt already talks about
reverb, a cathedral, tape saturation, 120 BPM, a cello ... the corresponding
slot is left alone. Calling agents can (and should) write rich prompts
themselves and pass ``--no-enrich``; see ``references/prompting-*.md``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Pattern, Sequence, Tuple

MAX_PROMPT_CHARS = 700  # comfortably inside the 256-token T5Gemma window


@dataclass(frozen=True)
class EnrichedPrompt:
    prompt: str
    original: str
    mode: str
    category: str
    added: Tuple[str, ...]
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "prompt": self.prompt,
            "original": self.original,
            "mode": self.mode,
            "category": self.category,
            "added": list(self.added),
            "notes": list(self.notes),
        }


def _words(words: Sequence[str]) -> Pattern[str]:
    """Whole-word matcher that tolerates common suffixes (s, es, ing, ed)."""
    alt = "|".join(re.escape(w).replace(r"\ ", r"[\s-]") for w in words)
    return re.compile(rf"\b(?:{alt})(?:s|es|ing|ed)?\b", re.IGNORECASE)


def _clean(prompt: str) -> str:
    prompt = re.sub(r"\s+", " ", (prompt or "").strip())
    return prompt.rstrip(" .,;:!")


def _has(pattern: Pattern[str], text: str) -> bool:
    return bool(pattern.search(text))


# --------------------------------------------------------------------------- #
# Sound effects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _SfxCategory:
    name: str
    pattern: Pattern[str]
    mic: str
    room: str
    texture: str
    ambience: bool = False


_SFX_CATEGORIES: Tuple[_SfxCategory, ...] = (
    _SfxCategory(
        "ui_synth",
        _words(["laser", "zap", "beep", "blip", "bleep", "boop", "glitch", "interface", "ui", "notification",
                "power up", "power down", "sci-fi", "scifi", "robot", "robotic", "hologram", "synth", "arcade",
                "8-bit", "riser", "sweep", "transition", "alert", "chime", "ping", "level up", "game over",
                "menu", "cursor", "futuristic", "cyber", "spaceship", "teleport", "warp"]),
        "clean designed sound effect with precise stereo placement",
        "dry, no room reverb",
        "crisp high-fidelity studio sound design",
    ),
    _SfxCategory(
        "magic_fantasy",
        _words(["magic", "magical", "spell", "shimmer", "sparkle", "sparkling", "fantasy", "mystical",
                "enchant", "fairy", "wand", "potion", "portal", "energy swirl"]),
        "layered sound design with wide stereo movement",
        "airy natural reverb tail",
        "clean high-fidelity",
    ),
    _SfxCategory(
        "impact_cinematic",
        _words(["explosion", "explode", "boom", "impact", "crash", "slam", "punch", "gunshot", "gun", "shot",
                "cannon", "rumble", "debris", "bang", "smash", "hit", "whoosh", "swoosh", "sword", "slash",
                "stab", "collapse", "blast"]),
        "close-mic'd cinematic sound effect",
        "natural room tail",
        "tight punchy transient with full low end",
    ),
    _SfxCategory(
        "foley",
        _words(["footstep", "step", "walking", "running", "door", "knock", "creak", "cloth", "clothing",
                "rustle", "zipper", "key", "keys", "typing", "keyboard", "mouse click", "click", "pour",
                "pouring", "drip", "dripping", "glass", "cup", "mug", "dish", "dishes", "cutlery", "cooking",
                "chop", "chopping", "sizzle", "sizzling", "frying", "boil", "boiling", "paper", "page", "book",
                "pen", "writing", "scissors", "tape", "coin", "coins", "switch", "button", "drawer", "bag",
                "shoe", "boot", "jump", "knife", "fabric", "leather", "gravel", "crunch", "chew", "bottle",
                "cork", "match", "lighter", "clap", "snap", "squeak", "scrape", "splash", "bubble"]),
        "close-mic foley recording",
        "dry studio acoustics",
        "clean and detailed, no background noise",
    ),
    _SfxCategory(
        "ambience_nature",
        _words(["rain", "rainfall", "drizzle", "wind", "breeze", "forest", "woods", "birds", "birdsong",
                "river", "stream", "creek", "waterfall", "ocean", "sea", "waves", "beach", "shore", "storm",
                "thunderstorm", "thunder", "jungle", "cricket", "crickets", "insects", "night", "campfire",
                "fire", "crackling", "leaves", "snow", "desert", "mountain", "meadow", "lake", "swamp",
                "frogs", "dawn", "countryside"]),
        "binaural outdoor field recording",
        "open-air natural acoustics",
        "wide stereo ambience with natural dynamics",
        ambience=True,
    ),
    _SfxCategory(
        "ambience_urban",
        _words(["city", "traffic", "street", "crowd", "cafe", "café", "coffee shop", "restaurant", "market",
                "station", "subway", "metro", "airport", "mall", "office", "school", "playground",
                "construction", "crosswalk", "bar", "pub", "party", "stadium", "gym", "hospital", "lobby",
                "elevator", "library", "classroom", "walla"]),
        "stereo field recording on location",
        "natural reverberant room tone",
        "continuous ambient bed",
        ambience=True,
    ),
    _SfxCategory(
        "vehicle_machine",
        _words(["car", "engine", "motor", "motorcycle", "truck", "bus", "train", "airplane", "aeroplane",
                "plane", "jet", "helicopter", "boat", "ship", "machine", "machinery", "factory", "fan", "hvac",
                "air conditioner", "fridge", "washing machine", "generator", "vacuum", "drill", "saw",
                "hammer", "chainsaw", "siren", "horn", "tires", "tyres", "brake", "turbine", "propeller"]),
        "close-mic'd mechanical recording",
        "in-situ acoustics",
        "detailed and realistic",
    ),
    _SfxCategory(
        "animal",
        _words(["dog", "cat", "lion", "tiger", "horse", "cow", "rooster", "chicken", "wolf", "owl", "frog",
                "bear", "elephant", "monkey", "pig", "sheep", "goat", "duck", "bark", "meow", "roar", "growl",
                "howl", "purr", "chirp", "bird"]),
        "close-mic wildlife recording",
        "natural outdoor acoustics",
        "clean and realistic",
    ),
    _SfxCategory(
        "human_vocal",
        _words(["laugh", "laughter", "cough", "sneeze", "scream", "gasp", "breath", "breathing", "sigh",
                "cheer", "applause", "whistle", "heartbeat", "crying", "giggle", "yawn"]),
        "close-mic human vocalization",
        "dry booth acoustics",
        "clean and natural",
    ),
)
_SFX_DEFAULT = _SfxCategory(
    "general",
    re.compile(r"$^"),
    "clean high-fidelity recording",
    "natural acoustics",
    "detailed stereo image",
)

_MIC_WORDS = re.compile(
    r"\b(?:mic|microphone|binaural|stereo|mono|lavalier|contact mic|ribbon|condenser|shotgun|foley|"
    r"field recording|close[- ]?up|close[- ]?mic(?:'?d)?|overhead|ortf|hydrophone|sound design)\b",
    re.IGNORECASE,
)
_ROOM_WORDS = re.compile(
    r"\b(?:reverb\w*|room|studio|hall|cathedral|cave|church|outdoors?|indoors?|open[- ]air|acoustics|echo\w*|"
    r"dry[- ](?:room|studio|acoustics|sound|recording|booth|mix)|wet[- ](?:reverb|room|mix)|"
    r"tunnel|canyon|bathroom|stairwell|anechoic|warehouse|garage|booth)\b",
    re.IGNORECASE,
)
_TEXTURE_WORDS = re.compile(
    r"\b(?:lo-?fi|tape|vinyl|hi-?fi|high[- ]fidelity|clean|noisy|gritty|crisp|warm|saturat\w*|distort\w*|"
    r"bitcrush\w*|analog(?:ue)?|polished|vintage|muffled|tinny)\b",
    re.IGNORECASE,
)
_MUSIC_WORD = re.compile(r"\b(?:music|musical|song|melody|jingle|soundtrack|score)\b", re.IGNORECASE)
_NO_MUSIC = re.compile(r"\bno music\b", re.IGNORECASE)
_SFX_TAG = re.compile(r"TrackType\s*:", re.IGNORECASE)


def detect_sfx_category(prompt: str) -> _SfxCategory:
    """Pick the category with the most keyword hits (ties go to the earlier one)."""
    best, best_hits = _SFX_DEFAULT, 0
    for category in _SFX_CATEGORIES:
        hits = len(category.pattern.findall(prompt))
        if hits > best_hits:
            best, best_hits = category, hits
    return best


def enrich_sfx(
    prompt: str,
    *,
    mic: Optional[str] = None,
    room: Optional[str] = None,
    texture: Optional[str] = None,
    tags: bool = True,
    no_music: bool = True,
) -> EnrichedPrompt:
    """Make a sound-effect prompt read like a description of a real recording."""
    original = prompt
    base = _clean(prompt)
    if not base:
        raise ValueError("empty prompt")
    category = detect_sfx_category(base)
    parts: List[str] = []
    added: List[str] = []
    notes: List[str] = []

    if tags and not _SFX_TAG.search(base):
        parts.append("TrackType: SFX")
        added.append("tag:TrackType SFX")
    parts.append(base)

    slots = (
        ("mic", mic, category.mic, _MIC_WORDS),
        ("room", room, category.room, _ROOM_WORDS),
        ("texture", texture, category.texture, _TEXTURE_WORDS),
    )
    slot_parts: List[Tuple[str, str]] = []
    for name, explicit, default, detector in slots:
        if explicit:
            slot_parts.append((name, explicit.strip()))
            added.append(f"{name}:explicit")
        elif _has(detector, base):
            notes.append(f"kept your {name} wording")
        else:
            slot_parts.append((name, default))
            added.append(f"{name}:{default}")

    if no_music and not _MUSIC_WORD.search(base) and not _NO_MUSIC.search(base):
        tail = ["no music"]
        added.append("no music")
    else:
        tail = []

    def assemble(slots_used: Sequence[Tuple[str, str]]) -> str:
        return ", ".join(parts + [s for _, s in slots_used] + tail)

    final = assemble(slot_parts)
    # over budget -> drop the least important slots first (texture, then room)
    for drop in ("texture", "room"):
        if len(final) <= MAX_PROMPT_CHARS:
            break
        slot_parts = [(n, s) for n, s in slot_parts if n != drop]
        notes.append(f"dropped {drop} slot to stay under {MAX_PROMPT_CHARS} characters")
        final = assemble(slot_parts)
    return EnrichedPrompt(final, original, "sfx", category.name, tuple(added), tuple(notes))


# --------------------------------------------------------------------------- #
# Music
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class _Genre:
    name: str
    pattern: Pattern[str]
    bpm: Optional[int]
    instruments: str
    mood: str
    mix: str


_GENRES: Tuple[_Genre, ...] = (
    _Genre("lofi", _words(["lofi", "lo-fi", "lo fi", "chillhop"]), 80,
           "dusty Rhodes electric piano, mellow upright bass, lazy boom-bap drums with soft vinyl crackle",
           "relaxed and nostalgic", "warm tape-saturated mix"),
    _Genre("synthwave", _words(["synthwave", "retrowave", "outrun", "vaporwave", "80s"]), 100,
           "analog arpeggiated synths, gated-reverb drums, warm synth bass, lush pads",
           "neon-lit and nostalgic", "wide 80s-style stereo mix"),
    _Genre("drum_and_bass", _words(["drum and bass", "drum & bass", "dnb", "jungle"]), 174,
           "fast breakbeats, rolling sub bass, atmospheric pads",
           "intense and energetic", "punchy modern mix"),
    _Genre("techno", _words(["techno"]), 130,
           "hypnotic four-on-the-floor kick, driving sub bass, metallic percussion, evolving synth textures",
           "dark and hypnotic", "tight club mix"),
    _Genre("house", _words(["house", "edm", "dance", "club"]), 124,
           "four-on-the-floor kick, offbeat open hi-hats, warm sub bass, plucky chord stabs",
           "uplifting and groovy", "clean club mix"),
    _Genre("trap", _words(["trap"]), 140,
           "808 sub bass, rolling hi-hats, hard snare, dark synth pads",
           "dark and energetic", "loud punchy mix"),
    _Genre("hiphop", _words(["hip hop", "hip-hop", "boom bap", "beat"]), 90,
           "punchy kick, crisp snare, deep sub bass, sampled keys",
           "confident and laid-back", "punchy modern mix"),
    _Genre("ambient", _words(["ambient", "drone", "soundscape", "meditation", "atmospheric"]), None,
           "slowly evolving synth pads, soft textures, subtle shimmer",
           "calm and spacious", "wide reverberant mix"),
    _Genre("cinematic", _words(["cinematic", "epic", "trailer", "orchestral", "film score", "soundtrack"]), 80,
           "full string orchestra, deep brass, timpani, low taiko drums",
           "epic and emotional", "wide cinematic mix"),
    _Genre("jazz", _words(["jazz", "swing", "bossa nova", "bossa"]), 110,
           "upright bass, brushed drums, warm piano, muted trumpet",
           "smooth and sophisticated", "warm live-room mix"),
    _Genre("rock", _words(["rock", "punk", "metal", "grunge"]), 125,
           "overdriven electric guitars, bass guitar, punchy live drum kit",
           "driving and energetic", "tight live-band mix"),
    _Genre("piano", _words(["piano"]), 70,
           "solo grand piano with soft felt hammers",
           "intimate and emotional", "close-mic'd studio mix"),
    _Genre("acoustic", _words(["acoustic", "folk", "fingerpicked", "fingerstyle", "guitar"]), 95,
           "warm fingerpicked acoustic guitar, upright bass, brushed snare",
           "relaxed and uplifting", "clean studio mix"),
    _Genre("tension", _words(["tension", "suspense", "suspenseful", "horror", "thriller", "ominous", "dark"]), 70,
           "low drones, dissonant strings, sparse percussion, tension risers",
           "tense and ominous", "wide cinematic mix"),
    _Genre("corporate", _words(["corporate", "upbeat", "motivational", "commercial", "vlog", "promo", "pop"]), 110,
           "bright acoustic guitar, light claps, soft synth pads, round bass",
           "optimistic and positive", "clean modern mix"),
    _Genre("chill", _words(["chill", "relax", "relaxing", "study", "background", "calm"]), 90,
           "soft electric piano, warm pads, gentle drum machine, round bass",
           "calm and relaxed", "clean smooth mix"),
)
_GENRE_DEFAULT = _Genre(
    "general", re.compile(r"$^"), 100,
    "soft piano, warm pads, gentle percussion, round bass",
    "calm and positive", "clean studio mix",
)

_BPM_RE = re.compile(r"\b\d{2,3}\s*-?\s*bpm\b|\bbpm\s*[:=]?\s*\d{2,3}\b", re.IGNORECASE)
_INSTRUMENT_WORDS = _words([
    "guitar", "piano", "keys", "keyboard", "rhodes", "organ", "synth", "synthesizer", "pad", "bass", "drum",
    "kick", "snare", "hi-hat", "hihat", "percussion", "violin", "viola", "cello", "string", "orchestra",
    "brass", "trumpet", "trombone", "horn", "saxophone", "sax", "flute", "clarinet", "oboe", "harp",
    "banjo", "ukulele", "mandolin", "sitar", "tabla", "marimba", "vibraphone", "xylophone", "glockenspiel",
    "bell", "accordion", "harmonica", "808", "arpeggio", "arp", "pluck", "lead", "choir", "taiko", "timpani",
])
_MOOD_WORDS = _words([
    "calm", "relaxed", "relaxing", "chill", "peaceful", "serene", "uplifting", "upbeat", "happy", "joyful",
    "cheerful", "sad", "melancholic", "melancholy", "emotional", "dark", "ominous", "tense", "tension",
    "suspenseful", "epic", "heroic", "triumphant", "energetic", "intense", "aggressive", "dreamy",
    "nostalgic", "romantic", "mysterious", "playful", "quirky", "hopeful", "inspiring", "inspirational",
    "motivational", "cozy", "warm", "ethereal", "spacious", "groovy", "funky", "bright", "gentle", "soothing",
    "mournful", "somber", "sombre", "brooding", "haunting", "tender", "wistful", "bittersweet", "lonely", "eerie",
    "creepy", "spooky", "majestic", "dramatic", "whimsical", "mellow", "laid-back", "lush", "delicate", "hypnotic",
    "anthemic", "solemn", "reflective", "introspective", "sentimental", "festive", "sunny", "cold", "icy", "furious",
    "menacing", "sinister", "wondrous", "magical", "tranquil", "glorious", "nostalgia",
])
_SLOW_WORDS = _words(["slow", "ballad", "lullaby", "mournful", "funeral", "adagio", "languid", "sluggish"])
_FAST_WORDS = _words(["fast", "fast-paced", "uptempo", "up-tempo", "driving", "frantic", "racing", "chase",
                      "high-energy", "breakneck"])
_MIX_WORDS = re.compile(
    r"\b(?:mix|mixed|mastered|mastering|production|produced|lo-?fi|tape|vinyl|reverb\w*|dry|wide|live|"
    r"studio|clean|punchy|saturat\w*|compress\w*)\b",
    re.IGNORECASE,
)
_VOCAL_WORDS = _words(["vocal", "sing", "singer", "lyrics", "choir", "voice", "chant", "humming", "vocalise",
                       "a cappella", "rap", "rapper"])
_NO_VOCALS = re.compile(r"\b(?:no|without)\s+(?:vocals?|singing|lyrics)\b|\binstrumental\b", re.IGNORECASE)
_MUSIC_TAG = re.compile(r"TrackType\s*:", re.IGNORECASE)
_STEM_WORDS = re.compile(
    r"\b(?:stems?|isolated|single[- ]instrument)\b|"
    r"\bsolo\s+(?:[\w'-]+\s+){0,2}(?:guitar|piano|violin|viola|cello|bass|drums?|flute|sax\w*|trumpet|"
    r"trombone|clarinet|oboe|harp|banjo|ukulele|mandolin|organ|accordion|harmonica|synth\w*|marimba|"
    r"vibraphone|percussion|keys|keyboard)\b",
    re.IGNORECASE,
)


def detect_genre(prompt: str) -> _Genre:
    for genre in _GENRES:
        if genre.pattern.search(prompt):
            return genre
    return _GENRE_DEFAULT


def enrich_music(
    prompt: str,
    *,
    bpm: Optional[int] = None,
    genre: Optional[str] = None,
    instruments: Optional[str] = None,
    mood: Optional[str] = None,
    mix: Optional[str] = None,
    vocals: Optional[bool] = None,
    stem: Optional[bool] = None,
    tags: bool = True,
) -> EnrichedPrompt:
    """Make a music prompt concrete: BPM, instrumentation, mood, mix style.

    ``stem=True`` (or a prompt such as "solo cello" / "isolated bass stem") uses the
    ``TrackType: Instrument`` tag the upstream guide recommends for isolated
    instruments, and does not add a generic band line-up.
    """
    original = prompt
    base = _clean(prompt)
    if not base:
        raise ValueError("empty prompt")
    profile = detect_genre(f"{genre} {base}" if genre else base)
    wants_vocals = vocals if vocals is not None else _has(_VOCAL_WORDS, base)
    instrumental = (not wants_vocals) or bool(_NO_VOCALS.search(base))
    is_stem = stem if stem is not None else bool(_STEM_WORDS.search(base))
    parts: List[str] = []
    added: List[str] = []
    notes: List[str] = []

    if tags and not _MUSIC_TAG.search(base):
        if is_stem:
            parts.append("TrackType: Instrument")
            added.append("tag:TrackType Instrument")
        elif instrumental:
            parts.append("TrackType: Music, VocalType: Instrumental")
            added.append("tag:TrackType Music, VocalType Instrumental")
        else:
            parts.append("TrackType: Music")
            added.append("tag:TrackType Music")
    parts.append(base)

    body: List[Tuple[str, str]] = []
    if instruments:
        body.append(("instruments", instruments.strip()))
        added.append("instruments:explicit")
    elif _has(_INSTRUMENT_WORDS, base) or is_stem:
        notes.append("kept your instrumentation")
    else:
        body.append(("instruments", profile.instruments))
        added.append(f"instruments:{profile.instruments}")

    if mood:
        body.append(("mood", f"{mood.strip()} mood"))
        added.append("mood:explicit")
    elif _has(_MOOD_WORDS, base):
        notes.append("kept your mood wording")
    elif profile.name == "general":
        notes.append("no mood added (no genre recognised; describe the mood yourself)")
    else:
        body.append(("mood", f"{profile.mood} mood"))
        added.append(f"mood:{profile.mood}")

    if mix:
        body.append(("mix", mix.strip()))
        added.append("mix:explicit")
    elif _MIX_WORDS.search(base):
        notes.append("kept your mix wording")
    else:
        body.append(("mix", profile.mix))
        added.append(f"mix:{profile.mix}")

    tail: List[str] = []
    if instrumental and not is_stem and not _NO_VOCALS.search(base):
        tail.append("no vocals")
        added.append("no vocals")

    bpm_text = ""
    if bpm:
        bpm_text = f"{int(bpm)} BPM"
        added.append(f"bpm:{int(bpm)}")
    elif _BPM_RE.search(base):
        notes.append("kept your BPM")
    elif profile.bpm is not None:
        tempo = profile.bpm
        if _has(_SLOW_WORDS, base):
            tempo = min(tempo, 72)
        elif _has(_FAST_WORDS, base):
            tempo = max(tempo, 128)
        bpm_text = f"{tempo} BPM"
        added.append(f"bpm:{tempo} (typical for {profile.name})")
    else:
        notes.append(f"no BPM added for {profile.name} (free tempo)")

    def assemble(body_used: Sequence[Tuple[str, str]]) -> str:
        return ", ".join(parts + [s for _, s in body_used] + tail + ([bpm_text] if bpm_text else []))

    final = assemble(body)
    for drop in ("mix", "mood"):
        if len(final) <= MAX_PROMPT_CHARS:
            break
        body = [(n, s) for n, s in body if n != drop]
        notes.append(f"dropped {drop} slot to stay under {MAX_PROMPT_CHARS} characters")
        final = assemble(body)
    return EnrichedPrompt(final, original, "music", profile.name, tuple(added), tuple(notes))


def enrich(mode: str, prompt: str, **kwargs) -> EnrichedPrompt:
    """Dispatch to :func:`enrich_sfx` / :func:`enrich_music`."""
    if mode == "sfx":
        allowed = {"mic", "room", "texture", "tags", "no_music"}
        return enrich_sfx(prompt, **{k: v for k, v in kwargs.items() if k in allowed})
    if mode == "music":
        allowed = {"bpm", "genre", "instruments", "mood", "mix", "vocals", "stem", "tags"}
        return enrich_music(prompt, **{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"enrichment applies to sfx and music, not {mode!r}")
