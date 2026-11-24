# Prompting instrumental music (`--mode music`)

Read this when you write or review a music prompt, ask for a loop or a solo instrument, or need more than two minutes of music.

**Contents** - [What the model can and cannot do](#what-the-model-can-and-cannot-do) - [The four ingredients](#the-four-ingredients) - [AudioSparx tags](#audiosparx-tags) - [What the built-in enricher adds](#what-the-built-in-enricher-adds) - [Examples](#examples) - [Stems and solo instruments](#stems-and-solo-instruments) - [Duration, loops and music longer than 2 minutes](#duration-loops-and-music-longer-than-2-minutes) - [Common mistakes](#common-mistakes)

## What the model can and cannot do

* **Can:** instrumental music in many genres, ambient beds, loops, isolated instruments and stems, 44.1 kHz stereo, up to 120 s per clip (longer by continuation).
* **Cannot:** intelligible vocals or lyrics. Stability notes the models sometimes produce unintelligible vocal *textures*; singing a given lyric is not possible. Say so plainly if a user asks for a song with words, and offer an instrumental plus a spoken or sung-style narration via `--mode tts`.
* Sound effects belong to `--mode sfx`; the music model is not trained for one-shots.

## The four ingredients

| Ingredient | Why it matters | Example |
|---|---|---|
| **Genre / style** | anchors the overall sound | "warm lo-fi hip hop", "cinematic orchestral underscore" |
| **Instruments** | name them and say how they sound or feel | "mellow Rhodes electric piano, upright bass, brushed drums" |
| **Mood & energy** | steers harmony, dynamics and density | "relaxed and nostalgic", "tense and building" |
| **BPM** | fixes the tempo; beat-driven genres drift without it | "95 BPM" |

Add **mix / production** ("clean studio mix", "warm tape saturation", "wide cinematic reverb") - recordings in the training data carry such descriptors. Longer, descriptive sentences work well; Stability's own examples are 3-5 sentences long.

## AudioSparx tags

* `TrackType: Music, VocalType: Instrumental` - tends to give higher-quality, more coherent results (added by default).
* `Genre: Funk` (repeatable, e.g. `Genre: Funk, Genre: Jazz`) and `Instruments: Guitar, Saxophone, Bass, Piano` are also understood.
* `TrackType: Instrument` for an isolated instrument or stem (see below). `Format: Duo` when you want two instruments.

## What the built-in enricher adds

The CLI detects a genre profile from your words and adds only what is missing: instrumentation (if you name no instrument), mood (if you give none and a genre was recognised), mix style (if you give none), `no vocals`, and a BPM (skipped for free-tempo genres such as ambient). "slow" lowers and "fast" raises the default tempo. Anything you write wins; `--bpm`, `--genre`, `--instruments`, `--mood`, `--mix` override.

| profile | triggers | BPM | default instrumentation | mood | mix |
|---|---|---|---|---|---|
| `lofi` | lofi, lo-fi, chillhop | 80 | dusty Rhodes electric piano, mellow upright bass, lazy boom-bap drums with soft vinyl crackle | relaxed and nostalgic | warm tape-saturated mix |
| `synthwave` | synthwave, retrowave, outrun, vaporwave, 80s | 100 | analog arpeggiated synths, gated-reverb drums, warm synth bass, lush pads | neon-lit and nostalgic | wide 80s-style stereo mix |
| `drum_and_bass` | drum and bass, dnb, jungle | 174 | fast breakbeats, rolling sub bass, atmospheric pads | intense and energetic | punchy modern mix |
| `techno` | techno | 130 | hypnotic four-on-the-floor kick, driving sub bass, metallic percussion, evolving synth textures | dark and hypnotic | tight club mix |
| `house` | house, edm, dance, club | 124 | four-on-the-floor kick, offbeat open hi-hats, warm sub bass, plucky chord stabs | uplifting and groovy | clean club mix |
| `trap` | trap | 140 | 808 sub bass, rolling hi-hats, hard snare, dark synth pads | dark and energetic | loud punchy mix |
| `hiphop` | hip hop, boom bap, beat | 90 | punchy kick, crisp snare, deep sub bass, sampled keys | confident and laid-back | punchy modern mix |
| `ambient` | ambient, drone, soundscape, meditation, atmospheric | free | slowly evolving synth pads, soft textures, subtle shimmer | calm and spacious | wide reverberant mix |
| `cinematic` | cinematic, epic, trailer, orchestral, film score, soundtrack | 80 | full string orchestra, deep brass, timpani, low taiko drums | epic and emotional | wide cinematic mix |
| `jazz` | jazz, swing, bossa nova | 110 | upright bass, brushed drums, warm piano, muted trumpet | smooth and sophisticated | warm live-room mix |
| `rock` | rock, punk, metal, grunge | 125 | overdriven electric guitars, bass guitar, punchy live drum kit | driving and energetic | tight live-band mix |
| `piano` | piano | 70 | solo grand piano with soft felt hammers | intimate and emotional | close-mic'd studio mix |
| `acoustic` | acoustic, folk, fingerpicked, guitar | 95 | warm fingerpicked acoustic guitar, upright bass, brushed snare | relaxed and uplifting | clean studio mix |
| `tension` | tension, suspense, horror, thriller, ominous, dark | 70 | low drones, dissonant strings, sparse percussion, tension risers | tense and ominous | wide cinematic mix |
| `corporate` | corporate, upbeat, motivational, vlog, promo, pop | 110 | bright acoustic guitar, light claps, soft synth pads, round bass | optimistic and positive | clean modern mix |
| `chill` | chill, relax, study, background, calm | 90 | soft electric piano, warm pads, gentle drum machine, round bass | calm and relaxed | clean smooth mix |
| `general` | (no genre recognised) | 100 | soft piano, warm pads, gentle percussion, round bass | *(none added)* | clean studio mix |

## Examples

| You write | Final prompt |
|---|---|
| `calm acoustic background music for a travel vlog` | `TrackType: Music, VocalType: Instrumental, calm acoustic background music for a travel vlog, warm fingerpicked acoustic guitar, upright bass, brushed snare, clean studio mix, no vocals, 95 BPM` |
| `lofi study beat` | `TrackType: Music, VocalType: Instrumental, lofi study beat, dusty Rhodes electric piano, mellow upright bass, lazy boom-bap drums with soft vinyl crackle, relaxed and nostalgic mood, no vocals, 80 BPM` |
| `dark ambient drone` | no BPM (free tempo): `TrackType: Music, VocalType: Instrumental, dark ambient drone, slowly evolving synth pads, soft textures, subtle shimmer, wide reverberant mix, no vocals` |
| *agent-written, run with `--no-enrich`* | `TrackType: Music, VocalType: Instrumental, a warm, nostalgic lo-fi hip hop instrumental with a close-mic'd Rhodes playing mellow seventh chords, a dusty vinyl-sampled drum loop with a lazy swing, and a round upright bass; tape-saturated and slightly wobbly, 82 BPM, no vocals` |

## Stems and solo instruments

Start the prompt with `TrackType: Instrument` (the CLI does this for "solo cello", "isolated bass stem", or `--stem`), name the instrument and its technique, the style, the mood and the tempo; mention the recording environment ("cavernous reverb", "live-recorded in a vintage studio"). Use `Format: Duo` for two instruments. Example: `TrackType: Instrument, a sombre solo acoustic guitar track with delicate finger picking and cavernous reverb, 70 BPM`.

## Duration, loops and music longer than 2 minutes

* Choose a duration that fits the job: stingers and jingles 5-15 s, loops 8-30 s, beds and cues 30-120 s.
* **Loops:** `--loopable --duration 16` (use a length that is a whole number of bars at your BPM, e.g. 16 s = 8 bars at 120 BPM) blends the tail into the head so the file repeats without a click. The file ends up `--crossfade` seconds (default 2) shorter.
* **Past 120 s** the tool *continues* the music: the last `--context` seconds (default 30) are fed back through the model's inpainting path and only new audio is generated, then stitched with a short crossfade, so key, tempo and instrumentation carry on. Each extra step adds up to 90 s and costs about one full generation. A smaller `--context` gives the continuation more freedom (more variation); a larger one keeps it closer to what came before. Keep the prompt unchanged between steps - it should stay plausible for the music already there.
* If a seam is audible, increase `--crossfade` to 4-6, change `--seed`, or use `--extend-method loop` for texture-like music (pads, ambient).

## Common mistakes

* Asking for lyrics or "a song with a female singer" - not supported; produce the instrumental.
* No tempo for beat-driven music - add `--bpm` (or "95 BPM" in the prompt).
* Listing ten instruments - the model works better with 3-5 well-described ones.
* Mixing contradictory styles without saying how they combine.
* Using the music model for one-shot effects, or the SFX model for music.
