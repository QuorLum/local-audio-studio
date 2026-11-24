---
name: local-audio-studio
description: Generate speech (text-to-speech voiceovers and narration), sound effects (foley, ambience, UI and sci-fi sounds) and instrumental background music (scores, beds, loops) entirely on the local machine - free, offline, no API key - as a replacement for ElevenLabs. Kokoro-82M makes the voices; Stable Audio 3 Small makes effects and music; runs on CPU-only laptops and 4 GB GPUs. Use this skill whenever the user wants to create, narrate, voice, score or sound-design a video, game, podcast, app or prototype and wants it local, offline, free or private, or mentions ElevenLabs, Kokoro, Stable Audio, TTS, voiceover, narration, foley, ambience, SFX, sound effect, background music, soundtrack, jingle or loop, or asks to "generate audio" - even if they never name a tool.
compatibility: Python 3.10-3.12, PyTorch 2.7.1, git. About 6 GB of disk for all models (one-time download, then fully offline). CPU, or an NVIDIA GPU with at least 3.5 GB VRAM.
---

# Local Audio Studio

One CLI for three jobs, all running on the user's own machine: **speech** (`tts`), **sound effects and ambience** (`sfx`), **instrumental music** (`music`). No API key, no per-character cost, nothing leaves the computer after the one-time model download. Powered by Stability AI (Stable Audio 3) and Kokoro-82M.

All paths below are relative to this skill's folder. Use the project's Python: `.venv/bin/python` (Linux/macOS) or `.venv\Scripts\python.exe` (Windows) if a `.venv` exists, otherwise `python`.

## 1. Before the first generation: check the machine

```bash
python scripts/generate_audio.py --doctor
```

* `models:tts|sfx|music  warn missing` -> the weights are not downloaded yet. Tell the user what will be downloaded and how big it is (speech about 0.4 GB, effects or music about 3.6 GB each with a shared text encoder, everything about 6 GB), get a yes, then run only what is needed:
  `python scripts/setup_models.py --only tts --yes` (or `sfx`, `music`, `sfx,music`, `all`). Preview with `--dry-run`. No Hugging Face account or token is needed.
* Any `FAIL` (for example Python 3.13, missing package) -> run `./install.sh` (Linux/macOS/WSL2) or `.\install.ps1` (Windows) and re-check. `references/troubleshooting.md` covers the rest.
* This check is cheap; repeat it whenever a command fails with `error:` about models or imports.

## 2. Route the request

| The user wants | `--mode` | Engine | Output |
|---|---|---|---|
| Spoken words: narration, voiceover, dialogue lines, announcements, reading a script aloud | `tts` | Kokoro-82M | 24 kHz mono WAV |
| One discrete sound or an ambience: footsteps, door, whoosh, impact, UI blip, laser, animal, weather, room tone, cafe/rain/forest bed | `sfx` | Stable Audio 3 Small SFX | 44.1 kHz stereo WAV |
| Instrumental music: background bed, theme, score, stinger, loop, isolated instrument | `music` | Stable Audio 3 Small Music | 44.1 kHz stereo WAV |

Not supported - say so plainly and offer the closest alternative: **voice cloning**, **sung vocals or lyrics** (the music model makes instrumentals; for words use `tts`), **intelligible speech inside an effect** (generate speech separately and mix), streaming, SSML.

A scene usually needs several assets ("rainy cafe, narration, soft jazz" = an `sfx` bed + a `tts` read + a `music` track). Generate them as separate jobs, ideally in one `--batch` file (section 7), and mix them afterwards (section 8).

## 3. Generate

```bash
# speech
python scripts/generate_audio.py --mode tts --text "Welcome back to the channel." --voice af_heart --speed 1.0 -o out/intro.wav
python scripts/generate_audio.py --mode tts --text-file script.txt --voice bf_emma -o out/narration.wav

# sound effects and ambience
python scripts/generate_audio.py --mode sfx --prompt "footsteps on wet gravel" --duration 6 -o out/steps.wav
python scripts/generate_audio.py --mode sfx --prompt "steady rain on a tin roof" --duration 90 --loopable -o out/rain.wav

# music
python scripts/generate_audio.py --mode music --prompt "calm acoustic guitar for a travel vlog" --duration 45 -o out/bed.wav
```

The command prints only the output path on stdout (logs go to stderr). Add `--json` for a machine-readable result (path, duration, sample rate, seed, device, the prompt actually used).

| Flag | Applies to | Meaning |
|---|---|---|
| `--mode tts\|sfx\|music` | all | which engine (required) |
| `--text` / `--text-file F` | tts | what to say (`--prompt` is accepted as an alias) |
| `--prompt` | sfx, music | description of the sound or music |
| `--voice`, `--speed`, `--lang` | tts | voice (default `af_heart`, blends like `af_heart:0.7,af_bella:0.3`), speed 0.5-2.0, language code (auto from the voice) |
| `--duration SEC` | sfx, music | default 5 (sfx) / 30 (music); above 120 s it is extended automatically |
| `--steps N` | sfx, music | 1-12, default 8 (pingpong sampler) |
| `--cfg-scale`, `--negative-prompt` | sfx, music | default 1.0; above 1 enables the negative prompt (about 2x compute) |
| `--seed N` | all | omit for random; the seed used is reported so a take can be reproduced |
| `--loopable`, `--extend-method`, `--context`, `--crossfade` | sfx, music | seamless loops and >120 s handling (section 6) |
| `--device auto\|cuda\|cpu`, `--dtype` | sfx, music | default `auto`: CUDA fp16 if the GPU has >= 3.5 GB VRAM, else CPU fp32 |
| `--no-enrich`, `--mic/--room/--texture`, `--bpm/--genre/--instruments/--mood/--mix`, `--stem`, `--no-tags` | sfx, music | prompt enrichment control (section 4) |
| `--normalize DB`, `--no-normalize`, `--trim-silence`, `--fade-in/--fade-out`, `--sample-rate`, `--bit-depth` | all | post-processing (default: peak -1 dBFS, 5 ms click guard) |
| `--dry-run` | all | print routing, device decision, enriched prompt, chunking and the plan as JSON; **loads no model** |
| `--batch FILE`, `--list-voices`, `--doctor`, `--models-dir DIR`, `--quiet` | all | utilities |

## 4. Write the prompt (this decides the quality)

Stable Audio 3 learned from real recordings with descriptive metadata, so a prompt that reads like **a description of a real recording** sounds natural, while a bare noun ("rain") sounds synthetic. The CLI enriches prompts by default, adding only what is missing and never contradicting what you wrote. Check the result with `--dry-run`. If you write a complete prompt yourself, pass `--no-enrich`.

**Sound effects: source + action + production.** Name the physical source, how it is triggered and how it ends, then add microphone perspective, room acoustics and recording texture. Add `no music`.
* The enricher adds, for example: *foley* -> "close-mic foley recording, dry studio acoustics"; *outdoor ambience* -> "binaural outdoor field recording, open-air natural acoustics"; *UI/sci-fi* -> "clean designed sound effect ... dry, no room reverb"; and `TrackType: SFX`.
* Good: `TrackType: SFX, a heavy oak door slamming shut with a fast decay, close-mic'd in a dead room, a touch of analog saturation, no music`
* One sound per clip; layer ambience and events as separate files. Short duration for one-shots (0.5-3 s), 30-120 s for beds.

**Music: genre + instruments + mood + BPM + mix.** Name 3-5 concrete instruments and how they sound, the mood, a tempo, and the mix style; keep it instrumental.
* The enricher adds, for example: *acoustic* -> "95 BPM, warm fingerpicked acoustic guitar, upright bass, brushed snare, clean studio mix, no vocals" and `TrackType: Music, VocalType: Instrumental`.
* Good: `TrackType: Music, VocalType: Instrumental, warm nostalgic lo-fi hip hop with a close-mic'd Rhodes playing mellow seventh chords, dusty drum loop with lazy swing, round upright bass, tape-saturated, 82 BPM, no vocals`
* Solo instruments and stems: `--stem` (or "solo cello") uses `TrackType: Instrument`.

Full guides with the keyword tables, durations and pitfalls: `references/prompting-sfx.md`, `references/prompting-music.md`. Read the matching one before writing prompts for a larger project or when a result sounds wrong.

## 5. Speech tips

* Default voice `af_heart` (American English). `--list-voices` shows all 54 voices in 9 languages, including Hindi, Spanish, French, Italian, Japanese, Mandarin and Portuguese; the language follows the voice name.
* Write for the ear: spell out times, dates and phone numbers; a blank line makes a longer pause between paragraphs. Long scripts go in `--text-file`; the tool splits at sentence boundaries by itself.
* Respell or use `[word](/phonemes/)` markup for names Kokoro mispronounces. Details and the voice table: `references/tts-voices.md`.

## 6. Length, loops and long audio

* One Stable Audio 3 Small clip is at most **120 s**. Longer requests are extended automatically: `continue` (default) feeds the last `--context` seconds (30) back through the model's inpainting path and generates only the new part, stitched with a crossfade, so key and tempo carry on; `--extend-method loop` crossfade-loops one 120 s clip (best for rain, room tone, pads). Each extra step costs about one full generation, so warn the user before asking for 10+ minutes on a CPU.
* `--loopable` makes the end flow into the start (the file becomes `--crossfade` seconds shorter). For music loops use a whole number of bars (16 s = 8 bars at 120 BPM).

## 7. Device and VRAM rules (4 GB GPUs and CPUs)

* Leave `--device auto` alone: it uses CUDA + fp16 when the GPU has >= 3.5 GiB VRAM and >= 2.6 GiB is currently free, otherwise the CPU. Speech always runs fp32.
* **Run one generation at a time.** Never start two processes against the same small GPU. The tool keeps one model family in memory and runs `gc.collect()` and `torch.cuda.empty_cache()` whenever it switches between speech, effects and music.
* For many assets use **one process**: `python scripts/generate_audio.py --batch jobs.json` (a JSON list, `{"jobs": [...]}` or JSON Lines; each job has `mode`, `output` and the same fields as the flags, e.g. `{"mode": "sfx", "prompt": "door slam", "duration": 2, "output": "out/slam.wav"}`). Jobs are grouped by model, so each model loads once. See `examples/batch_jobs.json`.
* If a run fails with CUDA out-of-memory: close other GPU programs, shorten `--duration`, or re-run with `--device cpu`.
* Python API (same behaviour): `from localaudio import Studio` ... `with Studio() as s: s.sfx("door slam", duration=2, output="slam.wav")`. Always use the `with` block or call `s.close()` so GPU memory is released.
* The CPU path works but long music is slow; for the first draft ask for 20-30 s.

## 8. After generating

1. Confirm the file exists and the duration matches the request (`--json` gives `duration`, `sample_rate`, `seed`). Give the user the path.
2. Offer variations: re-run with another `--seed` (effects) or a tweaked prompt; mention the seed used for the take they liked.
3. Match the project: `--sample-rate 48000` for video editors, `--bit-depth 24`, `--normalize -3` for more headroom.
4. Mixing narration over music with the music ducked under the voice (tested with ffmpeg 6; the music drops about 18 dB while the voice speaks; lower `ratio` for gentler ducking):
   ```bash
   ffmpeg -i music.wav -i voice.wav -filter_complex "[1:a]aresample=44100,pan=stereo|c0=c0|c1=c0,asplit=2[v][sc];[0:a][sc]sidechaincompress=threshold=0.03:ratio=4:attack=20:release=400[bg];[bg][v]amix=inputs=2:duration=first:normalize=0[out]" -map "[out]" mix.wav
   ```

## 9. Licenses (mention when the user will publish or sell)

* Kokoro (speech): Apache-2.0, no restriction on the generated speech.
* Stable Audio 3 (effects, music): **Stability AI Community License** - free for personal and non-commercial use; commercial use is free below US $1M annual revenue but requires registering at stability.ai/community-license; you may not use outputs to train a foundation model; products or services built on it must show "Powered by Stability AI". The text encoder falls under Google's Gemma Terms. This is a summary, not legal advice: `references/licensing.md`, `licenses/`, `NOTICE`.

## Troubleshooting

`python scripts/generate_audio.py --doctor` and `python scripts/setup_models.py --check` first; then `references/troubleshooting.md` (install, downloads, GPU memory, speed, audio quality, Windows/macOS notes).
