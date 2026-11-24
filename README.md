# local-audio-studio

**Speech, sound effects and music from one command - 100% on your own machine, zero cost, no API key, works offline.**
An open-source, offline prototyping alternative to hosted audio services such as ElevenLabs, for video creators and developers. Runs on **CPU-only laptops** and on **GPUs with 4 GB of VRAM**.

![license](https://img.shields.io/badge/wrapper-MIT-green) ![python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue) ![offline](https://img.shields.io/badge/runs-offline-success)

> **Powered by Stability AI** (Stable Audio 3 Small for sound effects and music) and **Kokoro-82M** (speech).
> Not affiliated with ElevenLabs; the name appears only to describe what this is an alternative to.

| | Engine | Model | Output |
|---|---|---|---|
| **Speech** (`--mode tts`) | Kokoro-82M, Apache-2.0 | 54 voices, 9 languages | 24 kHz mono WAV |
| **Sound effects & ambience** (`--mode sfx`) | Stable Audio 3 Small SFX | text-to-audio, up to 120 s per clip | 44.1 kHz stereo WAV |
| **Instrumental music** (`--mode music`) | Stable Audio 3 Small Music | text-to-audio, up to 120 s per clip | 44.1 kHz stereo WAV |

It is both a **CLI / Python module** and an **agent skill** (`SKILL.md`): an AI coding agent can route "add rain and soft jazz under this narration" to the right model, write good prompts for it, and manage GPU memory.

## Contents

[Why](#why) - [Hardware](#hardware-requirements) - [Quickstart](#quickstart) - [Examples](#examples) - [CLI reference](#cli-reference) - [Long audio and loops](#long-audio-loops-and-the-120-second-limit) - [Prompt enrichment](#automatic-acoustic-prompt-enrichment) - [Python API](#python-api) - [Use as an agent skill](#use-as-an-agent-skill) - [How it works](#how-it-works) - [Verification status](#verification-status) - [Licenses](#licenses-and-attribution) - [Limitations](#limitations) - [Development](#development)

## Why

* **Free and private.** Scripts, voice-overs and game assets never leave your machine; there is no meter running.
* **One tool for three jobs.** Narration, foley/ambience and a music bed in a single workflow and a single model-memory budget.
* **Built for small hardware.** One model family in memory at a time with strict cleanup on every switch, fp16 on 4 GB GPUs, automatic CPU fallback, text encoder offloaded before sampling.
* **Good prompts by default.** Stable Audio 3 sounds natural when prompted like a description of a real recording; the tool adds microphone perspective, room acoustics, BPM and instrumentation for you.

What it is **not**: a clone of a commercial studio. There is no voice cloning, dubbing, streaming API or sung vocals. Quality is excellent for prototyping, drafts and small productions; judge it by ear for yours.

## Hardware requirements

| | CPU-only laptop | GPU with 4 GB VRAM (e.g. RTX 3050 laptop) | Bigger GPU |
|---|---|---|---|
| Speech (Kokoro, 82M) | works, faster than real time in our tests (one CPU core) | works | works |
| Effects and music (SA3 Small) | works; short effects are quick, **long music is slow** | **recommended**: fp16, ~2.4 GB peak at 120 s per upstream measurement | works |
| Precision | fp32 (bf16 on request) | fp16 (text encoder bf16, offloaded before sampling) | fp16 |
| RAM | **8 GB** recommended (about 4 GB in use while generating, our estimate) | 8 GB recommended | 8 GB |
| Disk | speech 0.4 GB, effects or music 3.6 GB each (shared encoder), all about 6 GB | same | same |
| OS | Linux, macOS (Apple Silicon runs on CPU), Windows, WSL2 | Linux, Windows, WSL2 | same |

`--device auto` (default) uses CUDA + fp16 when the GPU has at least 3.5 GiB of VRAM (and about 2.6 GiB is free), otherwise the CPU. Python 3.10-3.12 (Kokoro's text front-end does not support 3.13 yet), `git`, and about 6 GB of disk for all models.

## Quickstart

```bash
git clone https://github.com/QuorLum/local-audio-studio.git && cd local-audio-studio

./install.sh                  # Linux / macOS / WSL2: venv + right PyTorch build + requirements
#   .\install.ps1             # Windows PowerShell (see the note below)
```

The installer picks the CUDA PyTorch wheel if it finds an NVIDIA GPU (`--cpu` forces the CPU build, `--cuda cu118` another wheel set) and finishes by offering the model download. Or do it yourself later:

```bash
python scripts/setup_models.py                    # interactive menu (or everything when not in a terminal)
python scripts/setup_models.py --only tts         # speech only (~0.4 GB)
python scripts/setup_models.py --only sfx,music   # effects + music (~5 GB)
python scripts/setup_models.py --dry-run          # what would be downloaded, and how much
python scripts/setup_models.py --check            # verify an existing installation
python scripts/generate_audio.py --doctor         # is this machine ready?
```

**No Hugging Face account or token is needed.** Models land in `./models` (change with `--models-dir` or `LOCALAUDIO_MODELS_DIR`) and are git-ignored; downloads resume if interrupted. After setup, generation makes no network requests.

> **Windows:** run `.\install.ps1` in PowerShell (if scripts are blocked: `powershell -ExecutionPolicy Bypass -File .\install.ps1`). WSL2 + NVIDIA driver also works and then you use `install.sh` inside WSL2.

## Examples

```bash
# speech
python scripts/generate_audio.py --mode tts --text "Welcome back to the channel." --voice af_heart -o intro.wav
python scripts/generate_audio.py --mode tts --text-file examples/narration.txt --voice bf_emma --speed 0.95 -o narration.wav

# sound effects and ambience
python scripts/generate_audio.py --mode sfx --prompt "footsteps on wet gravel" --duration 6 -o steps.wav
python scripts/generate_audio.py --mode sfx --prompt "steady rain on a tin roof" --duration 90 --loopable -o rain.wav

# music
python scripts/generate_audio.py --mode music --prompt "calm acoustic guitar for a travel vlog" --duration 45 -o bed.wav
python scripts/generate_audio.py --mode music --prompt "solo cello, slow and mournful" --stem --duration 30 -o cello.wav

# preview the routing, device decision and enriched prompt without loading any model
python scripts/generate_audio.py --mode music --prompt "lofi study beat" --dry-run

# many assets in one process (one model load per engine)
python scripts/generate_audio.py --batch examples/batch_jobs.json
```

Only the output path is printed on stdout; progress and logs go to stderr (`--json` prints a full result record, `--quiet` silences stderr).

Example prompts and what the enricher turns them into:

| You write (`--mode sfx`) | Prompt sent to the model |
|---|---|
| `footsteps on wet gravel` | `TrackType: SFX, footsteps on wet gravel, close-mic foley recording, dry studio acoustics, clean and detailed, no background noise, no music` |
| `rain on a tin roof` | `TrackType: SFX, rain on a tin roof, binaural outdoor field recording, open-air natural acoustics, wide stereo ambience with natural dynamics, no music` |

| You write (`--mode music`) | Prompt sent to the model |
|---|---|
| `calm acoustic background music for a travel vlog` | `TrackType: Music, VocalType: Instrumental, calm acoustic background music for a travel vlog, warm fingerpicked acoustic guitar, upright bass, brushed snare, clean studio mix, no vocals, 95 BPM` |

## CLI reference

`python scripts/generate_audio.py --help` is authoritative. The essentials:

| Flag | Meaning |
|---|---|
| `--mode tts\|sfx\|music` | engine to use |
| `--text`, `--text-file`, `--prompt` | what to speak / what to generate |
| `--voice`, `--speed`, `--lang` | speech: voice or weighted blend (`af_heart:0.7,af_bella:0.3`), 0.5-2.0, language code |
| `--duration` | seconds (sfx 5, music 30 by default; above 120 s is extended automatically) |
| `--steps` | 1-12, default **8** (pingpong sampler) |
| `--cfg-scale`, `--negative-prompt` | default **1.0**; above 1 turns on the negative prompt (about 2x compute) |
| `--seed` | omit for random; the seed used is reported |
| `--device auto\|cuda\|cpu`, `--dtype` | placement and precision (`auto`: CUDA fp16 with >= 3.5 GiB VRAM, else CPU fp32) |
| `--loopable`, `--extend-method`, `--context`, `--crossfade` | loops and >120 s handling |
| `--no-enrich`, `--mic`, `--room`, `--texture`, `--bpm`, `--genre`, `--instruments`, `--mood`, `--mix`, `--stem`, `--no-tags` | prompt enrichment control |
| `-o/--output`, `--normalize`, `--trim-silence`, `--fade-in/out`, `--sample-rate`, `--bit-depth` | output and post-processing |
| `--dry-run`, `--batch`, `--list-voices`, `--doctor`, `--json`, `--quiet`, `--models-dir` | utilities |

Exit codes: `0` ok, `2` bad request, `3` models missing, `4` dependency problem, `5` device unavailable, `6` model files do not match the architecture.

## Long audio, loops and the 120-second limit

A single Stable Audio 3 Small clip is limited to 120 s. For longer requests the tool extends the audio automatically:

* **`--extend-method continue`** (default): the last `--context` seconds (30) are fed back through the model's official inpainting path and only the new part is generated, then stitched with a crossfade. Key, tempo and instrumentation carry on. Each step adds up to 90 s and costs about one 120 s generation.
* **`--extend-method loop`**: crossfaded looping of one 120 s clip; cheap and ideal for rain, room tone and pads.
* **`--loopable`**: blends the end into the start so a clip repeats without a click (the file gets `--crossfade` seconds shorter).

## Automatic acoustic prompt enrichment

Built in (`localaudio/prompts.py`, no LLM involved) and documented for calling agents in `SKILL.md`:

* **SFX**: injects microphone perspective, room acoustics and recording texture matched to the kind of sound (foley, outdoor ambience, UI/sci-fi, vehicles, animals ...), plus `no music` and the `TrackType: SFX` tag.
* **Music**: injects BPM, concrete instrumentation, mood and mix style for the detected genre, plus `no vocals` and `TrackType: Music, VocalType: Instrumental`.
* **Conflict-aware**: anything you already wrote (a cathedral, "tape saturated", 120 BPM, a cello) is left alone; `--no-enrich` uses your prompt verbatim.

The keyword tables and writing guidance are in `references/prompting-sfx.md` and `references/prompting-music.md`, following Stability's own prompting guide.

## Python API

```python
from localaudio import Studio

with Studio() as studio:          # models load on first use; the with-block releases GPU memory
    studio.tts("Hello from my laptop.", voice="af_heart", output="hello.wav")
    studio.sfx("rain on a tin roof", duration=8, output="rain.wav")
    r = studio.music("lofi study beat", duration=60, output="lofi.wav")
    print(r.path, r.duration, r.info["seed"])
```

## Use as an agent skill

The repository root *is* the skill folder: `SKILL.md` (triggers, routing, flags, VRAM rules, prompt rules) plus `scripts/` and `references/`. With Claude Code, copy or clone it to `~/.claude/skills/local-audio-studio/` (or `.claude/skills/` inside a project) and run the installer there. Other agents that read `SKILL.md` can use it the same way. The skill description makes the agent reach for it whenever the user wants local/free/offline voice, sound effects or music.

## How it works

```
scripts/generate_audio.py ──► localaudio.cli ──► Studio ──► ModelManager ──► one engine resident at a time
                                                  │            │               ├─ KokoroEngine   (Kokoro-82M, fp32)
   prompts.py  (acoustic enrichment)  ◄───────────┤            │               └─ Sa3Engine      (small-sfx | small-music)
   textchunk.py (sentence chunking)   ◄───────────┘            └─ unload() + gc.collect() + torch.cuda.empty_cache() on every switch
```

* **Speech:** Kokoro's `KModel`/`KPipeline` are loaded from local files. Scripts are split at sentence boundaries into chunks of 240 characters or fewer (Kokoro is most reliable on 100-200 phoneme tokens), with longer pauses at paragraph breaks.
* **Effects and music:** the official `stable_audio_3` package (MIT, pinned to a reviewed commit) does the sampling: pingpong sampler, 8 steps, CFG 1.0. Its default loader pulls *gated* Hugging Face repositories, so `localaudio/engines/sa3_loader.py` builds the same network from local files instead:
  * the architecture comes from a vendored, inference-only copy of Stability's `model_config.json`;
  * the weights come from the un-gated ComfyUI repackaging `Comfy-Org/stable-audio-3` (`checkpoints/stable_audio_3_small_sfx.safetensors`, `.../small_music.safetensors`, `text_encoders/t5gemma_b_b_ul2.safetensors`);
  * the T5Gemma text encoder is built from its single file and handed to the official conditioner, with tokenization done by the SentencePiece model Stability's own runtimes use (no BOS/EOS, 256 tokens, pad 0);
  * weights are loaded **strictly**: every tensor shape is checked and a mismatch raises an error instead of silently producing noise.
* **Memory on 4 GB cards:** the network is built and cast to fp16 on the CPU before it moves to the GPU (so the 2.3 GB fp32 copy never has to fit in VRAM); the text encoder is moved back to the CPU before sampling; decoding is chunked.
* **Longer than 120 s:** continuation through the official inpainting path or crossfaded looping (see above).
* **Why the checkpoint is 2.27 GB:** Stability's table lists 433M parameters for the generator; the single checkpoint file also contains the SAME-Small autoencoder and is stored in fp32 (567.6M tensors in total when the network is built from the vendored config, which equals 2.27 GB). The 6 GB total in this README is Kokoro (0.36 GB) + two checkpoints (4.54 GB) + the text encoder (about 1.1 GB; run `setup_models.py --dry-run` for exact sizes).

Alternative source (needs a Hugging Face token because the repositories are gated): `python scripts/setup_models.py --source official` downloads Stability's own bundles, the reference layout, which `--sa3-source auto` then prefers.

## Verification status

Be aware of what has and has not been exercised, and update this table after you run it on your hardware.

| Area | Status |
|---|---|
| Pure logic (chunking, enrichment, audio utilities, device policy, installer planning, CLI) | unit-tested (`pytest`, CPU, no models) |
| Stable Audio 3 loading path (config -> network -> strict weight load -> text encoder patches -> `generate()` -> 120 s continuation/loop) | integration-tested with **miniature random-weight models built by the real upstream code** (same architecture classes, tiny sizes) |
| Kokoro path (real `kokoro`, `misaki`, spaCy, espeak) | integration-tested with a random-weight model of the **real architecture** (same file size as the real 327 MB checkpoint) |
| Network built from the vendored config | **567.6M parameters = 2.270 GB in fp32, matching the published 2.27 GB checkpoint size** |
| SentencePiece tokenization vs Stability's reference tokenizer | identical on the sample prompts tested |
| Pinned tokenizer download (GitHub, SHA-256 checked) | exercised for real |
| **Real model weights, real audio quality, real GPU/VRAM numbers** | **not yet run by the author of this repository** - the first run on your machine is the real test: `python scripts/setup_models.py` then `--doctor` |
| `install.ps1` (Windows) | **not yet run** (written and structurally checked only); `install.sh` dry-run paths are exercised |

If the strict loader reports a shape mismatch for the downloaded checkpoint, nothing is generated from wrong weights; use `--source official` or open an issue with the printed message.

## Licenses and attribution

| Part | License |
|---|---|
| This repository's code (the skill wrapper) | **MIT** (`LICENSE`) |
| Kokoro-82M weights and `kokoro` / `misaki` code | **Apache-2.0** |
| Stable Audio 3 Small SFX / Music weights | **Stability AI Community License** (`licenses/STABILITY_AI_COMMUNITY_LICENSE.md`) |
| `stable-audio-3` inference code | MIT |
| T5Gemma text encoder inside Stable Audio 3 | **Gemma Terms of Use** (https://ai.google.dev/gemma/terms) |
| espeak-ng (shared library installed via PyPI, not part of this repo) | GPL-3.0 |

The wrapper's MIT license does not relax the model licenses. In short (not legal advice; read `references/licensing.md` and the license text):

* **Stable Audio 3 weights:** free for research, non-commercial and personal use; **commercial use is free while you and your affiliates earn under US $1M a year, and requires registering** at stability.ai/community-license; above that you need an enterprise license. You may not use the models or their **outputs** to train or improve a foundation generative AI model, and your use must follow Stability's Acceptable Use Policy. If you distribute the models or a product/service built on them you must include the agreement, keep the notice (`NOTICE`) and show **"Powered by Stability AI"**.
* **Kokoro:** Apache-2.0; no restriction on the speech you generate.
* This repo vendors small config files derived from Stability's and Google's model repositories; `NOTICE` describes the modifications as the license requires.

## Limitations

* No voice cloning, dubbing, streaming or sung/lyric vocals; Stable Audio 3 Small makes instrumentals and effects only.
* 120 s per clip; longer audio is stitched and may show a seam (see `references/prompting-music.md` for remedies).
* Long music on a CPU is slow. Stability also publishes much faster **optimized CPU runtimes** (TFLite / CoreML / MLX under `optimized/` in their repository); this project does not use them yet and runs the PyTorch path so that CPU and CUDA share one code path.
* Apple GPUs (MPS) are not used; Intel Macs are not supported by PyTorch 2.7.1 wheels.
* Non-English speech is thinner than English; Japanese and Mandarin need extra packages (`pip install "misaki[ja]"` / `"misaki[zh]"`).
* The Comfy-Org repackaging is community maintained: if its files change layout, the loader's strict check will tell you.

## Development

```bash
pip install pytest                  # after install.sh has installed the requirements
pytest                              # fast suite (CPU, no models needed)
pytest --run-slow                   # also runs the Kokoro end-to-end test (random weights, ~1.5 min)
```

```
SKILL.md              agent skill definition        install.sh / install.ps1   installers
scripts/              generate_audio.py, setup_models.py
localaudio/           cli, studio, manager, device, prompts, textchunk, audio, installer, doctor
localaudio/engines/   kokoro_engine, sa3_engine, sa3_loader
localaudio/assets/    vendored inference configs (see NOTICE)
references/           prompting-sfx, prompting-music, tts-voices, troubleshooting, licensing
examples/             batch_jobs.json, narration.txt        tests/   pytest suite
licenses/ NOTICE LICENSE
```

Contributions are welcome: please keep the loader strict, keep stdout clean, and add a test.
