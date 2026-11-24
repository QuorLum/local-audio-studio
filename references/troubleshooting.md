# Troubleshooting

Start with the two diagnostics:

```bash
python scripts/generate_audio.py --doctor        # Python, packages, GPU, RAM, disk, model readiness
python scripts/setup_models.py --check           # model files + architecture check (header only)
```

**Contents** - [Install problems](#install-problems) - [Model downloads](#model-downloads) - [GPU and memory](#gpu-and-memory) - [Speed](#speed) - [Audio quality](#audio-quality) - [Platform notes](#platform-notes)

## Install problems

| Symptom | Cause / fix |
|---|---|
| `Python 3.13 ... fail` in `--doctor`, or `misaki` refuses to install | Kokoro's text front-end (misaki 0.9.x) supports Python 3.10-3.12 only. Install 3.12 and re-run `install.sh` / `install.ps1`. |
| `error: The official stable_audio_3 package could not be imported` | Install it: `pip install --no-deps "stable-audio-3 @ git+https://github.com/Stability-AI/stable-audio-3@790793b89ddceea612be9ce42ff6830d18f6fa57"`. Needs `git`. |
| `error: PyTorch could not be imported` | The install did not finish. Re-run the installer. On a GPU machine make sure the CUDA wheel was chosen (`--cuda cu126`), not the CPU one. |
| `pip` tries to replace torch with a different version | Install `torch==2.7.1 torchaudio==2.7.1` first (the installer does), then `pip install -r requirements.txt`. 2.7.1 is the version Stability's package pins and tests. |
| `ERROR: Could not find a version that satisfies ... transformers>=5.8.0` | Old pip or Python < 3.10. `python -m pip install --upgrade pip`, use Python 3.10-3.12. |
| `The spaCy model en_core_web_sm is not installed` | `python scripts/setup_models.py --only tts` (installs it once) or `python -m spacy download en_core_web_sm`. It needs internet only at that moment. |
| `flash_attn not installed, disabling Flash Attention` | Harmless. Stable Audio 3 Small falls back to PyTorch's built-in attention. The message goes to stderr, never stdout. |

## Model downloads

| Symptom | Cause / fix |
|---|---|
| `Kokoro files are missing` / `Stable Audio 3 ... files are missing` | Run the `setup_models.py --only ...` command printed in the error. |
| Download stops or is slow | Re-run the same command: downloads resume. Behind a proxy set `HTTPS_PROXY`. The Hugging Face hub rate-limits anonymous clients; a free token (`HF_TOKEN=hf_...`) raises the limit for the default sources too. |
| `Access to stabilityai/... is gated` | Only with `--source official`. Open the model page, accept the license, then `--hf-token hf_...` or `hf auth login`. The default `--source comfy` never needs a token. |
| `ModelFormatError: ... tensors have a different shape` | The downloaded checkpoint does not match the bundled config (the repack changed). Run `setup_models.py --source official` (needs a token) and open an issue with the printed message. Nothing is silently loaded wrongly: the loader verifies every tensor shape. |
| `ModelFormatError: ... absent from the file` | Truncated or wrong file. `setup_models.py --force --only sfx` (or `music`). |
| Disk full | All models together need about 6 GB (`--dry-run` shows the exact plan). `--models-dir` or `LOCALAUDIO_MODELS_DIR` moves them to another drive. |
| Want it fully offline afterwards | After setup nothing touches the network (the loader blocks Hugging Face lookups while loading). `HF_HUB_OFFLINE=1` is optional extra insurance. |

## GPU and memory

| Symptom | Cause / fix |
|---|---|
| `--device auto` picks the CPU on a GPU machine | The GPU has less than 3.5 GiB of VRAM or less than 2.6 GiB is free right now. Close browsers, games and video apps (Windows reserves VRAM for the desktop), or force it: `--device cuda`. `LOCALAUDIO_MIN_VRAM_GB` changes the threshold. |
| `--device cuda ... no usable CUDA GPU` | CPU-only PyTorch is installed, or the NVIDIA driver is missing/old. `nvidia-smi` must work. Reinstall with `./install.sh --cuda cu126` (try `cu118` for older drivers). |
| `CUDA out of memory` | Shorter `--duration` (peak VRAM grows with length; Stability measured about 2.4 GB at 120 s on an H200 with unchunked decoding), stop other GPU programs, run one process at a time, or use `--device cpu`. Never run two generations in parallel on a 4 GB card. |
| Process killed / `MemoryError` on CPU | In CPU mode Stable Audio 3 holds the 2.3 GB network plus the 1.1 GB fp32 text encoder in RAM (about 4 GB with working memory; our estimate, not a measurement). 8 GB or more is recommended; close other apps. On a GPU machine the text encoder is kept in bf16 and the peak is lower. |

How the tool keeps a 4 GB card safe: the model is built and cast to fp16 on the CPU before it moves to the GPU; the text encoder is moved back to the CPU before sampling; only one model family is loaded at a time and `gc.collect()` + `torch.cuda.empty_cache()` run on every switch; chunked decoding keeps the final decode small.

## Speed

* Speech is fast on any CPU. Effects of a few seconds are quick on a CPU too.
* **Long music on a CPU is the slow case** (a full 120 s clip with the 433M-parameter transformer). Ask for 30-60 s first, loop it (`--loopable`), or use a GPU. We have not benchmarked PyTorch-on-CPU numbers; Stability reports 5.9 s for 120 s on an Apple CPU, but with their separate optimized CoreML/TFLite runtime (see `optimized/` in github.com/Stability-AI/stable-audio-3), which this project does not use.
* Past 120 s each continuation step costs about one more 120 s generation.
* `--steps` is fixed by design at 8 (1-12 allowed); fewer steps are faster but rougher.

## Audio quality

| Symptom | Fix |
|---|---|
| Effect sounds synthetic or vague | Rewrite the prompt as source + action + production (see `prompting-sfx.md`); try 3-4 seeds. |
| Music has no clear tempo or wanders | Add a BPM and 3-5 concrete instruments (see `prompting-music.md`). |
| Audible seam after 120 s | Raise `--crossfade` to 4-6, change `--seed`, or use `--extend-method loop` for textures. |
| Speech mispronounces a word | Respell it or use `[word](/phonemes/)` markup (see `tts-voices.md`). |
| Speech rushes | Lower `--max-chars` (e.g. 160) or `--speed 0.9`. |
| Speech sounds weak on a very short line | Put it in the same paragraph as a neighbouring sentence. |
| Clipping / too loud or quiet | Output is peak-normalized to -1 dBFS; use `--normalize -3` or `--no-normalize`. |
| Wrong sample rate in the video editor | `--sample-rate 48000` (resamples). Native: 24 kHz mono speech, 44.1 kHz stereo effects/music. |
| Click at the start/end of a clip | A 5 ms edge fade is applied by default; use `--fade-in/--fade-out` for more. |

## Platform notes

* **Windows:** use `install.ps1` (PowerShell). If scripts are blocked: `powershell -ExecutionPolicy Bypass -File .\install.ps1`. WSL2 with an NVIDIA GPU also works: install the Windows NVIDIA driver, then use `install.sh` inside WSL2 (keep the repo on the Linux filesystem for speed).
* **macOS:** Apple Silicon runs on the CPU (the MPS backend is not used). Intel Macs are not supported by PyTorch 2.7.1 wheels.
* **Linux:** nothing special; `nvidia-smi` must work for GPU use.
* **Hindi / Spanish / French / Italian / Portuguese** speech needs the espeak-ng library, which `pip install espeakng-loader` provides (already in `requirements.txt` via `misaki[en]`). `--doctor` reports it.
* **Japanese / Mandarin** speech: `pip install "misaki[ja]"` / `pip install "misaki[zh]"`.
