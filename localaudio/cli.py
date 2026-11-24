"""Command line interface (``python scripts/generate_audio.py`` or ``local-audio``).

stdout carries only the result (the output path, or JSON with ``--json`` /
``--dry-run``); every log line goes to stderr, so the tool is safe to call from
shell pipelines and agents.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from . import __version__
from . import config as C
from . import textchunk as T
from .errors import LocalAudioError, RequestError

EPILOG = """\
examples:
  speech   generate_audio.py --mode tts --text "Welcome back to the channel." --voice af_heart -o intro.wav
  effect   generate_audio.py --mode sfx --prompt "footsteps on wet gravel" --duration 6 -o steps.wav
  ambience generate_audio.py --mode sfx --prompt "rain on a tin roof" --duration 90 --loopable -o rain.wav
  music    generate_audio.py --mode music --prompt "calm acoustic guitar for a travel vlog" --duration 45 -o bed.wav
  preview  generate_audio.py --mode music --prompt "lofi study beat" --dry-run     (no model is loaded)
  setup    python scripts/setup_models.py        (one-time model download)

Speech: Kokoro-82M (Apache-2.0). Effects and music: Powered by Stability AI (Stable Audio 3, Stability AI
Community License) - see NOTICE and licenses/.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_audio.py",
        description="100% local text-to-speech, sound effects and instrumental music.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"local-audio-studio {__version__}")
    p.add_argument("--mode", choices=C.MODES, help="tts = speech (Kokoro), sfx = sound effects/ambience, "
                                                   "music = instrumental music (Stable Audio 3 Small)")
    src = p.add_argument_group("input")
    src.add_argument("--prompt", help="description of the sound/music to generate (sfx, music)")
    src.add_argument("--text", help="text to speak (tts). --prompt is accepted as an alias")
    src.add_argument("--text-file", help="read the text to speak from a file ('-' = stdin)")
    src.add_argument("--batch", metavar="FILE", help="JSON / JSON-Lines file with many jobs (one model load per mode)")

    tts = p.add_argument_group("speech (tts)")
    tts.add_argument("--voice", default=C.DEFAULT_VOICE,
                     help=f"Kokoro voice, or a blend like 'af_heart:0.7,af_bella:0.3' (default {C.DEFAULT_VOICE})")
    tts.add_argument("--speed", type=float, default=1.0, help="speaking speed 0.5-2.0 (default 1.0)")
    tts.add_argument("--lang", help="Kokoro language code (a b e f h i j p z); inferred from the voice by default")
    tts.add_argument("--max-chars", type=int, default=T.DEFAULT_MAX_CHARS,
                     help=f"max characters per synthesized chunk (default {T.DEFAULT_MAX_CHARS})")
    tts.add_argument("--sentence-pause", type=float, default=T.DEFAULT_SENTENCE_PAUSE,
                     help="seconds of silence between sentences")
    tts.add_argument("--paragraph-pause", type=float, default=T.DEFAULT_PARAGRAPH_PAUSE,
                     help="seconds of silence between paragraphs")
    tts.add_argument("--list-voices", action="store_true", help="list available voices and exit")

    snd = p.add_argument_group("sound effects & music (sfx, music)")
    snd.add_argument("--duration", type=float, help="seconds (sfx default 5, music default 30; >120 s is extended)")
    snd.add_argument("--steps", type=int, default=C.SA3_DEFAULT_STEPS,
                     help=f"sampler steps 1-{C.SA3_MAX_STEPS} (default {C.SA3_DEFAULT_STEPS}, pingpong sampler)")
    snd.add_argument("--cfg-scale", type=float, default=C.SA3_DEFAULT_CFG,
                     help="classifier-free guidance (default 1.0 = fastest; >1 enables --negative-prompt)")
    snd.add_argument("--negative-prompt", help="what to avoid (only effective with --cfg-scale > 1)")
    snd.add_argument("--extend-method", choices=("auto", "continue", "loop"), default="auto",
                     help="how to go beyond 120 s: continue = causal continuation, loop = crossfaded looping")
    snd.add_argument("--context", type=float, default=30.0, metavar="SEC",
                     help="seconds of existing audio fed back when continuing past 120 s (default 30)")
    snd.add_argument("--crossfade", type=float, default=2.0, metavar="SEC",
                     help="crossfade length for stitching/looping (default 2)")
    snd.add_argument("--loopable", action="store_true", help="make the result loop seamlessly")

    enr = p.add_argument_group("prompt enrichment (sfx, music)")
    enr.add_argument("--no-enrich", action="store_true", help="use the prompt exactly as written")
    enr.add_argument("--no-tags", action="store_true", help="do not prepend 'TrackType: ...' style tags")
    enr.add_argument("--mic", help="sfx: microphone perspective, e.g. 'close-mic foley recording'")
    enr.add_argument("--room", help="sfx: room acoustics, e.g. 'dry studio acoustics'")
    enr.add_argument("--texture", help="sfx: recording texture, e.g. 'clean high-fidelity'")
    enr.add_argument("--bpm", type=int, help="music: tempo")
    enr.add_argument("--genre", help="music: genre hint used to pick default instrumentation")
    enr.add_argument("--instruments", help="music: concrete instrumentation")
    enr.add_argument("--mood", help="music: mood")
    enr.add_argument("--mix", help="music: mix style, e.g. 'clean studio mix'")
    enr.add_argument("--vocals", action="store_true",
                     help="music: do not add 'no vocals' (the model never produces intelligible lyrics)")
    enr.add_argument("--stem", action="store_true",
                     help="music: isolated instrument / stem (uses the 'TrackType: Instrument' tag)")

    dev = p.add_argument_group("device & models")
    dev.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto",
                     help="auto = CUDA fp16 if >=3.5 GB VRAM, else CPU (default)")
    dev.add_argument("--dtype", choices=("auto", "fp16", "bf16", "fp32"), default="auto",
                     help="sfx/music precision (auto = fp16 on CUDA, fp32 on CPU)")
    dev.add_argument("--models-dir", help="model directory (default ./models or $LOCALAUDIO_MODELS_DIR)")
    dev.add_argument("--sa3-source", choices=("auto", "comfy", "official"), default="auto",
                     help="which downloaded Stable Audio 3 files to use (default auto)")
    dev.add_argument("--seed", type=int, help="random seed (omit or use -1 for random; the used seed is reported)")

    out = p.add_argument_group("output")
    out.add_argument("-o", "--output", help="output file (default ./output/<mode>_<timestamp>.wav)")
    out.add_argument("--normalize", type=float, default=-1.0, metavar="DBFS", help="peak-normalize to this level (default -1)")
    out.add_argument("--no-normalize", action="store_true", help="keep the model's native level")
    out.add_argument("--trim-silence", action="store_true", help="cut leading/trailing silence")
    out.add_argument("--fade-in", type=float, default=0.0, metavar="SEC")
    out.add_argument("--fade-out", type=float, default=0.0, metavar="SEC")
    out.add_argument("--sample-rate", type=int, help="resample the output (e.g. 48000 for video editors)")
    out.add_argument("--bit-depth", choices=("16", "24", "32f"), default="16", help="WAV sample format (default 16)")
    out.add_argument("--json", action="store_true", help="print a JSON result instead of the bare path")
    out.add_argument("--quiet", action="store_true", help="no progress or log output on stderr")
    out.add_argument("--dry-run", action="store_true", help="show the routing, enriched prompt and plan; load nothing")
    out.add_argument("--doctor", action="store_true", help="check the installation and hardware, then exit")
    return p


# --------------------------------------------------------------------------- #
def _read_text(args: argparse.Namespace) -> str:
    if args.text_file:
        if args.text_file == "-":
            return sys.stdin.read()
        with open(args.text_file, "r", encoding="utf-8") as fh:
            return fh.read()
    return args.text or args.prompt or ""


def _post_options(args: argparse.Namespace):
    from .studio import PostOptions

    return PostOptions(
        normalize_db=None if args.no_normalize else args.normalize,
        trim_silence=args.trim_silence,
        fade_in=args.fade_in,
        fade_out=args.fade_out,
        sample_rate=args.sample_rate,
        bit_depth=args.bit_depth,
    )


def _enrich_options(args: argparse.Namespace) -> Dict[str, Any]:
    opts: Dict[str, Any] = {"tags": not args.no_tags}
    if args.mode == "sfx":
        for key in ("mic", "room", "texture"):
            if getattr(args, key):
                opts[key] = getattr(args, key)
    else:
        for key in ("bpm", "genre", "instruments", "mood", "mix"):
            if getattr(args, key):
                opts[key] = getattr(args, key)
        if args.vocals:
            opts["vocals"] = True
        if args.stem:
            opts["stem"] = True
    return opts


def _list_voices(args: argparse.Namespace) -> int:
    from .models import ModelPaths

    paths = ModelPaths(args.models_dir)
    installed = set(paths.installed_voices())
    rows: List[Dict[str, Any]] = []
    for name in C.KOKORO_VOICES:
        rows.append({
            "voice": name,
            "language": C.KOKORO_LANGS.get(name[0], "?"),
            "gender": "female" if name[1:2] == "f" else "male",
            "installed": name in installed,
        })
    extra = sorted(installed - set(C.KOKORO_VOICES))
    rows += [{"voice": n, "language": "?", "gender": "?", "installed": True} for n in extra]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(f"{'voice':<16}{'language':<24}{'gender':<8}installed")
        for r in rows:
            print(f"{r['voice']:<16}{r['language']:<24}{r['gender']:<8}{'yes' if r['installed'] else 'no'}")
        print(f"\ndefault: {C.DEFAULT_VOICE}   (models dir: {paths.root})", file=sys.stderr)
    return 0


def _doctor(args: argparse.Namespace) -> int:
    from .doctor import run_doctor

    return run_doctor(args.models_dir, as_json=args.json, sa3_source=args.sa3_source)


def _make_studio(args: argparse.Namespace):
    from .studio import Studio

    return Studio(args.models_dir, device=args.device, dtype=args.dtype, sa3_source=args.sa3_source, quiet=args.quiet)


def _run_batch(args: argparse.Namespace) -> int:
    from .studio import read_batch_file

    specs = read_batch_file(args.batch)
    with _make_studio(args) as studio:
        results = studio.batch(specs, post=_post_options(args), dry_run=args.dry_run)
    print(json.dumps(results, indent=2))
    return 0 if all(r.get("ok") for r in results) else 1


def run(args: argparse.Namespace) -> int:
    if args.list_voices:
        return _list_voices(args)
    if args.doctor:
        return _doctor(args)
    if args.batch:
        return _run_batch(args)
    if not args.mode:
        raise RequestError("--mode is required (tts, sfx or music). See --help.")

    from .jobs import SoundJob, SpeechJob
    from . import prompts as P

    seed = None if (args.seed is None or args.seed < 0) else args.seed
    text = _read_text(args)
    enrichment = None

    if args.mode == "tts":
        job: Any = SpeechJob(
            text=text, voice=args.voice, speed=args.speed, lang=args.lang, seed=seed,
            max_chars=args.max_chars, sentence_pause=args.sentence_pause, paragraph_pause=args.paragraph_pause,
        ).validate()
    else:
        if not text.strip():
            raise RequestError("--prompt is required for sfx and music")
        prompt = text
        if not args.no_enrich:
            enrichment = P.enrich(args.mode, text, **_enrich_options(args))
            prompt = enrichment.prompt
        duration = args.duration if args.duration is not None else (5.0 if args.mode == "sfx" else 30.0)
        job = SoundJob(
            prompt=prompt, mode=args.mode, duration=duration, steps=args.steps, cfg_scale=args.cfg_scale,
            seed=seed, negative_prompt=args.negative_prompt, extend_method=args.extend_method,
            context_seconds=args.context, crossfade_seconds=args.crossfade, loopable=args.loopable,
        ).validate()

    with _make_studio(args) as studio:
        if args.dry_run:
            plan = studio.plan(args.mode, job, enrichment, args.output, _post_options(args))
            print(json.dumps(plan, indent=2, ensure_ascii=False))
            return 0
        if enrichment is not None and not args.quiet:
            print(f"prompt: {enrichment.prompt}", file=sys.stderr)
        result = studio._render(
            args.mode, job, args.output, _post_options(args),
            {"enrichment": enrichment.to_dict()} if enrichment else None,
        )
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.path)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except LocalAudioError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
