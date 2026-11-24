#!/usr/bin/env python3
"""Unified entry point: text-to-speech, sound effects and music, 100% local.

    python scripts/generate_audio.py --mode tts   --text "Hello world" -o hello.wav
    python scripts/generate_audio.py --mode sfx   --prompt "door slam" --duration 3 -o slam.wav
    python scripts/generate_audio.py --mode music --prompt "lofi beat" --duration 60 -o beat.wav

Works straight from a checkout (no `pip install` of this project needed) as long
as the dependencies from requirements.txt are installed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localaudio.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
