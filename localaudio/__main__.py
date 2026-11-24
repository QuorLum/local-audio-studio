"""``python -m localaudio ...`` is the same as ``scripts/generate_audio.py``."""

from .cli import main

raise SystemExit(main())
