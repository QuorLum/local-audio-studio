#!/usr/bin/env python3
"""Download the model weights for local-audio-studio (interactive or scripted).

    python scripts/setup_models.py                   # menu in a terminal, everything otherwise
    python scripts/setup_models.py --only tts        # just speech (~0.4 GB)
    python scripts/setup_models.py --only sfx,music  # sound effects + music (~5 GB)
    python scripts/setup_models.py --dry-run         # what would be downloaded, and how much
    python scripts/setup_models.py --check           # verify an existing installation

No Hugging Face token is needed for the default sources. See --help.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from localaudio.installer import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
