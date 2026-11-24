"""local-audio-studio: 100% local text-to-speech, sound effects and music.

The package is deliberately light to import: nothing heavy (torch, transformers,
kokoro, stable_audio_3) is imported until an engine is actually loaded, so
``--help``, ``--dry-run`` and the prompt/chunking helpers work on a bare Python.

Typical use from Python::

    from localaudio import Studio

    with Studio() as studio:
        studio.tts("Hello from my laptop.", voice="af_heart", output="hello.wav")
        studio.sfx("rain on a tin roof", duration=8, output="rain.wav")
        studio.music("lofi study beat", duration=60, output="lofi.wav")
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["Studio", "__version__"]


def __getattr__(name: str):  # PEP 562 lazy attribute access
    if name == "Studio":
        from .studio import Studio

        return Studio
    raise AttributeError(f"module 'localaudio' has no attribute {name!r}")
