"""Exception types. Each carries the process exit code the CLI should use."""

from __future__ import annotations


class LocalAudioError(Exception):
    """Base class for every error this project raises on purpose."""

    exit_code = 1


class RequestError(LocalAudioError):
    """The caller asked for something invalid (bad flag value, empty prompt, ...)."""

    exit_code = 2


class ModelsMissingError(LocalAudioError):
    """Required model files are not on disk. Run scripts/setup_models.py."""

    exit_code = 3


class DependencyError(LocalAudioError):
    """A Python dependency or system library is missing or unusable."""

    exit_code = 4


class DeviceError(LocalAudioError):
    """The requested compute device is not available."""

    exit_code = 5


class ModelFormatError(LocalAudioError):
    """Downloaded weights do not match the architecture we expect."""

    exit_code = 6
