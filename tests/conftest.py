"""Shared pytest configuration.

Fast tests need only numpy. Tests that touch PyTorch / Stable Audio 3 / Kokoro
skip themselves when those packages are missing. ``--run-slow`` (or
``LOCALAUDIO_SLOW_TESTS=1``) enables the heavier integration tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow integration tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: slow integration test (enable with --run-slow)")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow") or os.environ.get("LOCALAUDIO_SLOW_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="slow integration test; use --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def torch_stack():
    """PyTorch + the official stable_audio_3 package, or skip."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_audio_3")
    pytest.importorskip("safetensors")
    return torch


@pytest.fixture(scope="session")
def tiny_model(torch_stack):
    from tiny_sa3 import build_tiny_model

    return build_tiny_model()


class FakeCuda:
    """Minimal stand-in for ``torch.cuda`` driven by constants."""

    def __init__(self, available=True, total_gib=4.0, free_gib=3.8, name="Fake GPU 4GB"):
        self._available, self._total, self._free, self._name = available, total_gib, free_gib, name
        self.empty_cache_calls = 0
        self.synchronize_calls = 0
        self.ipc_collect_calls = 0

    def is_available(self):
        return self._available

    def get_device_name(self, _i=0):
        return self._name

    def mem_get_info(self, _i=0):
        gib = 1024**3
        return int(self._free * gib), int(self._total * gib)

    def empty_cache(self):
        self.empty_cache_calls += 1

    def synchronize(self):
        self.synchronize_calls += 1

    def ipc_collect(self):
        self.ipc_collect_calls += 1

    def is_bf16_supported(self):
        return True


class FakeTorch:
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"
    __version__ = "0.0-fake"

    def __init__(self, **cuda_kwargs):
        self.cuda = FakeCuda(**cuda_kwargs)


@pytest.fixture
def fake_torch():
    return FakeTorch
