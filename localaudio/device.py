"""Device selection, precision policy and strict memory cleanup.

Policy (from the project spec):

* ``auto``: CUDA with fp16 when the GPU has at least 3.5 GiB of VRAM (and
  enough of it is currently free), otherwise CPU with fp32.
* ``cuda``: force the GPU (error if there is none).
* ``cpu``: force the CPU (fp32 by default, bf16 on request).

``torch`` is injected as an argument everywhere so the logic is unit-testable
without PyTorch installed.
"""

from __future__ import annotations

import gc
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from . import config as C
from .errors import DependencyError, DeviceError, RequestError

GIB = 1024**3

DEVICE_CHOICES = ("auto", "cuda", "cpu")
_DTYPE_ALIASES = {
    "auto": None,
    "fp16": "float16",
    "float16": "float16",
    "half": "float16",
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
    "full": "float32",
}
DTYPE_CHOICES = ("auto", "fp16", "bf16", "fp32")


@dataclass(frozen=True)
class DevicePlan:
    device: str  # "cuda" | "cpu"
    dtype: str  # "float16" | "bfloat16" | "float32"
    reason: str
    gpu_name: Optional[str] = None
    vram_total_gb: Optional[float] = None
    vram_free_gb: Optional[float] = None

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    def torch_dtype(self, torch_mod: Any) -> Any:
        return getattr(torch_mod, self.dtype)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def short(self) -> str:
        if self.is_cuda and self.gpu_name:
            free = f", {self.vram_free_gb:.1f} GiB free" if self.vram_free_gb is not None else ""
            return f"cuda/{self.dtype} ({self.gpu_name}{free})"
        return f"{self.device}/{self.dtype}"


def import_torch() -> Any:
    try:
        import torch  # noqa: WPS433

        return torch
    except Exception as exc:  # pragma: no cover - exercised on bare installs
        raise DependencyError(
            "PyTorch could not be imported. Install the project dependencies first "
            "(./install.sh on Linux/macOS, .\\install.ps1 on Windows) "
            f"[{type(exc).__name__}: {exc}]"
        ) from exc


def normalize_dtype(dtype: str) -> Optional[str]:
    key = (dtype or "auto").lower()
    if key not in _DTYPE_ALIASES:
        raise RequestError(f"Unknown dtype {dtype!r}; choose one of {', '.join(DTYPE_CHOICES)}")
    return _DTYPE_ALIASES[key]


def query_cuda(torch_mod: Any) -> Optional[Dict[str, Any]]:
    """Return ``{name, total_gb, free_gb}`` for GPU 0, or ``None`` if unusable."""
    try:
        if not torch_mod.cuda.is_available():
            return None
        name = str(torch_mod.cuda.get_device_name(0))
        try:
            free, total = torch_mod.cuda.mem_get_info(0)
        except Exception:
            props = torch_mod.cuda.get_device_properties(0)
            total = int(props.total_memory)
            free = total
        return {"name": name, "total_gb": total / GIB, "free_gb": free / GIB}
    except Exception:
        return None


def plan_device(
    requested: str = "auto",
    dtype: str = "auto",
    *,
    min_vram_gb: float = C.MIN_VRAM_GB_SA3,
    min_free_gb: Optional[float] = C.MIN_FREE_VRAM_GB_SA3,
    cuda_default_dtype: str = "float16",
    torch_mod: Any = None,
) -> DevicePlan:
    """Decide where and in which precision a model should run."""
    requested = (requested or "auto").lower()
    if requested == "gpu":
        requested = "cuda"
    if requested not in DEVICE_CHOICES:
        raise RequestError(f"Unknown device {requested!r}; choose one of {', '.join(DEVICE_CHOICES)}")
    wanted_dtype = normalize_dtype(dtype)
    min_vram_gb = C.min_vram_gb(min_vram_gb)

    if requested == "cpu":
        return _cpu_plan(wanted_dtype, "CPU requested")

    torch_mod = torch_mod if torch_mod is not None else import_torch()
    info = query_cuda(torch_mod)

    if requested == "cuda":
        if info is None:
            raise DeviceError(
                "--device cuda was requested but no usable CUDA GPU was found. "
                "Use --device auto (falls back to CPU) or install a CUDA build of PyTorch."
            )
        dt = wanted_dtype or cuda_default_dtype
        note = "CUDA requested"
        if info["total_gb"] < min_vram_gb:
            note += f" (warning: only {info['total_gb']:.1f} GiB VRAM, {min_vram_gb:.1f} GiB recommended)"
        return DevicePlan("cuda", dt, note, info["name"], info["total_gb"], info["free_gb"])

    # requested == "auto"
    if info is None:
        return _cpu_plan(wanted_dtype, "no CUDA GPU detected")
    if info["total_gb"] < min_vram_gb:
        return _cpu_plan(
            wanted_dtype,
            f"GPU has {info['total_gb']:.1f} GiB VRAM (< {min_vram_gb:.1f} GiB required)",
            info,
        )
    if min_free_gb is not None and info["free_gb"] < min_free_gb:
        return _cpu_plan(
            wanted_dtype,
            f"only {info['free_gb']:.1f} GiB VRAM free (< {min_free_gb:.1f} GiB needed); "
            "close other GPU programs or use --device cuda to force",
            info,
        )
    dt = wanted_dtype or cuda_default_dtype
    return DevicePlan(
        "cuda",
        dt,
        f"{info['total_gb']:.1f} GiB VRAM >= {min_vram_gb:.1f} GiB",
        info["name"],
        info["total_gb"],
        info["free_gb"],
    )


def _cpu_plan(wanted_dtype: Optional[str], reason: str, info: Optional[dict] = None) -> DevicePlan:
    dt = wanted_dtype or "float32"
    if dt == "float16":
        reason += "; fp16 is not supported on CPU, using fp32"
        dt = "float32"
    kwargs: Dict[str, Any] = {}
    if info:
        kwargs = {
            "gpu_name": info["name"],
            "vram_total_gb": info["total_gb"],
            "vram_free_gb": info["free_gb"],
        }
    return DevicePlan("cpu", dt, reason, **kwargs)


# --------------------------------------------------------------------------- #
# Memory hygiene
# --------------------------------------------------------------------------- #
def cleanup_memory(torch_mod: Any = None) -> None:
    """``gc.collect()`` + ``torch.cuda.empty_cache()`` (the spec's switch ritual).

    Does not import torch if it was never imported, so it is free to call on
    CPU-only / torch-less paths.
    """
    gc.collect()
    torch_mod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    if torch_mod is not None:
        try:
            if torch_mod.cuda.is_available():
                torch_mod.cuda.synchronize()
                torch_mod.cuda.empty_cache()
                if hasattr(torch_mod.cuda, "ipc_collect"):
                    torch_mod.cuda.ipc_collect()
        except Exception:
            pass
    gc.collect()


def cuda_memory_line(torch_mod: Any = None) -> str:
    torch_mod = torch_mod if torch_mod is not None else sys.modules.get("torch")
    try:
        if torch_mod is not None and torch_mod.cuda.is_available():
            used = torch_mod.cuda.memory_allocated() / GIB
            peak = torch_mod.cuda.max_memory_allocated() / GIB
            return f"VRAM allocated {used:.2f} GiB (peak {peak:.2f} GiB)"
    except Exception:
        pass
    return ""


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def total_ram_gb() -> Optional[float]:
    """Best-effort physical RAM in GiB without extra dependencies."""
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/meminfo", "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("MemTotal:"):
                        return int(line.split()[1]) * 1024 / GIB
        elif sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) / GIB
        elif sys.platform == "win32":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return stat.ullTotalPhys / GIB
    except Exception:
        return None
    return None


def describe_hardware(torch_mod: Any = None) -> Dict[str, Any]:
    """Facts for ``--doctor``; never raises."""
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "ram_gb": None,
        "torch": None,
        "cuda": None,
    }
    ram = total_ram_gb()
    info["ram_gb"] = round(ram, 1) if ram else None
    try:
        torch_mod = torch_mod if torch_mod is not None else import_torch()
        info["torch"] = getattr(torch_mod, "__version__", "?")
        cuda = query_cuda(torch_mod)
        if cuda:
            info["cuda"] = {
                "name": cuda["name"],
                "vram_total_gb": round(cuda["total_gb"], 2),
                "vram_free_gb": round(cuda["free_gb"], 2),
            }
    except DependencyError:
        pass
    return info
