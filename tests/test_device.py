import pytest

from localaudio import config as C
from localaudio import device as D
from localaudio.errors import DeviceError, RequestError


def plan(torch_mod, **kw):
    return D.plan_device(torch_mod=torch_mod, **kw)


def test_auto_prefers_cuda_fp16_on_4gb_card(fake_torch):
    p = plan(fake_torch(total_gib=4.0, free_gib=3.7))
    assert (p.device, p.dtype) == ("cuda", "float16")
    assert p.gpu_name == "Fake GPU 4GB" and p.vram_total_gb == pytest.approx(4.0)


def test_auto_falls_back_to_cpu_when_no_gpu(fake_torch):
    p = plan(fake_torch(available=False))
    assert (p.device, p.dtype) == ("cpu", "float32")
    assert "no CUDA" in p.reason


def test_auto_falls_back_when_vram_below_3_5_gib(fake_torch):
    p = plan(fake_torch(total_gib=3.0, free_gib=2.9))
    assert p.device == "cpu" and "3.0 GiB" in p.reason


def test_auto_falls_back_when_not_enough_free_vram(fake_torch):
    p = plan(fake_torch(total_gib=8.0, free_gib=1.0))
    assert p.device == "cpu" and "free" in p.reason


def test_exactly_3_5_gib_is_enough(fake_torch):
    assert plan(fake_torch(total_gib=3.5, free_gib=3.4)).device == "cuda"


def test_forced_cuda_without_gpu_is_an_error(fake_torch):
    with pytest.raises(DeviceError):
        plan(fake_torch(available=False), requested="cuda")


def test_forced_cuda_on_small_card_warns_but_runs(fake_torch):
    p = plan(fake_torch(total_gib=2.0, free_gib=1.9), requested="cuda")
    assert p.device == "cuda" and "warning" in p.reason


def test_forced_cpu_never_touches_torch():
    p = D.plan_device("cpu")  # no torch_mod needed
    assert (p.device, p.dtype) == ("cpu", "float32")


def test_dtype_rules(fake_torch):
    assert plan(fake_torch(), dtype="bf16").dtype == "bfloat16"
    cpu_fp16 = plan(fake_torch(), requested="cpu", dtype="fp16")
    assert cpu_fp16.dtype == "float32" and "not supported on CPU" in cpu_fp16.reason
    assert plan(fake_torch(), requested="cpu", dtype="bf16").dtype == "bfloat16"
    assert plan(fake_torch(), cuda_default_dtype="float32").dtype == "float32"


def test_invalid_arguments(fake_torch):
    with pytest.raises(RequestError):
        plan(fake_torch(), requested="tpu")
    with pytest.raises(RequestError):
        plan(fake_torch(), dtype="int4")


def test_min_vram_env_override(fake_torch, monkeypatch):
    monkeypatch.setenv(C.ENV_MIN_VRAM, "6")
    assert plan(fake_torch(total_gib=4.0, free_gib=3.9)).device == "cpu"


def test_plan_helpers(fake_torch):
    p = plan(fake_torch())
    assert "cuda/float16" in p.short() and p.is_cuda
    assert p.torch_dtype(fake_torch()) == "float16"
    assert p.to_dict()["device"] == "cuda"


def test_cleanup_runs_gc_and_empty_cache(fake_torch):
    t = fake_torch()
    D.cleanup_memory(t)
    assert t.cuda.empty_cache_calls == 1 and t.cuda.synchronize_calls == 1


def test_cleanup_is_safe_without_cuda_or_torch(fake_torch):
    D.cleanup_memory(fake_torch(available=False))
    D.cleanup_memory(None)  # must not import torch or raise


def test_describe_hardware_never_raises(fake_torch):
    info = D.describe_hardware(fake_torch())
    assert info["cuda"]["name"] == "Fake GPU 4GB"
    assert "python" in info
