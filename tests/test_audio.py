import numpy as np
import pytest

from localaudio import audio as A

SR = 8000


def sine(seconds=1.0, freq=220.0, channels=1, amp=0.5, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    wave = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.tile(wave, (channels, 1))


def test_as_channels_first():
    assert A.as_channels_first(np.zeros(10)).shape == (1, 10)
    assert A.as_channels_first(np.zeros((2, 10))).shape == (2, 10)
    with pytest.raises(ValueError):
        A.as_channels_first(np.zeros((2, 2, 2)))


def test_silence_and_duration():
    s = A.silence(0.5, SR, channels=2)
    assert s.shape == (2, SR // 2) and not s.any()
    assert A.duration_seconds(s, SR) == pytest.approx(0.5)


def test_crossfade_length_and_endpoints():
    a, b = sine(1.0), sine(1.0, freq=440)
    n = SR // 4
    out = A.crossfade(a, b, n)
    assert out.shape[1] == a.shape[1] + b.shape[1] - n
    np.testing.assert_allclose(out[:, : a.shape[1] - n], a[:, :-n])  # untouched head
    np.testing.assert_allclose(out[:, a.shape[1]:], b[:, n:])  # untouched tail


def test_linear_crossfade_of_identical_audio_is_transparent():
    a = sine(1.0)
    out = A.crossfade(a, a, SR // 4, curve="linear")
    np.testing.assert_allclose(out[:, :a.shape[1]], a, atol=1e-6)


def test_equal_power_crossfade_has_no_dip():
    noise = np.random.default_rng(0).standard_normal((1, SR)).astype(np.float32) * 0.1
    other = np.random.default_rng(1).standard_normal((1, SR)).astype(np.float32) * 0.1
    n = SR // 2
    out = A.crossfade(noise, other, n)
    seam = out[:, noise.shape[1] - n : noise.shape[1]]
    assert np.sqrt((seam**2).mean()) == pytest.approx(0.1, rel=0.25)


def test_crossfade_zero_and_channel_mismatch():
    a, b = sine(0.2), sine(0.2)
    assert A.crossfade(a, b, 0).shape[1] == a.shape[1] + b.shape[1]
    with pytest.raises(ValueError):
        A.crossfade(sine(0.2, channels=2), b, 10)


def test_fade_edges():
    x = np.ones((1, SR), dtype=np.float32)
    y = A.fade(x, SR, fade_in=0.1, fade_out=0.1)
    assert y[0, 0] == pytest.approx(0.0, abs=1e-6) and y[0, -1] == pytest.approx(0.0, abs=1e-6)
    assert y[0, SR // 2] == pytest.approx(1.0)
    assert x[0, 0] == 1.0  # input not modified


def test_peak_normalize():
    x = sine(0.5, amp=0.1)
    y = A.peak_normalize(x, -1.0)
    assert np.abs(y).max() == pytest.approx(10 ** (-1 / 20), rel=1e-4)
    z = np.zeros((1, 100), dtype=np.float32)
    assert A.peak_normalize(z) is z


def test_trim_silence():
    x = np.concatenate([np.zeros((1, 2000)), sine(0.5), np.zeros((1, 3000))], axis=1).astype(np.float32)
    y = A.trim_silence(x, SR, threshold_db=-50, pad_ms=10)
    assert y.shape[1] < x.shape[1]
    assert y.shape[1] >= int(0.5 * SR)
    silent = np.zeros((1, 500), dtype=np.float32)
    assert A.trim_silence(silent, SR).shape == silent.shape


def test_make_loopable_is_seamless():
    x = sine(3.0, freq=50.0)  # slow wave: easy to check continuity
    y = A.make_loopable(x, SR, crossfade_seconds=0.5)
    assert y.shape[1] == x.shape[1] - int(0.5 * SR)
    wrap = np.abs(y[0, 0] - y[0, -1])
    typical_step = np.abs(np.diff(y[0])).max()
    assert wrap <= typical_step * 1.5 + 1e-3  # jump across the loop point looks like any other sample step


def test_loop_to_duration_exact_length():
    clip = sine(2.0)
    out = A.loop_to_duration(clip, SR, 7.3, crossfade_seconds=0.5)
    assert out.shape[1] == int(round(7.3 * SR))
    assert A.loop_to_duration(clip, SR, 1.0).shape[1] == SR


def test_assemble_speech_adds_pauses():
    seg = [(sine(0.5), 0.25), (sine(0.5), 0.0)]
    out = A.assemble_speech(seg, SR)
    assert out.shape[1] == int(1.0 * SR + 0.25 * SR)
    assert A.assemble_speech([], SR).shape == (1, 0)


def test_write_audio_roundtrip(tmp_path):
    sf = pytest.importorskip("soundfile")
    x = sine(0.5, channels=2, amp=0.3)
    path = A.write_audio(tmp_path / "sub" / "clip.wav", x, SR, "16")
    data, sr = sf.read(path, always_2d=True)
    assert sr == SR and data.shape == (x.shape[1], 2)
    np.testing.assert_allclose(data.T, x, atol=1 / 2**14)
    p24 = A.write_audio(tmp_path / "c24.wav", x, SR, "24")
    assert sf.info(p24).subtype == "PCM_24"
    pf = A.write_audio(tmp_path / "noext", x, SR)
    assert pf.suffix == ".wav"
    with pytest.raises(Exception):
        A.write_audio(tmp_path / "bad.wav", x, SR, "8")


def test_write_audio_clips_overs(tmp_path):
    sf = pytest.importorskip("soundfile")
    x = np.array([[2.0, -2.0, 0.5]], dtype=np.float32)
    path = A.write_audio(tmp_path / "over.wav", x, SR, "16")
    data, _ = sf.read(path)
    assert data.max() <= 1.0 and data.min() >= -1.0


def test_resample_changes_length():
    pytest.importorskip("torchaudio")
    x = sine(1.0)
    y = A.resample(x, SR, 16000)
    assert y.shape[1] == 2 * x.shape[1]
