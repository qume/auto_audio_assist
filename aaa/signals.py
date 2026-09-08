"""Test-signal generation. Everything is deterministic (fixed seed) so runs are repeatable."""
import numpy as np

SR = 48000


def ess(f1=20.0, f2=20000.0, T=5.0, sr=SR, fade=0.02):
    """Exponential sine sweep and its inverse filter (amplitude-compensated, time-reversed)."""
    n = int(T * sr)
    t = np.arange(n) / sr
    R = np.log(f2 / f1)
    x = np.sin(2 * np.pi * f1 * T / R * (np.exp(t * R / T) - 1))
    nf = int(fade * sr)
    w = np.ones(n)
    w[:nf] = np.sin(np.linspace(0, np.pi / 2, nf)) ** 2
    w[-nf:] = w[:nf][::-1]
    x *= w
    # inverse: reverse and apply -6 dB/oct envelope so deconvolution is flat
    env = np.exp(-t * R / T)
    inv = x[::-1] * env
    inv /= np.abs(np.fft.rfft(x) * np.fft.rfft(inv)).max() / (n)  # rough normalisation
    return x.astype(np.float32), inv.astype(np.float32)


def band_noise(T, lo, hi, sr=SR, seed=1234):
    """Deterministic band-limited noise, unity peak."""
    rng = np.random.default_rng(seed)
    n = int(T * sr)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    X[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(X, n)
    nf = int(0.05 * sr)
    w = np.ones(n)
    w[:nf] = np.linspace(0, 1, nf)
    w[-nf:] = np.linspace(1, 0, nf)
    return (x / np.abs(x).max() * w).astype(np.float32)


def pink_noise(T, sr=SR, seed=99):
    rng = np.random.default_rng(seed)
    n = int(T * sr)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / sr)
    f[0] = f[1]
    X /= np.sqrt(f)
    X[f < 20] = 0
    x = np.fft.irfft(X, n)
    return (0.3 * x / np.abs(x).max()).astype(np.float32)


def build_measurement(channels=2, sr=SR, sweep_T=5.0, level_db=-12.0, gap=1.0, pre=0.5,
                      bursts=4, burst_T=1.5, burst_gap=0.5, lf_band=(40.0, 150.0), warmup_T=3.0):
    """Multichannel test sequence.

    Layout (all offsets in samples are returned in `markers`):
      pre-silence | warm-up noise (all channels) | gap |
      sweep ch0 | gap | sweep ch1 | gap ... | sweep all-in-phase | gap |
      LF bursts: [all channels in phase] gap [ch0 in phase, others inverted] gap  x bursts
    Returns (stereo/multichannel float32 array, markers dict, inverse sweep filter).

    The warm-up matters: amplifiers, AV receivers and network renderers (AirPlay/RAOP, Chromecast)
    mute or ramp for up to a second after a stream starts, which would silently swallow the low
    frequencies of whichever sweep came first.  Leading with silence does NOT wake them.
    """
    sweep, inv = ess(T=sweep_T, sr=sr)
    amp = 10 ** (level_db / 20)
    g = int(gap * sr)
    segs = []
    markers = {"sr": sr, "sweep_len": len(sweep), "channels": channels, "sweeps": {}, "bursts": []}
    pos = int(pre * sr)
    segs.append(np.zeros((pos, channels), np.float32))
    if warmup_T > 0:
        wu = band_noise(warmup_T, 60.0, 6000.0, sr=sr, seed=7) * amp
        blk = np.zeros((len(wu) + g, channels), np.float32)
        for ch in range(channels):
            blk[: len(wu), ch] = wu
        markers["warmup"] = {"start": pos, "len": len(wu)}
        segs.append(blk)
        pos += len(blk)
    for ch in range(channels):
        blk = np.zeros((len(sweep) + g, channels), np.float32)
        blk[: len(sweep), ch] = sweep * amp
        markers["sweeps"][str(ch)] = pos
        segs.append(blk)
        pos += len(blk)
    blk = np.zeros((len(sweep) + g, channels), np.float32)
    for ch in range(channels):
        blk[: len(sweep), ch] = sweep * amp
    markers["sweeps"]["all"] = pos
    segs.append(blk)
    pos += len(blk)
    noise = band_noise(burst_T, *lf_band, sr=sr) * amp * 1.5
    bg = int(burst_gap * sr)
    for i in range(bursts):
        for kind in ("inphase", "antiphase"):
            blk = np.zeros((len(noise) + bg, channels), np.float32)
            for ch in range(channels):
                sign = -1.0 if (kind == "antiphase" and ch > 0) else 1.0
                blk[: len(noise), ch] = noise * sign
            markers["bursts"].append({"kind": kind, "start": pos, "len": len(noise)})
            segs.append(blk)
            pos += len(blk)
    segs.append(np.zeros((int(0.5 * sr), channels), np.float32))
    sig = np.concatenate(segs)
    markers["total_len"] = len(sig)
    return sig, markers, inv


def build_single_sweep(channel, channels=2, sr=SR, sweep_T=5.0, level_db=-12.0, pre=0.5, gap=1.0,
                       warmup_T=2.0):
    """One sweep on one channel, for near-field per-driver measurements.
    Returns (signal, markers, inverse filter) in the same shape `analysis.extract_irs` expects."""
    sweep, inv = ess(T=sweep_T, sr=sr)
    amp = 10 ** (level_db / 20)
    g = int(gap * sr)
    segs = [np.zeros((int(pre * sr), channels), np.float32)]
    pos = int(pre * sr)
    markers = {"sr": sr, "sweep_len": len(sweep), "channels": channels, "sweeps": {}, "bursts": []}
    if warmup_T > 0:
        wu = band_noise(warmup_T, 60.0, 6000.0, sr=sr, seed=7) * amp
        blk = np.zeros((len(wu) + g, channels), np.float32)
        blk[: len(wu), channel] = wu
        markers["warmup"] = {"start": pos, "len": len(wu)}
        segs.append(blk)
        pos += len(blk)
    blk = np.zeros((len(sweep) + g, channels), np.float32)
    blk[: len(sweep), channel] = sweep * amp
    markers["sweeps"]["0"] = pos            # '0' so extract_irs uses it as the reference sweep
    segs.append(blk)
    pos += len(blk)
    sig = np.concatenate(segs)
    markers["total_len"] = len(sig)
    return sig, markers, inv
