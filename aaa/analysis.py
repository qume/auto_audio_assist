"""Measurement analysis: deconvolution, frequency response, polarity checks, room modes.

All functions are pure numpy/scipy and produce plain-dict "evidence" that is saved to the
session and handed to the LLM.  Designed for N channels; the UI currently uses 2.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, fftconvolve, find_peaks

C_SOUND = 343.0


# ---------------------------------------------------------------- basic helpers
def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


def bandpass(x, lo, hi, sr, order=4):
    nyq = sr / 2
    lo = max(lo, 1.0)
    hi = min(hi, nyq * 0.99)
    sos = butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def lowpass(x, fc, sr, order=4):
    sos = butter(order, fc / (sr / 2), btype="low", output="sos")
    return sosfiltfilt(sos, x)


def highpass(x, fc, sr, order=4):
    sos = butter(order, fc / (sr / 2), btype="high", output="sos")
    return sosfiltfilt(sos, x)


def log_grid(f1=20.0, f2=20000.0, ppo=48):
    n = int(np.log2(f2 / f1) * ppo) + 1
    return f1 * 2 ** (np.arange(n) / ppo)


def smooth_response(freqs, mag_db, grid, octaves):
    """Fractional-octave smoothing of a dB response onto a log grid (power average)."""
    p = 10 ** (mag_db / 10)
    out = np.empty(len(grid))
    half = 2 ** (octaves / 2)
    for i, fc in enumerate(grid):
        m = (freqs >= fc / half) & (freqs <= fc * half)
        if not m.any():
            j = np.searchsorted(freqs, fc)
            m = slice(max(j - 1, 0), j + 1)
        out[i] = 10 * np.log10(np.mean(p[m]) + 1e-20)
    return out


def freq_response(ir, sr, window_s=0.5, grid=None, octaves=1 / 6):
    """IR -> smoothed magnitude response (dB) on a log grid."""
    n = int(window_s * sr)
    seg = ir[:n].copy()
    fade = int(0.02 * sr)
    if len(seg) > fade:
        seg[-fade:] *= np.linspace(1, 0, fade)
    nfft = max(1 << (len(seg) - 1).bit_length(), 1 << 15)
    X = np.fft.rfft(seg, nfft)
    f = np.fft.rfftfreq(nfft, 1 / sr)
    grid = log_grid() if grid is None else grid
    return grid, smooth_response(f, db(X), grid, octaves)


# ---------------------------------------------------------------- deconvolution
def deconvolve(rec, inv):
    """Full-length deconvolution of a recording with the inverse sweep filter."""
    return fftconvolve(rec, inv, mode="full")


def find_direct(full, lo, hi, sr, band=(200.0, 8000.0), thresh=0.3):
    """Index of the direct-sound peak in full[lo:hi]: first envelope crossing of `thresh`*max on a
    band-limited copy (removes LF mode ringing), then the strongest sample within the next 4 ms."""
    seg = bandpass(full[lo:hi], band[0], band[1], sr)
    a = np.abs(seg)
    first = int(np.argmax(a > thresh * a.max()))
    w = int(0.004 * sr)
    pk = first + int(np.argmax(a[first: first + w]))
    return lo + pk


def extract_irs(rec, markers, inv, sr, search_s=None, pre_ms=5.0, ir_len_s=0.7):
    """Deconvolve the whole recording once and cut out one IR per sweep.

    The system latency (playback + capture chain) is found from the first sweep and then
    every other IR is searched in a +-30 ms window around its expected position so that
    different speaker distances are preserved (relative timing between channels matters).
    """
    full = deconvolve(rec, inv)
    L = markers["sweep_len"]
    first = markers["sweeps"]["0"]
    # peak of the deconvolved sweep for a sweep starting at s is near s + L - 1 + latency
    base = first + L - 1
    # unknown playback latency (ssh / manual playback): search the whole rest of the recording;
    # find_direct returns the FIRST significant arrival, i.e. the channel-0 sweep even though later
    # sweeps are also in the window
    hi = len(rec) if search_s is None else base + int(search_s * sr)
    latency = find_direct(full, base, hi, sr) - base
    pre = int(pre_ms / 1000 * sr)
    n = int(ir_len_s * sr)
    irs, onsets = {}, {}
    tol = int(0.03 * sr)
    for key, start in markers["sweeps"].items():
        exp = start + L - 1 + latency
        lo = max(exp - tol, 0)
        pk = find_direct(full, lo, exp + tol, sr)
        onsets[key] = (pk - exp) / sr * 1000.0  # ms relative to channel-0 arrival
        cut = full[pk - pre: pk - pre + n]
        if len(cut) < n:
            cut = np.pad(cut, (0, n - len(cut)))
        irs[key] = cut.astype(np.float64)
    return irs, latency, onsets, full


# ---------------------------------------------------------------- polarity
def onset_index(ir, sr, thresh=0.25):
    a = np.abs(ir)
    pk = int(np.argmax(a))
    lim = thresh * a[pk]
    i = pk
    back = int(0.003 * sr)
    while i > max(pk - back, 0) and a[i - 1] > lim:
        i -= 1
    return i, pk


def dominant_sign(ir, sr, band, pre_ms=1.0, post_ms=2.5):
    """Sign of the dominant peak of the band-limited direct sound."""
    x = bandpass(ir, band[0], band[1], sr)
    _, pk = onset_index(x, sr)
    lo = max(pk - int(pre_ms / 1000 * sr), 0)
    hi = pk + int(post_ms / 1000 * sr)
    seg = x[lo:hi]
    j = int(np.argmax(np.abs(seg)))
    v = seg[j]
    return int(np.sign(v)), float(np.abs(v) / (np.abs(x).max() + 1e-12))


def relative_polarity(ir_a, ir_b, sr, band=(80.0, 1200.0), win_ms=8.0):
    """Cross-correlate the band-limited direct sound of two IRs after aligning onsets.
    Returns (+1/-1, confidence 0..1, lag_ms)."""
    a = bandpass(ir_a, *band, sr)
    b = bandpass(ir_b, *band, sr)
    pa = int(np.argmax(np.abs(a)))
    pb = int(np.argmax(np.abs(b)))
    w = int(win_ms / 1000 * sr)
    sa = a[max(pa - w, 0): pa + w]
    sb = b[max(pb - w, 0): pb + w]
    n = min(len(sa), len(sb))
    sa, sb = sa[:n], sb[:n]
    cc = np.correlate(sa, sb, mode="full")
    k = int(np.argmax(np.abs(cc)))
    val = cc[k]
    norm = np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2)) + 1e-12
    return int(np.sign(val)), float(abs(val) / norm), (k - (n - 1)) / sr * 1000.0


def burst_test(rec, markers, latency, sr, band=(40.0, 150.0)):
    """Compare LF energy of in-phase vs anti-phase noise bursts (needs roughly centred mic)."""
    x = bandpass(rec, band[0], band[1], sr)
    res = {"inphase": [], "antiphase": []}
    for b in markers["bursts"]:
        s = b["start"] + latency + int(0.1 * sr)
        e = b["start"] + latency + b["len"] - int(0.1 * sr)
        if e <= s or e > len(x):
            continue
        res[b["kind"]].append(float(db(np.sqrt(np.mean(x[s:e] ** 2)))))
    ip = float(np.median(res["inphase"])) if res["inphase"] else float("nan")
    ap = float(np.median(res["antiphase"])) if res["antiphase"] else float("nan")
    return {"inphase_db": ip, "antiphase_db": ap, "diff_db": ip - ap,
            "inphase_runs": res["inphase"], "antiphase_runs": res["antiphase"]}


def crossover_notch(grid, resp, lo=800.0, hi=8000.0, min_depth=5.0):
    """Look for a narrow dip in the 1/6-oct response typical of a reversed tweeter."""
    m = (grid >= lo) & (grid <= hi)
    g, r = grid[m], resp[m]
    base = smooth_response(g, r, g, 1.5)  # wide trend
    excess = r - base
    idx, props = find_peaks(-excess, prominence=min_depth)
    if len(idx) == 0:
        return None
    j = idx[int(np.argmax(props["prominences"]))]
    return {"freq": float(g[j]), "depth_db": float(-excess[j]), "prominence_db": float(props["prominences"].max())}


def lf_sum_test(irs, sr, f_lo=25.0, f_hi=70.0):
    """Very-low-frequency sum test: level of the all-channels sweep vs the power sum of the individual
    channel sweeps.  Wavelengths here are so long that mic placement barely matters: in phase gives
    about +3 dB (coherent sum over power sum), reversed gives a deep loss.  Room modes add scatter."""
    if "all" not in irs:
        return None
    grid = log_grid(20, 200, 24)
    resp = {k: freq_response(ir, sr, 0.5, grid, 1 / 3)[1] for k, ir in irs.items()}
    m = (grid >= f_lo) & (grid <= f_hi)
    singles = [resp[k][m] for k in irs if k != "all"]
    psum = 10 * np.log10(np.sum([10 ** (r / 10) for r in singles], axis=0))
    diff = resp["all"][m] - psum
    return {"band_hz": [f_lo, f_hi], "all_minus_power_sum_db_mean": round(float(np.mean(diff)), 1),
            "all_minus_power_sum_db_median": round(float(np.median(diff)), 1),
            "per_freq": [[round(float(f), 1), round(float(d), 1)] for f, d in zip(grid[m], diff)],
            "expected_if_in_phase_db": "about +3 (range 0..+3)", "expected_if_reversed_db": "well below -3"}


def polarity_evidence(irs, onsets, rec, markers, latency, sr, distances=None):
    """Assemble all polarity-related evidence for N channels (channel '0' is the reference)."""
    chans = [k for k in irs if k != "all"]
    ev = {"channels": {}, "pairs": {}, "burst_test": None, "lf_sum_test": lf_sum_test(irs, sr), "notes": []}
    for k in chans:
        ir = irs[k]
        lf_sign, lf_str = dominant_sign(ir, sr, (150.0, 800.0))
        hf_sign, hf_str = dominant_sign(ir, sr, (3000.0, 10000.0))
        grid, resp = freq_response(ir, sr, window_s=0.05, octaves=1 / 6)  # short window: direct sound
        ev["channels"][k] = {
            "arrival_offset_ms": round(onsets[k], 2),
            "lf_sign": lf_sign, "lf_peak_rel": round(lf_str, 2),
            "hf_sign": hf_sign, "hf_peak_rel": round(hf_str, 2),
            "lf_hf_relation": "same" if lf_sign == hf_sign else "opposite",
            "crossover_notch": crossover_notch(grid, resp),
        }
    for k in chans[1:]:
        sign, conf, lag = relative_polarity(irs["0"], irs[k], sr)
        hs, hc, _ = relative_polarity(irs["0"], irs[k], sr, band=(3000.0, 10000.0), win_ms=2.0)
        ms, mc, _ = relative_polarity(irs["0"], irs[k], sr, band=(800.0, 2500.0), win_ms=4.0)
        exp_lag = None
        if distances and len(distances) > int(k):
            exp_lag = round((distances[int(k)] - distances[0]) / C_SOUND * 1000.0, 2)
        ev["pairs"][f"0-{k}"] = {"relative_sign": sign, "correlation": round(conf, 3),
                                 "relative_sign_mid_800_2500": ms, "correlation_mid": round(mc, 3),
                                 "relative_sign_hf_3k_10k": hs, "correlation_hf": round(hc, 3),
                                 "measured_arrival_diff_ms": round(onsets[k], 2),
                                 "expected_arrival_diff_ms_from_distances": exp_lag}
    ev["burst_test"] = burst_test(rec, markers, latency, sr)
    # L/R channel swap check: the user's distances predict which speaker's sound arrives first
    if distances and len(distances) >= 2:
        exp = (distances[1] - distances[0]) / C_SOUND * 1000.0
        meas = onsets.get("1", 0.0)
        if abs(exp) > 0.6 and abs(meas) > 0.6 and np.sign(exp) != np.sign(meas):
            ev["channel_swap_suspected"] = True
            ev["notes"].append(f"Channel 1 was expected to arrive {exp:+.1f} ms relative to channel 0 from your distances but "
                               f"arrived {meas:+.1f} ms. Either the distance estimate is off or LEFT/RIGHT are swapped between "
                               "source and speakers. Confirm by ear with the 'Identify channels' test before touching cables.")
        else:
            ev["channel_swap_suspected"] = False
    if distances and len(distances) >= 2:
        pd = abs(distances[1] - distances[0])
        ev["burst_test"]["path_difference_m"] = round(pd, 2)
        if pd > 0.4:
            ev["notes"].append("Mic is off-centre by >0.4 m path difference: burst sum test is low confidence, "
                               "rely on the aligned cross-correlation sign.")
    # algorithmic verdicts
    verdict = {}
    for k in chans[1:]:
        p = ev["pairs"][f"0-{k}"]
        bt = ev["burst_test"]
        votes = []
        if p["correlation"] > 0.3:
            votes.append(p["relative_sign"])
        if bt and not np.isnan(bt["diff_db"]) and abs(bt["diff_db"]) > 2.5 and len(chans) == 2 \
                and bt.get("path_difference_m", 0) <= 0.4:
            votes.append(1 if bt["diff_db"] > 0 else -1)
        st = ev["lf_sum_test"]
        if st and len(chans) == 2:
            d = st["all_minus_power_sum_db_median"]
            if d >= 0.0:
                votes.append(1)
            elif d <= -4.0:
                votes.append(-1)
        if not votes:
            verdict[f"0-{k}"] = "inconclusive"
        elif all(v == 1 for v in votes):
            verdict[f"0-{k}"] = "in_phase"
        elif all(v == -1 for v in votes):
            verdict[f"0-{k}"] = "reversed"
        else:
            verdict[f"0-{k}"] = "conflicting"
            ev["notes"].append("Indicators disagree. Most reliable resolution: swap + and - on ONE speaker, re-measure, "
                               "and keep the wiring that gives the higher LF sum test and burst test values.")
    # tweeter/woofer consistency: LF band agreement vs HF band agreement between channel pairs
    notch = {k: ev["channels"][k]["crossover_notch"] for k in chans}
    tw = {k: "ok" for k in chans}
    for k in chans[1:]:
        p = ev["pairs"][f"0-{k}"]
        if p["correlation"] > 0.3 and p["correlation_hf"] > 0.3 and p["relative_sign"] != p["relative_sign_hf_3k_10k"]:
            # one speaker's tweeter is wired opposite to the other's.  Which one?  The one with the
            # deeper crossover notch is the likelier culprit; otherwise report both as suspect.
            d0 = notch["0"]["depth_db"] if notch["0"] else 0.0
            dk = notch[k]["depth_db"] if notch[k] else 0.0
            if dk >= d0 + 3:
                tw[k] = "tweeter_reversed_relative_to_ch0 (deeper crossover notch here)"
            elif d0 >= dk + 3:
                tw["0"] = f"tweeter_reversed_relative_to_ch{k} (deeper crossover notch here)"
            else:
                tw[k] = "tweeter_polarity_differs_from_ch0 (cannot tell which speaker; swap one jumper and re-measure)"
                tw["0"] = tw[k].replace("ch0", f"ch{k}")
        if p["correlation"] > 0.3 and p["correlation_mid"] > 0.3 and p["relative_sign"] != p["relative_sign_mid_800_2500"] \
                and p["relative_sign_mid_800_2500"] == p["relative_sign_hf_3k_10k"]:
            tw[k] = "mid+tweeter_polarity_differs_from_ch0 (bi-wire/jumper on the HF section reversed)"
    if all(n is not None and n["depth_db"] >= 8 for n in notch.values()):
        for k in chans:
            tw[k] += "; both channels show a crossover notch (design or both tweeters reversed)"
    verdict["tweeter"] = tw
    ev["algorithmic_verdict"] = verdict
    return ev


# ---------------------------------------------------------------- room modes
def predicted_modes(dims, n_max=3):
    out = []
    names = ["length", "width", "height"]
    for name, L in zip(names, dims):
        if not L or L <= 0:
            continue
        for n in range(1, n_max + 1):
            out.append({"freq": round(C_SOUND / (2 * L) * n, 1), "type": f"axial {name} n={n}"})
    return sorted(out, key=lambda d: d["freq"])


def room_mode_evidence(irs, sr, dims=None, f_lo=25.0, f_hi=300.0, min_excess=5.0, max_peaks=5):
    """Find dominant LF peaks/dips in the (long-window) response, per channel and combined."""
    grid = log_grid(20, 20000, 48)
    resp = {}
    for k, ir in irs.items():
        _, r = freq_response(ir, sr, window_s=0.5, grid=grid, octaves=1 / 12)
        resp[k] = r
    m = (grid >= f_lo) & (grid <= f_hi)
    g = grid[m]
    # analyse the 'all' response if present else channel 0; also each channel for consistency
    chans = [k for k in resp if k != "all"]
    out = {"grid": g.round(2).tolist(), "responses": {}, "peaks": [], "dips": [], "predicted": predicted_modes(dims or []),
           "analysed": "power average of individual channel responses (immune to inter-channel polarity errors)"}
    for k, r in resp.items():
        out["responses"][k] = np.round(r[m], 2).tolist()
    r = 10 * np.log10(np.mean([10 ** (resp[k][m] / 10) for k in chans], axis=0))
    out["responses"]["avg"] = np.round(r, 2).tolist()
    base = smooth_response(g, r, g, 1.0)
    excess = r - base
    pk, props = find_peaks(excess, prominence=2.0)
    cands = []
    for i, p in zip(pk, props["prominences"]):
        if excess[i] < min_excess:
            continue
        # -3 dB bandwidth for Q estimate
        lvl = excess[i] - 3
        a = i
        while a > 0 and excess[a] > lvl:
            a -= 1
        b = i
        while b < len(g) - 1 and excess[b] > lvl:
            b += 1
        bw = max(g[b] - g[a], g[i] * 0.05)
        q = float(np.clip(g[i] / bw, 1.5, 12.0))
        per_ch = {c: round(float((resp[c][m] - smooth_response(g, resp[c][m], g, 1.0))[i]), 1) for c in resp if c != "all"}
        match = None
        for pm in out["predicted"]:
            if abs(pm["freq"] - g[i]) / g[i] < 0.08:
                match = pm["type"]
                break
        cands.append({"freq": round(float(g[i]), 1), "excess_db": round(float(excess[i]), 1),
                      "prominence_db": round(float(p), 1), "q_est": round(q, 1),
                      "per_channel_excess_db": per_ch, "predicted_match": match})
    cands.sort(key=lambda d: -d["excess_db"])
    out["peaks"] = cands[:max_peaks]
    dk, dprops = find_peaks(-excess, prominence=6.0)
    out["dips"] = [{"freq": round(float(g[i]), 1), "depth_db": round(float(-excess[i]), 1)} for i in dk][:4]
    out["excess_db"] = np.round(excess, 2).tolist()
    # broad tonal balance: octave-band mean levels relative to the 200-2000 Hz midband (catches wide
    # humps/troughs that a peak detector against a 1-octave trend cannot see)
    full = 10 * np.log10(np.mean([10 ** (resp[k] / 10) for k in chans], axis=0))
    ref = float(np.mean(full[(grid >= 200) & (grid <= 2000)]))
    bands = {}
    for fc in (31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000):
        mm = (grid >= fc / np.sqrt(2)) & (grid < fc * np.sqrt(2))
        bands[str(fc)] = round(float(np.mean(full[mm]) - ref), 1)
    out["octave_band_levels_re_midband_db"] = bands
    return out


def corrections_from_modes(modes, max_cut_db=10.0):
    return [{"type": "peaking", "freq_hz": p["freq"], "gain_db": -round(min(p["excess_db"], max_cut_db), 1),
             "q": p["q_est"], "reason": f"room mode peak +{p['excess_db']} dB" +
             (f" ({p['predicted_match']})" if p.get("predicted_match") else "")}
            for p in modes["peaks"]]


def compare_runs(irs_prev, irs_now, sr):
    """Per-channel, per-band change between two runs taken from the same mic position: which speaker
    (and which driver section) changed polarity.  Returns {channel: {band: (sign, corr)}}."""
    out = {}
    for k in irs_now:
        if k == "all" or k not in irs_prev:
            continue
        out[k] = {}
        for name, band, w in (("lf_80_1200", (80.0, 1200.0), 8.0), ("mid_800_2500", (800.0, 2500.0), 4.0), ("hf_3k_10k", (3000.0, 10000.0), 2.0)):
            s, c, _ = relative_polarity(irs_prev[k], irs_now[k], sr, band=band, win_ms=w)
            out[k][name] = {"sign": s, "corr": round(c, 2)}
    return out
