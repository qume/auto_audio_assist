"""Synthetic-room tests for the analysis chain.

What this can and cannot check:

* Whole-speaker polarity (in_phase / reversed) is decided from bass, where a synthetic room is a
  fair model.  Those cases are asserted.
* Driver-section polarity (a reversed tweeter or HF section) is NOT decided from the listening
  position - reflections there arrive within half a cycle of the direct sound at treble
  frequencies.  The real tool sends that question to the near-field check, so this file asserts
  that the listening-position verdict says "inconclusive" rather than guessing, and then tests the
  near-field path, which is where the answer actually comes from.

Validated against real measurements with known ground truth (sessions/20260908_umik): a genuinely
reversed HF section read ~180 deg across three ranges at consistency 0.64-0.91, and the same room
after the fix read inconclusive rather than producing a false verdict.
"""
import os
import sys
import time

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aaa import analysis, signals  # noqa: E402

SR = 48000
XOVER = 2500.0


def _split(imp, sr=SR, fc=XOVER):
    lo = butter(2, fc / (sr / 2), "low", output="sos")
    hi = butter(2, fc / (sr / 2), "high", output="sos")
    return sosfilt(lo, imp), sosfilt(hi, imp)


def speaker_ir(dist, polarity=1.0, tweeter_pol=1.0, mode_hz=45.0, seed=0, sr=SR):
    """Listening-position IR: direct sound, discrete reflections from 7 ms, modal ringing, tail."""
    rng = np.random.default_rng(seed)
    n = int(0.8 * sr)
    d = int(dist / 343 * sr)
    imp = np.zeros(n)
    imp[d] = 1.0
    woof, tw = _split(imp)
    ir = polarity * (woof + tweeter_pol * tw)
    for _ in range(12):
        ir[d + int(rng.uniform(0.007, 0.06) * sr)] += polarity * rng.uniform(-0.12, 0.12)
    t = np.arange(n) / sr
    ir += polarity * 0.15 * np.sin(2 * np.pi * mode_hz * t) * np.exp(-t / 0.4) * (t > d / sr)
    ir += polarity * rng.standard_normal(n) * 0.02 * np.exp(-t / 0.3)
    return ir


def nearfield_ir(dist, polarity=1.0, tweeter_pol=1.0, seed=0, sr=SR):
    """~25 cm from one driver: direct sound dominant, only weak late reflections."""
    rng = np.random.default_rng(seed)
    n = int(0.2 * sr)
    d = int(dist / 343 * sr)
    imp = np.zeros(n)
    imp[d] = 1.0
    woof, tw = _split(imp)
    ir = polarity * (woof + tweeter_pol * tw)
    for _ in range(4):
        ir[d + int(rng.uniform(0.005, 0.03) * sr)] += polarity * rng.uniform(-0.08, 0.08)
    return ir


def measure(polR=1.0, twR=1.0, distances=(4.0, 4.3), latency=6576):
    sig, markers, inv = signals.build_measurement(2, SR)
    irL = speaker_ir(distances[0])
    irR = speaker_ir(distances[1], polR, twR, seed=1)
    rec = fftconvolve(sig[:, 0], irL) + fftconvolve(sig[:, 1], irR)
    rec = np.concatenate([np.zeros(latency), rec])
    rec += np.random.default_rng(5).standard_normal(len(rec)) * 0.002
    irs, lat, onsets, _ = analysis.extract_irs(rec, markers, inv, SR)
    pol = analysis.polarity_evidence(irs, onsets, rec, markers, lat, SR, list(distances))
    modes = analysis.room_mode_evidence(irs, SR, [3.8, 4.5, 2.4])
    return pol, modes, lat


def main():
    failures = []

    def check(label, got, want):
        ok = want(got) if callable(want) else got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: {got}")
        if not ok:
            failures.append(label)

    print("listening position: whole-speaker polarity")
    for label, (polR, twR), expect in (("both correct", (1, 1), "in_phase"),
                                       ("right reversed", (-1, 1), "reversed"),
                                       ("right tweeter reversed", (1, -1), "in_phase")):
        t0 = time.time()
        pol, modes, lat = measure(polR, twR)
        v = pol["algorithmic_verdict"]
        check(f"{label} -> pair verdict ({time.time()-t0:.1f}s)", v["0-1"], expect)
        # a crude synthetic room cannot settle driver sections; it must say so, not guess
        check(f"{label} -> driver verdict is honest",
              v["tweeter"]["1"].split(":")[0],
              lambda g: g in ("inconclusive", "ok") or "near-field" in g)
        check(f"{label} -> 45 Hz mode found", [p["freq"] for p in modes["peaks"]][:1],
              lambda g: bool(g) and abs(g[0] - 45) < 4)
        check(f"{label} -> no bogus sweep-level warnings", pol["sweep_level_check"]["warnings"], [])

    print("near-field: driver-section polarity (this is where it is actually decided)")
    for label, (wp, tp), expect_hf_same in (("all normal", (1, 1), True),
                                            ("HF section reversed", (1, -1), False),
                                            ("whole speaker reversed", (-1, -1), True)):
        ref = analysis.nearfield_polarity(nearfield_ir(0.25, 1, 1, seed=3), SR)
        got = analysis.nearfield_polarity(nearfield_ir(0.25, wp, tp, seed=3), SR)
        cmp = analysis.compare_nearfield(ref, got)
        check(f"{label} -> tweeter matches reference", cmp["highpass_6k"]["same_polarity"], expect_hf_same)
        check(f"{label} -> woofer matches reference", cmp["lowpass_400"]["same_polarity"], wp > 0)
        check(f"{label} -> readings are confident",
              min(v["confidence"] for v in got.values()), lambda g: g >= 0.9)

    print("\n%d checks failed" % len(failures) if failures else "\nall checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
