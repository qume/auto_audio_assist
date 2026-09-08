"""Command-line measurement runner (no GUI): same signals, analysis and evidence as the app.

Example:
  python3 -m aaa.cli --sink ssh:lunar@media:default --source alsa:hw:AppleJ415HPAI,0 \
      --dist 4.5 4.5 --dims 5 4 2.4 --session sessions/mytest [--llm]
"""
import argparse
import json
import os
import time
import numpy as np

from . import audio_io, signals, analysis, plots, prompts
from .llm import LLM
from .session import Session

SR = 48000


def run_measurement(sink, source, sess, dist=None, dims=None, alsa=None, mic_profile=None, label=""):
    rundir, n = sess.new_run()
    sig, markers, inv = signals.build_measurement(2, SR)
    ch, fmt = alsa or (None, None)
    mono, _ = audio_io.play_and_record(sig, sink, source, rundir, SR, 1.5, ch, fmt)
    irs, lat, onsets, full = analysis.extract_irs(mono, markers, inv, SR)
    pol = analysis.polarity_evidence(irs, onsets, mono, markers, lat, SR, dist)
    modes = analysis.room_mode_evidence(irs, SR, dims)
    grid = analysis.log_grid()
    comp = np.interp(np.log10(grid), np.log10([p[0] for p in mic_profile]), [p[1] for p in mic_profile]) if mic_profile else 0
    fr = {k: analysis.freq_response(ir, SR, 0.5, grid, 1 / 6)[1] - comp for k, ir in irs.items()}
    ref = float(np.mean(fr["0"][(grid >= 200) & (grid <= 2000)]))
    fr = {k: v - ref for k, v in fr.items()}
    g = np.array(modes["grid"])
    c2 = np.interp(np.log10(g), np.log10([p[0] for p in mic_profile]), [p[1] for p in mic_profile]) if mic_profile else 0
    for k in modes["responses"]:
        modes["responses"][k] = (np.array(modes["responses"][k]) - c2 - ref).round(2).tolist()
    corr = analysis.corrections_from_modes(modes)
    pngs = {
        "fr": plots.png_frequency_response(os.path.join(rundir, "fr.png"), grid, fr, f"Run {n} {label}: frequency response (1/6 oct)",
                                           labels={"0": "Left", "1": "Right", "all": "Both"}),
        "lf": plots.png_frequency_response(os.path.join(rundir, "lf.png"), g, modes["responses"], f"Run {n} {label}: 25-300 Hz, 1/12 oct",
                                           marks=[(p["freq"], f"{p['freq']} Hz +{p['excess_db']} dB") for p in modes["peaks"]], xlim=(25, 300),
                                           labels={"0": "Left", "1": "Right", "all": "Both", "avg": "L/R power avg"}),
        "ir": plots.png_impulse(os.path.join(rundir, "ir.png"), irs, SR, f"Run {n} {label}: direct sound per channel"),
    }
    ev = {"run": n, "label": label, "latency_samples": int(lat), "onsets_ms": onsets, "polarity": pol, "room_modes": modes, "corrections": corr}
    np.save(os.path.join(rundir, "irs.npy"), np.stack([irs[k] for k in sorted(irs)]))
    prev = os.path.join(sess.dir, f"run{n-1:02d}", "irs.npy")
    if n > 1 and os.path.exists(prev):
        pv = np.load(prev)
        ev["change_vs_previous_run"] = analysis.compare_runs({k: pv[i] for i, k in enumerate(sorted(irs))}, irs, SR)
    json.dump(ev, open(os.path.join(rundir, "evidence.json"), "w"), indent=1, default=str)
    p01 = pol["pairs"].get("0-1", {})
    sess.state.setdefault("run_summary", []).append({
        "run": n, "label": label, "sign": p01.get("relative_sign"), "corr": p01.get("correlation"),
        "burst_db": pol["burst_test"]["diff_db"], "lf_sum_db": (pol.get("lf_sum_test") or {}).get("all_minus_power_sum_db_median"),
        "arrival_diff_ms": p01.get("measured_arrival_diff_ms"), "verdict": pol["algorithmic_verdict"].get("0-1")})
    sess.save()
    return ev, pngs, rundir


def summary(ev):
    pol = ev["polarity"]
    p = pol["pairs"]["0-1"]
    out = [f"Run {ev['run']} {ev.get('label','')}: verdict {json.dumps(pol['algorithmic_verdict'])}",
           f"  corr sign full/mid/hf: {p['relative_sign']}/{p['relative_sign_mid_800_2500']}/{p['relative_sign_hf_3k_10k']} "
           f"(corr {p['correlation']}/{p['correlation_mid']}/{p['correlation_hf']})",
           f"  arrival R-L {p['measured_arrival_diff_ms']} ms (expected {p['expected_arrival_diff_ms_from_distances']})",
           f"  burst diff {pol['burst_test']['diff_db']:.1f} dB | LF sum {(pol.get('lf_sum_test') or {}).get('all_minus_power_sum_db_median')} dB",
           f"  LF peaks: {[(q['freq'], q['excess_db']) for q in ev['room_modes']['peaks']]}  dips: {[(d['freq'], d['depth_db']) for d in ev['room_modes']['dips']]}"]
    out += ["  NOTE " + n for n in pol["notes"]]
    if ev.get("change_vs_previous_run"):
        out.append("  change vs previous run (sign -1 = this channel's section flipped): " + json.dumps(ev["change_vs_previous_run"]))
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sink", required=True, help="pw sink name, ssh:user@host:dev, or file:manual")
    ap.add_argument("--source", required=True, help="pw source name, alsa:hw:X,Y, or ssh:user@host:hw:X,Y")
    ap.add_argument("--dist", nargs=2, type=float, help="mic distance to L and R speaker (m)")
    ap.add_argument("--dims", nargs=3, type=float, help="room L W H (m)")
    ap.add_argument("--session", help="session directory (reuse to append runs)")
    ap.add_argument("--label", default="")
    ap.add_argument("--mic-profile", help="json with points [[f,dB],...]")
    ap.add_argument("--llm", action="store_true", help="ask the LLM (claude CLI, fable) to review polarity")
    a = ap.parse_args()
    sink = {"kind": "ssh" if a.sink.startswith("ssh:") else "file" if a.sink.startswith("file:") else "pw", "name": a.sink, "id": None}
    kind = "ssh" if a.source.startswith("ssh:") else "alsa" if a.source.startswith("alsa:") else "pw"
    source = {"kind": kind, "name": a.source, "id": None}
    if kind == "ssh":
        alsa = audio_io.probe_remote_alsa(*audio_io._ssh_parts(a.source))
    elif kind == "alsa":
        alsa = audio_io.probe_alsa_format(source)
    else:
        alsa = None
    sess = Session(a.session)
    mp = json.load(open(a.mic_profile))["points"] if a.mic_profile else None
    ev, pngs, rundir = run_measurement(sink, source, sess, a.dist, a.dims, alsa, mp, a.label)
    print(summary(ev))
    print("run comparison:")
    for r in sess.state["run_summary"]:
        print(f"  run {r['run']:>2} {r.get('label',''):<14} sign {r['sign']:>2} corr {r['corr']}  burst {r['burst_db']:+.1f} dB  LF sum {r['lf_sum_db']:+.1f} dB  arrival {r['arrival_diff_ms']:+.2f} ms  {r['verdict']}")
    if a.llm:
        prompt = prompts.POLARITY.format(channels=2, ch0_name="Left", ch1_name="Right", setup=sess.state.get("setup", "n/a"), bursts=4,
                                         evidence=prompts.j({k: v for k, v in ev["polarity"].items() if k != "algorithmic_verdict"}),
                                         verdict=prompts.j(ev["polarity"]["algorithmic_verdict"]),
                                         image_desc="(1) frequency response of L, R and both; (2) direct-sound impulse responses per channel")
        text, data = LLM("claude-cli", "fable", log_dir=sess.path("llm")).ask(prompts.SYSTEM, prompt, [pngs["fr"], pngs["ir"]], tag=f"polarity_run{ev['run']}")
        print(text)


if __name__ == "__main__":
    main()
