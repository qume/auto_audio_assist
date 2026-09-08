# AGENTS.md

Guidance for AI coding agents (and humans) working on `auto_audio_assist`.

## What this is

A Python 3 / tkinter tool that measures a stereo system with sweeps and noise bursts, detects
setup faults (speaker or driver-section polarity, channel swap, gross room modes), and prepares
evidence for an LLM to review at each stage. Read `README.md` first, then `aaa/analysis.py`.

## Ground rules

- **Never play sound through the speakers without the user's explicit go-ahead** for that
  specific playback. Silent probes (device opens, zero-filled files) are fine. Say what you are
  about to play and how long it is.
- **Keep the evidence honest.** Every verdict must be traceable to numbers in `evidence.json`.
  If a test is unreliable in some situation (off-centre mic, unknown latency, noisy room), add a
  note to `evidence["polarity"]["notes"]` rather than silently trusting it.
- **Only relative polarity is knowable** from one microphone. Never claim absolute polarity.
- **Repeatability over cleverness.** Test signals use fixed seeds. Analysis is pure
  numpy/scipy. Prompts are templates in `aaa/prompts.py` that pass numbers as JSON and require a
  JSON block back. Do not paraphrase measurements into prose for the model.
- **No new dependencies** beyond the scientific stack (numpy, scipy, matplotlib) and tkinter
  unless the user agrees. LLM backends use `urllib`, deliberately.
- **Frugal with LLM calls.** One call per stage. Cache anything researchable (mic profiles live
  in `mics/`). The user pays for these.
- Stay **Linux-first but cross-platform-shaped**: put OS specifics in `aaa/audio_io.py` and
  `aaa/eqexport.py`, keep `signals`, `analysis`, `prompts`, `llm` platform-neutral.

## Layout and responsibilities

| File | Owns | Do not put here |
|---|---|---|
| `aaa/signals.py` | stimulus generation, marker layout | any I/O |
| `aaa/analysis.py` | deconvolution, IR extraction, polarity tests, room modes, run comparison, corrections | Tk, subprocess, file paths |
| `aaa/audio_io.py` | device lists, play/record (PipeWire, ALSA local + remote over ssh, manual), remote sink install | DSP |
| `aaa/llm.py` | backends, logging, JSON extraction | prompt wording |
| `aaa/prompts.py` | prompt templates and the system prompt | code that runs |
| `aaa/plots.py` | Tk canvas plotting, matplotlib PNGs for the LLM | analysis decisions |
| `aaa/eqexport.py` | APO text, PipeWire filter-chain config, install/uninstall | measurement |
| `aaa/session.py` | session dirs, config, mic cache paths | logic |
| `aaa/app.py` | the wizard; worker threads; UI text | DSP or prompt content |
| `aaa/cli.py` | headless runner mirroring the app's measure step | UI |

## Threading in the GUI

Worker threads run audio and LLM calls via `App.run_bg(fn, done)`. **Never touch Tk from a
worker.** Post status text with `App.set_status()` (it goes through the queue); the `done`
callback runs on the main thread. Tk raises "main thread is not in main loop" otherwise.

## Measurement conventions

- Sample rate 48 kHz, float32 everywhere, 48 points per octave log grid from `analysis.log_grid`.
- Channel `"0"` is the reference (left). Pair keys are `"0-1"`, `"0-2"`, ... . `"all"` is the
  all-channels-together sweep.
- Latency is unknown by design: `extract_irs` searches the whole recording for the first sweep and
  aligns each channel within ±30 ms of its expected position, using `find_direct` (first envelope
  crossing on a 200 Hz–8 kHz band-limited copy, then the strongest sample within 4 ms). Do not
  replace this with a plain `argmax`; room-mode ringing and reflections can exceed the direct sound.
- Levels are uncalibrated. Displays are referenced to channel 0's 200–2000 Hz mean.
- Multi-capsule raw mics: use capsule 0 only (see `audio_io.play_and_record`).
- The Asahi Linux mic filter chain adds a 120 Hz high-pass; it is compensated analytically in
  `App.set_mic_profile` when that source is selected. Prefer the raw `alsa:hw:...HPAI` device,
  which returns float32 regardless of the requested format (`probe_alsa_format` detects this).

## EQ sink and remote install

`eqexport.as_pipewire_conf` builds a filter-chain with one path per channel (`invert` → `delay`
→ `bq_peaking`...). `target.object` in `playback.props` must name an existing node, otherwise
PipeWire silently links to the default sink — check with `pw-link -l | grep aaa_room_eq_output`.
Validate a chain without restarting anything: `pw-cli -m load-module libpipewire-module-filter-chain
"<the args block>"` loads it for as long as pw-cli runs. The remote installer (`install_pipewire_remote`)
writes the conf over ssh, restarts the remote user services and sets the default sink by node name.

## Adding a test or verdict

1. Compute it in `analysis.py` as plain data; add it to the evidence dict with its inputs.
2. Add its influence to `algorithmic_verdict` with a clear threshold and an `inconclusive` path.
3. Explain it in `prompts.POLARITY` or `prompts.MODES` so the LLM knows how to read it.
4. Surface it in `app._refresh_4` / `_refresh_5` and `cli.summary`.
5. Extend `tests/sim_test.py` with a synthetic case that should trigger it.

## Testing

```bash
python3 tests/sim_test.py     # must print in_phase / reversed / tweeter-differs for the 3 cases
python3 tests/gui_smoke.py    # needs a display; drives all 7 steps with simulated audio
```

There is no audio hardware in CI. Anything touching real devices should degrade to a clear error,
not a hang: recorders use SIGINT with timeouts, ssh uses `BatchMode=yes` and `ConnectTimeout`.

## Polarity: what each method can and cannot decide

- **Whole-speaker** polarity: decide from bass (`lf_sum_test`, `burst_test`, broadband
  cross-correlation). Reliable from a listening position.
- **Driver sections** (reversed tweeter / HF section): NOT decidable from a listening position.
  Reflections arrive within half a cycle of the direct sound above ~1 kHz, and a per-band sign is
  itself ambiguous by half a cycle. Proof from real data: an untouched speaker reported a flipped
  2-5 kHz band at correlation 0.95 between two runs. Use `interchannel_phase` (a real reversal
  holds ~180 deg across octaves; require `consistency >= 0.6` in at least two ranges) and otherwise
  return "inconclusive" and point at the near-field check. Do not lower those gates to make a
  synthetic test pass.
- **Near-field** (`cli --nearfield`, `nearfield_polarity`): the answer for driver sections. Judge
  by the sign of the FIRST excursion after onset, never the largest peak, and only read a filtered
  view where that driver dominates (check `level_db`).
- Filtering for polarity must be **causal** (`lowpass_causal` / `highpass_causal`). `sosfiltfilt`
  is zero-phase and puts ringing before the impulse, making a first-excursion sign meaningless.

## Windowing

`extract_irs` keeps ~5 ms of pre-roll ahead of the impulse, so `freq_response` finds the onset and
windows from there. A gate applied from sample 0 measures pre-arrival noise (this was a real bug -
it produced a +20 dB treble rise). A short gate (5-10 ms) is quasi-anechoic, i.e. the speaker with
the room excluded, and is only valid above roughly 1/window; a 0.5 s gate is the in-room response.

## Self-checks that must stay

- `analysis.lf_sum_test` marks itself `invalid` above +3.5 dB. Coherent summation of two sources
  cannot exceed +3 dB over their power sum, so a higher figure proves the playback gain changed
  between sweeps. It is then excluded from the verdict vote — do not "fix" this by raising the
  threshold.
- `analysis.sweep_level_check` measures band levels from the raw recording during each sweep,
  bypassing IR extraction entirely. It exists because an AV receiver that unmutes ~1 s into a
  stream removes the low end of the first sweep only, which looks exactly like a dead woofer.
  The warm-up burst in `signals.build_measurement` is the countermeasure; keep both.

## Things that look like bugs but aren't

- Deconvolved responses at 170 dB: uncalibrated inverse filter; only differences matter.
- Burst test disagreeing with the correlation tests: expected when the mic is off-centre by more
  than ~0.4 m path difference; it is excluded from the vote in that case.
- `channel_swap_suspected`: a hint based on the user's distance estimates. It must be confirmed
  by ear with the "Identify channels" test before anyone touches cables.
