"""Prompt templates.  Kept deliberately stable and explicit so that different people running the
tool get consistent guidance; all measured numbers are passed as JSON, never paraphrased."""
import json

SYSTEM = """You are an experienced acoustics and hi-fi setup engineer assisting a semi-automated tool
called auto_audio_assist.  The tool plays sine sweeps and noise bursts through the user's speakers,
records them with a microphone, and computes evidence algorithmically.  Your job at each stage is to
double-check the algorithmic verdict against the raw numbers (and images when given), interpret
them for the user in plain language, and propose concrete next actions.

Rules:
- Focus on finding SETUP PROBLEMS (reversed polarity on a speaker or driver section, gross room
  modes), not on producing a perfectly flat response.
- Be decisive but honest about confidence.  If the evidence is weak say so and say what would make
  it stronger (e.g. move the mic, raise the volume, re-run).
- Absolute polarity of the whole system is unknowable from one mic; only RELATIVE polarity between
  channels or driver sections matters.  Do not claim absolute polarity.
- Keep advice practical for someone standing in their living room with a screwdriver.
- Be consistent: the same evidence must lead to the same conclusion every time.
- Always finish your answer with exactly one fenced ```json block following the schema you are
  given.  Put the human-readable explanation BEFORE the JSON block."""

MIC_PROFILE = """We need a rough on-axis frequency response of the microphone described below, to
compensate our measurements.  Use whatever you know about this exact device or its closest
relatives (same laptop family / same capsule type).  Precision of +-3 dB is fine; we only look for
gross problems.  Normalise to 0 dB at 1 kHz.  Cover 20 Hz to 20 kHz with about 15-25 points, and
be explicit about which parts are known vs guessed.  If the mic is a calibrated reference mic,
return a flat curve.

Microphone / device: {mic_desc}
Computer: {computer}
Capture path notes: {path_notes}

Schema:
```json
{{"points": [[20, -20.0], [50, -10.0], ...], "confidence": "low|medium|high", "notes": "..."}}
```"""

POLARITY = """STAGE: speaker polarity check ({channels} channels; channel 0 = {ch0_name}, channel 1 = {ch1_name}).

Setup described by the user:
{setup}

How the evidence was produced: one exponential sine sweep per channel then all channels together,
then {bursts} pairs of 40-150 Hz noise bursts (all channels in phase / channel 1+ inverted).  Impulse
responses (IRs) were obtained by deconvolution, direct-sound onsets aligned per channel.
'relative_sign' is the sign of the peak cross-correlation between channel IRs in a band
(+1 same polarity, -1 opposite); 'correlation' is its normalised magnitude (0..1, >0.3 usable).
'burst_test.diff_db' is LF level of in-phase bursts minus anti-phase bursts (positive = in phase,
only meaningful if the mic is roughly equidistant from both speakers).
'crossover_notch' is a dip in the short-window (direct sound) response between 800 Hz and 8 kHz.
'lf_sum_test' compares the all-channels sweep with the power sum of the single sweeps at 25-70 Hz:
about +3 dB means in phase, well below -3 dB means reversed.  If it carries an 'invalid' field the
playback gain was not constant and it has already been discarded - say so rather than using it.
'sweep_level_check' measures band levels straight from the recording during each sweep, bypassing
IR extraction; use it to judge whether every sweep was actually reproduced at the same gain.
Arrival time differences let you sanity-check that the correct direct sound was found.

Evidence (JSON):
{evidence}

Algorithmic verdict: {verdict}

Images attached: {image_desc}

Tasks:
1. Say whether channel pairs are in phase or reversed, and whether any tweeter/mid section looks
   reversed relative to the other speaker (jumper/bi-wire error).  State confidence.
2. If something is wrong, give step-by-step physical instructions (which speaker, which terminals).
3. If evidence is inconclusive, say exactly what to change for a better measurement.

Schema:
```json
{{"pair_verdicts": {{"0-1": "in_phase|reversed|inconclusive"}},
  "tweeter_verdicts": {{"0": "ok|suspect_reversed|inconclusive", "1": "..."}},
  "confidence": "low|medium|high",
  "summary": "one or two sentences",
  "actions": ["...", "..."],
  "remeasure_recommended": true}}
```"""

MODES = """STAGE: room mode / gross frequency response check.

Setup described by the user:
{setup}
Room dimensions (m, L x W x H): {dims}
Mic position notes: {mic_pos}
Mic compensation applied: {mic_comp}

Evidence: smoothed (1/12 octave) magnitude response 25-300 Hz per channel and the power average,
the deviation from a 1-octave running trend ('excess_db'), detected peaks with an estimated Q,
detected deep dips, and axial modes predicted from the room dimensions.

{evidence}

Algorithmically proposed corrections (parametric EQ cuts): {corrections}

Images attached: {image_desc}

Tasks:
1. Identify the few (max 5) frequencies that are genuine problems worth correcting.  Ignore
   wiggles under ~4 dB.  Distinguish room modes (match predicted modes, high Q, both channels)
   from speaker/boundary effects (broad, channel-specific) and from mic limitations.
2. Deep dips cannot be fixed by EQ; say so and suggest physical changes instead.
3. Suggest physical changes worth trying before EQ (speaker distance to walls, listening position,
   toe-in, furniture, bass traps) - concrete and ordered by likely payoff.
4. Confirm or adjust each proposed EQ correction (frequency, gain, Q).  Cuts only, max -10 dB.

Schema:
```json
{{"problems": [{{"freq_hz": 45.0, "kind": "room mode|boundary|speaker|mic", "severity": "high|medium|low", "note": "..."}}],
  "physical_suggestions": ["...", "..."],
  "eq": [{{"freq_hz": 45.0, "gain_db": -6.0, "q": 4.0}}],
  "summary": "one or two sentences",
  "confidence": "low|medium|high"}}
```"""

FINAL = """STAGE: hand-over.  The user has finished physical changes.  Final corrections to apply:
{corrections}

The user's playback system: {setup}
Software options already prepared by the tool: (a) a PipeWire filter-chain 'Room EQ' virtual sink on
the source computer (Linux), (b) an Equalizer APO / EasyEffects import file, (c) a plain table.

Tasks: write short, practical guidance for applying these corrections (1) on the amplifier's own EQ
if it has one (tell them how to approximate with bass/treble or a graphic EQ if no parametric EQ),
(2) with an inexpensive external DSP/EQ device (name the category of device and what to look for),
(3) in software on the source computer (Linux: the prepared PipeWire sink or EasyEffects; also one
line each for macOS and Windows).  Also add a two-line reminder about re-checking after moving
anything.  Keep it under 350 words.

Schema:
```json
{{"amp_eq": "...", "external_device": "...", "software": "...", "notes": "..."}}
```"""

FREE = """The user asks, in the context of the current measurement session:
{question}

Current evidence summary (JSON): {evidence}

Answer concisely.  Schema:
```json
{{"answer": "..."}}
```"""


def j(x):
    return json.dumps(x, indent=1, default=str)
