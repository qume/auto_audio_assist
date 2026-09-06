"""Tk wizard: Setup -> Mic profile -> Level check -> Measure -> Polarity -> Room modes -> Export."""
import json
import os
import queue
import subprocess
import threading
import traceback
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np

from . import audio_io, signals, analysis, plots, prompts, eqexport
from .llm import LLM
from .session import Session, load_config, save_config, mic_profile_path

STEPS = ["1. Setup", "2. Mic profile", "3. Level check", "4. Measure", "5. Polarity", "6. Room modes", "7. Export"]
SR = 48000


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("auto_audio_assist - speaker setup debugger")
        self.geometry("1180x820")
        self.cfg = {"channels": 2, "llm_backend": "claude-cli", "llm_model": "fable", "llm_url": "", "llm_key": "",
                    "use_llm": True, "room_dims": "", "dist_l": "", "dist_r": "", "setup": "", "mic_desc": "",
                    "mic_pos": "", "sink": "", "source": "", "remote_host": "lunar@media", "remote_dev": "default"}
        self.cfg.update(load_config())
        self.session = Session()
        self.q = queue.Queue()
        self.busy = False
        self.mic_profile = None          # list of [f, dB]
        self.last = None                 # dict with irs, evidence, ... of the last measurement
        self.corrections = []
        self.llm_json = {}
        self.sinks, self.sources = [], []
        self._build()
        self.after(100, self._poll)
        self.log(f"Session folder: {self.session.dir}")

    # ------------------------------------------------------------------ layout
    def _build(self):
        pw = ttk.Panedwindow(self, orient="horizontal")
        pw.pack(fill="both", expand=True)
        left = ttk.Frame(pw, width=150)
        self.steps = tk.Listbox(left, exportselection=False, font=("TkDefaultFont", 10))
        for s in STEPS:
            self.steps.insert("end", s)
        self.steps.pack(fill="both", expand=True, padx=4, pady=4)
        self.steps.bind("<<ListboxSelect>>", lambda e: self.show_step(self.steps.curselection()[0] if self.steps.curselection() else 0))
        pw.add(left, weight=0)
        right = ttk.Panedwindow(pw, orient="vertical")
        pw.add(right, weight=1)
        self.content = ttk.Frame(right)
        right.add(self.content, weight=3)
        bottom = ttk.Frame(right)
        right.add(bottom, weight=1)
        ttk.Label(bottom, text="Assistant / log").pack(anchor="w")
        self.out = scrolledtext.ScrolledText(bottom, height=10, wrap="word", font=("TkFixedFont", 9))
        self.out.pack(fill="both", expand=True)
        row = ttk.Frame(bottom)
        row.pack(fill="x")
        self.question = ttk.Entry(row)
        self.question.pack(side="left", fill="x", expand=True, padx=2)
        self.question.bind("<Return>", lambda e: self.ask_free())
        ttk.Button(row, text="Ask the assistant", command=self.ask_free).pack(side="left")
        self.status = ttk.Label(row, text="idle", width=28, anchor="e")
        self.status.pack(side="right")
        self.frames = [self._setup_frame(), self._mic_frame(), self._level_frame(), self._measure_frame(),
                       self._polarity_frame(), self._modes_frame(), self._export_frame()]
        self.show_step(0)

    def show_step(self, i):
        for f in self.frames:
            f.pack_forget()
        self.frames[i].pack(fill="both", expand=True)
        self.steps.selection_clear(0, "end")
        self.steps.selection_set(i)
        refresh = getattr(self, f"_refresh_{i}", None)
        if refresh:
            refresh()

    def log(self, msg, tag=None):
        self.out.insert("end", msg.rstrip() + "\n\n")
        self.out.see("end")
        self.session.log(msg[:2000])

    # ------------------------------------------------------------------ threading helpers
    def run_bg(self, fn, done=None, label="working..."):
        if self.busy:
            messagebox.showinfo("Busy", "Please wait for the current task to finish.")
            return
        self.busy = True
        self.status.config(text=label)

        def worker():
            try:
                res = fn()
                self.q.put(("ok", done, res))
            except Exception as e:  # noqa
                self.q.put(("err", None, traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()

    def _poll(self):
        try:
            while True:
                kind, done, res = self.q.get_nowait()
                if kind == "status":
                    self.status.config(text=res)
                    continue
                self.busy = False
                self.status.config(text="idle")
                if kind == "err":
                    self.log("ERROR:\n" + res)
                    messagebox.showerror("Error", res.strip().splitlines()[-1])
                elif done:
                    done(res)
        except queue.Empty:
            pass
        self.after(150, self._poll)

    def set_status(self, text):
        """Thread-safe: worker threads must never touch Tk directly."""
        self.q.put(("status", None, text))

    # ------------------------------------------------------------------ helpers
    def llm(self):
        return LLM(self.cfg["llm_backend"], self.cfg["llm_model"] or None, self.cfg["llm_url"] or None,
                   self.cfg["llm_key"] or None, log_dir=self.session.path("llm"))

    def sink(self):
        for s in self.sinks:
            if s["name"] == self.cfg["sink"]:
                return s
        return self.sinks[0] if self.sinks else None

    def source(self):
        for s in self.sources:
            if s["name"] == self.cfg["source"]:
                return s
        return self.sources[0] if self.sources else None

    def dims(self):
        try:
            d = [float(x) for x in self.cfg["room_dims"].replace("x", " ").replace(",", " ").split()]
            return d if len(d) == 3 else None
        except ValueError:
            return None

    def distances(self):
        try:
            return [float(self.cfg["dist_l"]), float(self.cfg["dist_r"])]
        except ValueError:
            return None

    def setup_text(self):
        return (f"{self.cfg['setup']}\nOutput device: {self.cfg['sink']}. Mic: {self.cfg['mic_desc']} ({self.cfg['source']}). "
                f"Mic distance to left/right speaker (m): {self.cfg['dist_l'] or '?'} / {self.cfg['dist_r'] or '?'}. "
                f"Mic position: {self.cfg['mic_pos']}")

    def alsa_opts(self):
        return getattr(self, "_alsa_ch", None), getattr(self, "_alsa_fmt", None)

    # ------------------------------------------------------------------ step 1: setup
    def _setup_frame(self):
        f = ttk.Frame(self.content, padding=8)
        self.vars = {}
        r = 0
        ttk.Label(f, text="Output that feeds the amplifier:").grid(row=r, column=0, sticky="w")
        self.sink_cb = ttk.Combobox(f, width=60, state="readonly")
        self.sink_cb.grid(row=r, column=1, sticky="we")
        ttk.Button(f, text="Set as system default", command=self.make_default).grid(row=r, column=2, padx=4)
        r += 1
        ttk.Label(f, text="Remote box (ssh user@host) with ALSA device:").grid(row=r, column=0, sticky="w")
        rrow = ttk.Frame(f)
        rrow.grid(row=r, column=1, sticky="we")
        self.vars["remote_host"] = tk.StringVar(value=self.cfg.get("remote_host", ""))
        self.vars["remote_dev"] = tk.StringVar(value=self.cfg.get("remote_dev", "default"))
        ttk.Entry(rrow, textvariable=self.vars["remote_host"], width=28).pack(side="left")
        ttk.Entry(rrow, textvariable=self.vars["remote_dev"], width=12).pack(side="left", padx=4)
        ttk.Button(rrow, text="Install system-wide sink for it", command=self.install_remote).pack(side="left", padx=4)
        ttk.Button(rrow, text="Remove", command=self.remove_remote).pack(side="left")
        r += 1
        ttk.Label(f, text="Microphone (source):").grid(row=r, column=0, sticky="w")
        self.src_cb = ttk.Combobox(f, width=60, state="readonly")
        self.src_cb.grid(row=r, column=1, sticky="we")
        ttk.Button(f, text="Refresh devices", command=self.refresh_devices).grid(row=r, column=2, padx=4)
        r += 1
        for key, label in [("mic_desc", "Microphone description (model):"), ("mic_pos", "Mic position notes:"),
                           ("setup", "Speakers / amp / source description:"), ("room_dims", "Room L x W x H (m):"),
                           ("dist_l", "Mic distance to LEFT speaker (m):"), ("dist_r", "Mic distance to RIGHT speaker (m):")]:
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w")
            v = tk.StringVar(value=self.cfg.get(key, ""))
            self.vars[key] = v
            ttk.Entry(f, textvariable=v, width=70).grid(row=r, column=1, sticky="we")
            r += 1
        ttk.Separator(f).grid(row=r, column=0, columnspan=3, sticky="we", pady=6)
        r += 1
        self.vars["use_llm"] = tk.BooleanVar(value=bool(self.cfg.get("use_llm", True)))
        ttk.Checkbutton(f, text="Use an LLM to review each stage", variable=self.vars["use_llm"]).grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        ttk.Label(f, text="LLM backend:").grid(row=r, column=0, sticky="w")
        self.vars["llm_backend"] = tk.StringVar(value=self.cfg["llm_backend"])
        cb = ttk.Combobox(f, textvariable=self.vars["llm_backend"], values=["claude-cli", "anthropic", "openai"], state="readonly", width=20)
        cb.grid(row=r, column=1, sticky="w")
        cb.bind("<<ComboboxSelected>>", self._backend_changed)
        r += 1
        for key, label in [("llm_model", "Model (claude-cli: fable/opus/sonnet; anthropic: claude-opus-5; openai-compatible: any):"),
                           ("llm_url", "API base URL (not needed for claude-cli):"), ("llm_key", "API key (not needed for claude-cli):")]:
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w")
            v = tk.StringVar(value=self.cfg.get(key, ""))
            self.vars[key] = v
            ttk.Entry(f, textvariable=v, width=70, show="*" if key == "llm_key" else "").grid(row=r, column=1, sticky="we")
            r += 1
        ttk.Button(f, text="Save and continue ->", command=self.save_setup).grid(row=r, column=1, sticky="e", pady=8)
        f.columnconfigure(1, weight=1)
        return f

    def _backend_changed(self, *_):
        from .llm import DEFAULTS
        d = DEFAULTS[self.vars["llm_backend"].get()]
        self.vars["llm_model"].set(d["model"])
        self.vars["llm_url"].set(d["url"])

    def _refresh_0(self):
        self.refresh_devices()

    def refresh_devices(self):
        self.sinks = audio_io.output_options(self.vars["remote_host"].get().strip(), self.vars["remote_dev"].get().strip())
        self.sources = audio_io.list_sources()
        self.sink_cb["values"] = [f"{s['desc']}  [{s['name']}]" for s in self.sinks]
        self.src_cb["values"] = [f"{s['desc']}  [{s['name']}]" for s in self.sources]
        cur = audio_io.default_sink_name()
        for i, s in enumerate(self.sinks):
            if s["name"] == (self.cfg["sink"] or cur):
                self.sink_cb.current(i)
        if not self.sink_cb.get() and self.sinks:
            self.sink_cb.current(0)
        for i, s in enumerate(self.sources):
            if s["name"] == self.cfg["source"]:
                self.src_cb.current(i)
        if not self.src_cb.get() and self.sources:
            # prefer a raw ALSA internal mic on Asahi, else the first source
            idx = next((i for i, s in enumerate(self.sources) if "HPAI" in s["name"]), 0)
            self.src_cb.current(idx)
        if not self.vars["mic_desc"].get():
            self.vars["mic_desc"].set("built-in microphone of " + audio_io.laptop_model())

    def make_default(self):
        i = self.sink_cb.current()
        if i >= 0 and self.sinks[i]["kind"] == "pw":
            audio_io.set_default_sink(self.sinks[i]["id"])
            self.log(f"System default output set to {self.sinks[i]['desc']}")
        else:
            messagebox.showinfo("Not a PipeWire sink", "Only PipeWire sinks can be the system default. For a remote box use "
                                "'Install system-wide sink for it' first, then pick the new sink.")

    def install_remote(self):
        host, dev = self.vars["remote_host"].get().strip(), self.vars["remote_dev"].get().strip() or "default"
        if not host:
            return
        if not messagebox.askyesno("Install", f"Create a PipeWire sink 'Amp via {host}' and a user service that streams it to\n"
                                              f"ssh {host} aplay -D {dev}?  (Runs at login; audio is silent until you route something to it.)"):
            return
        p = audio_io.install_remote_sink(host, dev)
        self.log(f"Installed {p}; service {audio_io.remote_sink_status()}. Sink name: {audio_io.REMOTE_SINK}")
        self.refresh_devices()

    def remove_remote(self):
        audio_io.uninstall_remote_sink()
        self.log("Removed remote sink and service.")
        self.refresh_devices()

    def save_setup(self):
        for k, v in self.vars.items():
            self.cfg[k] = v.get()
        if self.sink_cb.current() >= 0:
            self.cfg["sink"] = self.sinks[self.sink_cb.current()]["name"]
        if self.src_cb.current() >= 0:
            self.cfg["source"] = self.sources[self.src_cb.current()]["name"]
        save_config(self.cfg)
        self.session.state["config"] = {k: v for k, v in self.cfg.items() if k != "llm_key"}
        self.session.save()
        src = self.source()
        if src and src["kind"] == "alsa":
            def probe():
                return audio_io.probe_alsa_format(src)

            def done(res):
                self._alsa_ch, self._alsa_fmt = res
                self.log(f"Raw ALSA mic: {res[0]} channel(s), sample format {res[1]} (auto-detected). Capsule 0 is used.")
                self.show_step(1)
            self.run_bg(probe, done, "probing mic format...")
        else:
            self.show_step(1)

    # ------------------------------------------------------------------ step 2: mic profile
    def _mic_frame(self):
        f = ttk.Frame(self.content, padding=8)
        ttk.Label(f, text="Rough microphone frequency response used to compensate the measurements.  "
                          "Ask the LLM to research it (cached per mic), or use flat for a calibrated mic.",
                  wraplength=900).pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=4)
        ttk.Button(row, text="Get mic profile from LLM", command=self.get_mic_profile).pack(side="left")
        ttk.Button(row, text="Use flat (reference mic)", command=lambda: self.set_mic_profile([[20, 0], [20000, 0]], "flat")).pack(side="left", padx=4)
        ttk.Button(row, text="Load from file...", command=self.load_mic_profile).pack(side="left")
        ttk.Button(row, text="Continue ->", command=lambda: self.show_step(2)).pack(side="right")
        self.mic_canvas = tk.Canvas(f, bg="white", height=320)
        self.mic_canvas.pack(fill="both", expand=True)
        self.mic_plot = plots.CanvasPlot(self.mic_canvas, "Microphone response (dB re 1 kHz)")
        self.mic_note = ttk.Label(f, text="", wraplength=900)
        self.mic_note.pack(anchor="w")
        return f

    def _refresh_1(self):
        p = mic_profile_path(self.cfg["mic_desc"] + self.cfg["source"])
        if self.mic_profile is None and os.path.exists(p):
            d = json.load(open(p))
            self.set_mic_profile(d["points"], f"cached ({d.get('confidence','?')}): {d.get('notes','')}")

    def path_notes(self):
        src = self.cfg["source"]
        if "j415-mic" in src or "effect_output" in src:
            return ("Asahi Linux processed mic: 3-capsule beamformer, +36 dB gain and a 2nd-order 120 Hz high-pass "
                    "(the tool compensates the high-pass analytically; do NOT include it in your curve).")
        if src.startswith("alsa:"):
            return "Raw ALSA capture of the internal mic array, no processing, capsules averaged."
        return "Standard PipeWire source."

    def set_mic_profile(self, points, note=""):
        pts = sorted([[float(a), float(b)] for a, b in points])
        if "j415-mic" in self.cfg["source"] or "effect_output" in self.cfg["source"]:
            fc = 120.0
            pts = [[f, d + 10 * np.log10(f ** 4 / (f ** 4 + fc ** 4))] for f, d in pts]
            note += "  (+ analytic 120 Hz high-pass of the Asahi mic chain)"
        self.mic_profile = pts
        self.mic_plot.clear()
        self.mic_plot.add([p[0] for p in pts], [p[1] for p in pts], "mic")
        self.mic_plot.draw()
        self.mic_note.config(text=note)
        self.session.state["mic_profile"] = {"points": pts, "note": note}
        self.session.save()

    def get_mic_profile(self):
        prompt = prompts.MIC_PROFILE.format(mic_desc=self.cfg["mic_desc"], computer=audio_io.laptop_model(), path_notes=self.path_notes())

        def work():
            return self.llm().ask(prompts.SYSTEM, prompt, tag="mic_profile")

        def done(res):
            text, data = res
            self.log("LLM mic profile:\n" + text)
            if data and "points" in data:
                json.dump(data, open(mic_profile_path(self.cfg["mic_desc"] + self.cfg["source"]), "w"), indent=1)
                self.set_mic_profile(data["points"], f"LLM ({data.get('confidence','?')}): {data.get('notes','')}")
            else:
                messagebox.showwarning("No profile", "The LLM answer had no usable JSON; using flat.")
                self.set_mic_profile([[20, 0], [20000, 0]], "flat (fallback)")
        self.run_bg(work, done, "asking LLM about the mic...")

    def load_mic_profile(self):
        p = filedialog.askopenfilename(filetypes=[("JSON", "*.json"), ("All", "*")])
        if p:
            d = json.load(open(p))
            self.set_mic_profile(d["points"], f"file: {os.path.basename(p)}")

    def mic_comp(self, grid):
        if not self.mic_profile:
            return np.zeros(len(grid))
        pts = np.array(self.mic_profile)
        return np.interp(np.log10(grid), np.log10(pts[:, 0]), pts[:, 1])

    # ------------------------------------------------------------------ step 3: level check
    def _level_frame(self):
        f = ttk.Frame(self.content, padding=8)
        ttk.Label(f, text="Turn the amplifier on, set a loud-ish listening volume, keep the room quiet.  "
                          "Target: pink noise at least 25 dB above the noise floor in the 125 Hz - 4 kHz bands, no clipping.  "
                          "Absolute level depends on the mic (raw laptop mics are very quiet, that is fine).", wraplength=900).pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=4)
        ttk.Button(row, text="Measure noise floor (3 s)", command=self.noise_floor).pack(side="left")
        ttk.Button(row, text="Play pink noise + record (6 s)", command=self.level_test).pack(side="left", padx=4)
        ttk.Button(row, text="Export pink noise WAV...", command=lambda: self.export_wav(np.stack([signals.pink_noise(20.0, SR)] * 2, 1), "pink_noise_20s.wav")).pack(side="left")
        ttk.Button(row, text="Identify channels (LEFT only, 4 s)", command=self.identify_channels).pack(side="left", padx=4)
        ttk.Button(row, text="Continue ->", command=lambda: self.show_step(3)).pack(side="right")
        self.level_text = tk.Text(f, height=8, font=("TkFixedFont", 10))
        self.level_text.pack(fill="x")
        self.noise_db = None
        self.noise_seg = None
        return f

    def noise_floor(self):
        src = self.source()

        def work():
            x = audio_io.record_only(src, 3.0, self.session.path("level"), SR, *self.alsa_opts())
            return float(analysis.db(np.sqrt(np.mean(x ** 2)))), x

        def done(res):
            v, x = res
            self.noise_seg = x
            self.noise_db = v
            self.level_text.insert("end", f"Noise floor: {v:.1f} dBFS\n")
        self.run_bg(work, done, "recording noise floor...")

    def identify_channels(self):
        snk = self.sink()
        if not snk:
            return

        def work():
            sig = signals.pink_noise(4.0, SR)
            p = self.session.path("level", "left_only.wav")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            audio_io.write_wav(p, np.stack([sig, np.zeros_like(sig)], 1), SR)
            if snk["kind"] == "file":
                return "manual"
            audio_io.play(p, snk, blocking=True)
            return "played"

        def done(res):
            if res == "manual":
                self.log(f"Manual mode: play {self.session.path('level', 'left_only.wav')} yourself.")
                return
            ans = messagebox.askquestion("Identify channels", "Did the noise come from the LEFT speaker?\n(Yes = channels correct, No = left/right swapped)")
            self.session.state["channel_id_ok"] = (ans == "yes")
            self.log("Channel identification: " + ("LEFT plays left: channels correct." if ans == "yes" else
                     "LEFT played on the RIGHT speaker: swap the two speaker cables at the amp's L/R outputs (keep each cable's +/- as is)."))
        self.run_bg(work, done, "playing left-only noise...")

    def level_test(self):
        src, snk = self.source(), self.sink()

        if snk["kind"] == "file" and not messagebox.askokcancel("Manual playback", "Start playing the exported pink noise file on your device, then press OK. Recording 6 s."):
            return

        def work():
            sig = signals.pink_noise(6.0, SR)
            st = np.stack([sig, sig], 1)
            mono, _ = audio_io.play_and_record(st, snk, src, self.session.path("level"), SR, 0.5, *self.alsa_opts(), manual_wait_s=0.0)
            # locate the plateau (unknown latency): the 4 s window with the highest energy
            env = np.convolve(mono ** 2, np.ones(4 * SR) / (4 * SR), "valid")
            i0 = int(np.argmax(env))
            seg = mono[i0: i0 + 4 * SR]
            bands = {}
            for lo, hi in [(60, 125), (125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]:
                b = float(analysis.db(np.sqrt(np.mean(analysis.bandpass(seg, lo, hi, SR) ** 2))))
                n = float(analysis.db(np.sqrt(np.mean(analysis.bandpass(self.noise_seg, lo, hi, SR) ** 2)))) if self.noise_seg is not None else float("nan")
                bands[f"{lo}-{hi}"] = (b, b - n)
            return float(analysis.db(np.sqrt(np.mean(seg ** 2)))), float(analysis.db(np.abs(mono).max())), bands

        def done(res):
            rms, peak, bands = res
            snr = rms - self.noise_db if self.noise_db is not None else float("nan")
            mid = [v[1] for k, v in bands.items() if k in ("125-250", "250-500", "500-1000", "1000-2000", "2000-4000")]
            mid_snr = float(np.nanmin(mid)) if mid else float("nan")
            if peak >= -1:
                verdict, hint = "ADJUST", "-> clipping: lower amp volume"
            elif not np.isnan(mid_snr) and mid_snr < 25:
                verdict, hint = "ADJUST", f"-> weakest mid band only {mid_snr:.0f} dB above noise: raise amp volume by ~{max(5, 25 - mid_snr):.0f} dB or quieten the room"
            elif np.isnan(mid_snr):
                verdict, hint = "?", "-> measure the noise floor first for an SNR figure"
            else:
                verdict, hint = "OK", ""
            self.level_text.insert("end", f"Pink noise: RMS {rms:.1f} dBFS, peak {peak:.1f} dBFS, broadband SNR {snr:.1f} dB  [{verdict}] {hint}\n"
                                          "  band SNR: " + "  ".join(f"{k}:{v[1]:.0f}" for k, v in bands.items()) + "\n")
            self.session.state["level"] = {"rms": rms, "peak": peak, "snr": snr, "noise": self.noise_db, "bands": bands}
            self.session.save()
        self.run_bg(work, done, "playing pink noise...")

    # ------------------------------------------------------------------ step 4: measure
    def _measure_frame(self):
        f = ttk.Frame(self.content, padding=8)
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Button(row, text="Run measurement (~40 s: sweeps L, R, both + LF bursts)", command=self.measure).pack(side="left")
        ttk.Button(row, text="Export test signal WAV...", command=lambda: self.export_wav(signals.build_measurement(self.cfg["channels"], SR)[0], "aaa_test_signal.wav")).pack(side="left", padx=4)
        self.measure_info = ttk.Label(row, text="")
        self.measure_info.pack(side="left", padx=10)
        ttk.Button(row, text="Continue ->", command=lambda: self.show_step(4)).pack(side="right")
        self.fr_canvas = tk.Canvas(f, bg="white", height=300)
        self.fr_canvas.pack(fill="both", expand=True, pady=4)
        self.fr_plot = plots.CanvasPlot(self.fr_canvas, "Frequency response, 1/6 oct, mic-compensated (dB)")
        self.ir_canvas = tk.Canvas(f, bg="white", height=220)
        self.ir_canvas.pack(fill="both", expand=True)
        self.ir_plot = plots.CanvasPlot(self.ir_canvas, "Direct sound, 80-1200 Hz band, normalised (polarity view)", xlog=False, xlabel="ms", ylabel="")
        return f

    def measure(self):
        src, snk = self.source(), self.sink()
        if not src or not snk:
            messagebox.showerror("Devices", "Select an output and a microphone in Setup first.")
            return
        rundir, n = self.session.new_run()
        dims, dist = self.dims(), self.distances()
        manual = snk["kind"] == "file"
        if manual and not messagebox.askokcancel("Manual playback", "Export the test signal first (button next to this one) and cue it on your device.\n"
                                                 "Press OK, then start playback within 15 s. Recording runs for the file length + 15 s."):
            return

        def work():
            sig, markers, inv = signals.build_measurement(self.cfg["channels"], SR)
            mono, _ = audio_io.play_and_record(sig, snk, src, rundir, SR, 1.5, *self.alsa_opts(),
                                               progress=lambda p: self.set_status(f"measuring {p*100:.0f}%"), manual_wait_s=15.0)
            self.set_status("analysing...")
            irs, lat, onsets, full = analysis.extract_irs(mono, markers, inv, SR)
            pol = analysis.polarity_evidence(irs, onsets, mono, markers, lat, SR, dist)
            modes = analysis.room_mode_evidence(irs, SR, dims)
            grid = analysis.log_grid()
            comp = self.mic_comp(grid)
            fr = {}
            for k, ir in irs.items():
                _, r = analysis.freq_response(ir, SR, 0.5, grid, 1 / 6)
                fr[k] = r - comp
            # levels are uncalibrated: reference everything to channel 0's 200-2000 Hz mean
            ref = float(np.mean(fr["0"][(grid >= 200) & (grid <= 2000)]))
            for k in fr:
                fr[k] = fr[k] - ref
            # mic compensation (and the same reference) for the LF evidence too
            g = np.array(modes["grid"])
            c2 = self.mic_comp(g)
            for k in modes["responses"]:
                modes["responses"][k] = (np.array(modes["responses"][k]) - c2 - ref).round(2).tolist()
            corr = analysis.corrections_from_modes(modes)
            pngs = {
                "fr": plots.png_frequency_response(os.path.join(rundir, "fr.png"), grid, fr, f"Run {n}: frequency response (1/6 oct, mic compensated)",
                                                   labels={"0": "Left", "1": "Right", "all": "Both"}),
                "lf": plots.png_frequency_response(os.path.join(rundir, "lf.png"), g, {k: v for k, v in modes["responses"].items()},
                                                   f"Run {n}: 25-300 Hz, 1/12 oct", marks=[(p["freq"], f"{p['freq']} Hz +{p['excess_db']} dB") for p in modes["peaks"]],
                                                   xlim=(25, 300), labels={"0": "Left", "1": "Right", "all": "Both", "avg": "L/R power avg"}),
                "ir": plots.png_impulse(os.path.join(rundir, "ir.png"), irs, SR, f"Run {n}: direct sound per channel"),
            }
            ev = {"run": n, "latency_samples": int(lat), "onsets_ms": onsets, "polarity": pol, "room_modes": modes, "corrections": corr}
            p01 = pol["pairs"].get("0-1", {})
            self.session.state.setdefault("run_summary", []).append({
                "run": n, "sign": p01.get("relative_sign"), "burst_db": pol["burst_test"]["diff_db"],
                "lf_sum_db": (pol.get("lf_sum_test") or {}).get("all_minus_power_sum_db_median", float("nan")),
                "verdict": pol["algorithmic_verdict"].get("0-1")})
            json.dump(ev, open(os.path.join(rundir, "evidence.json"), "w"), indent=1, default=str)
            np.save(os.path.join(rundir, "irs.npy"), np.stack([irs[k] for k in sorted(irs)]))
            return {"irs": irs, "fr": fr, "grid": grid, "ev": ev, "pngs": pngs, "rundir": rundir, "n": n}

        def done(res):
            self.last = res
            self.corrections = res["ev"]["corrections"]
            self.llm_json = {}
            fr = res["fr"]
            self.fr_plot.clear()
            for k, lab in (("0", "Left"), ("1", "Right"), ("all", "Both")):
                if k in fr:
                    self.fr_plot.add(res["grid"], fr[k], lab)
            self.fr_plot.draw()
            self.ir_plot.clear()
            n = int(0.012 * SR)
            for k in ("0", "1"):
                x = analysis.bandpass(res["irs"][k], 80, 1200, SR)[:n]
                self.ir_plot.add(np.arange(n) / SR * 1000, x / (np.abs(x).max() + 1e-12), "Left" if k == "0" else "Right")
            self.ir_plot.draw()
            v = res["ev"]["polarity"]["algorithmic_verdict"]
            self.measure_info.config(text=f"Run {res['n']} done. Polarity L-R: {v.get('0-1')}. LF peaks: " +
                                     ", ".join(f"{p['freq']} Hz" for p in res["ev"]["room_modes"]["peaks"]))
            self.log(f"Run {res['n']} complete -> {res['rundir']}\nAlgorithmic verdict: {json.dumps(v)}")
            self.show_step(4)
        self.run_bg(work, done, "measuring...")

    # ------------------------------------------------------------------ step 5: polarity
    def _polarity_frame(self):
        f = ttk.Frame(self.content, padding=8)
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Button(row, text="Ask LLM to review polarity", command=self.review_polarity).pack(side="left")
        ttk.Button(row, text="I changed wiring -> re-measure", command=lambda: self.show_step(3)).pack(side="left", padx=4)
        ttk.Button(row, text="Polarity OK, continue ->", command=lambda: self.show_step(5)).pack(side="right")
        self.pol_text = scrolledtext.ScrolledText(f, wrap="word", font=("TkFixedFont", 9))
        self.pol_text.pack(fill="both", expand=True, pady=4)
        return f

    def _refresh_4(self):
        self.pol_text.delete("1.0", "end")
        if not self.last:
            self.pol_text.insert("end", "No measurement yet.")
            return
        pol = self.last["ev"]["polarity"]
        v = pol["algorithmic_verdict"]
        s = ["ALGORITHMIC VERDICT", f"  Left vs Right: {v.get('0-1')}", f"  Tweeter sections: {json.dumps(v.get('tweeter'))}", ""]
        p = pol["pairs"].get("0-1", {})
        s += ["EVIDENCE",
              f"  aligned cross-correlation sign 80-1200 Hz: {p.get('relative_sign')} (corr {p.get('correlation')})",
              f"  mid 800-2500 Hz: {p.get('relative_sign_mid_800_2500')} (corr {p.get('correlation_mid')})   HF 3-10 kHz: {p.get('relative_sign_hf_3k_10k')} (corr {p.get('correlation_hf')})",
              f"  arrival difference R-L: {p.get('measured_arrival_diff_ms')} ms measured vs {p.get('expected_arrival_diff_ms_from_distances')} ms from your distances",
              f"  LF burst test in-phase minus anti-phase: {pol['burst_test']['diff_db']:.1f} dB (positive = in phase; needs a centred mic)",
              f"  LF sum test 25-70 Hz, both-sweep minus power sum of singles: {(pol.get('lf_sum_test') or {}).get('all_minus_power_sum_db_median', float('nan')):.1f} dB (in phase ~ +3, reversed << -3)"]
        runs = self.session.state.get("run_summary", [])
        if len(runs) > 1:
            s += ["", "RUN COMPARISON (swap one speaker's + and - between runs; the wiring with the HIGHER burst and LF-sum values is correct)",
                  "  run  corr-sign  burst diff dB  LF sum dB  verdict"]
            for r in runs:
                s.append(f"  {r['run']:>3}  {r['sign']:>9}  {r['burst_db']:>13.1f}  {r['lf_sum_db']:>9.1f}  {r['verdict']}")
        for k, c in pol["channels"].items():
            s.append(f"  ch {k}: LF sign {c['lf_sign']} HF sign {c['hf_sign']} ({c['lf_hf_relation']}), crossover notch: {c['crossover_notch']}")
        s += [""] + ["NOTE: " + n for n in pol["notes"]]
        if self.llm_json.get("polarity"):
            s += ["", "LLM REVIEW", json.dumps(self.llm_json["polarity"], indent=1)]
        self.pol_text.insert("end", "\n".join(s))

    def review_polarity(self):
        if not self.last:
            return
        ev = self.last["ev"]["polarity"]
        prompt = prompts.POLARITY.format(channels=self.cfg["channels"], ch0_name="Left", ch1_name="Right", setup=self.setup_text(),
                                         bursts=4, evidence=prompts.j({k: v for k, v in ev.items() if k != "algorithmic_verdict"}),
                                         verdict=prompts.j(ev["algorithmic_verdict"]),
                                         image_desc="(1) frequency response of L, R and both; (2) direct-sound impulse responses per channel, full band / LF / HF")
        imgs = [self.last["pngs"]["fr"], self.last["pngs"]["ir"]]

        def work():
            return self.llm().ask(prompts.SYSTEM, prompt, imgs, tag="polarity")

        def done(res):
            text, data = res
            self.llm_json["polarity"] = data
            self.log("LLM polarity review:\n" + text)
            self._refresh_4()
        self.run_bg(work, done, "LLM reviewing polarity...")

    # ------------------------------------------------------------------ step 6: room modes
    def _modes_frame(self):
        f = ttk.Frame(self.content, padding=8)
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Button(row, text="Ask LLM to review room modes", command=self.review_modes).pack(side="left")
        ttk.Button(row, text="I moved things -> re-measure", command=lambda: self.show_step(3)).pack(side="left", padx=4)
        ttk.Button(row, text="Use LLM's EQ list", command=self.use_llm_eq).pack(side="left", padx=4)
        ttk.Button(row, text="Accept corrections, continue ->", command=lambda: self.show_step(6)).pack(side="right")
        self.lf_canvas = tk.Canvas(f, bg="white", height=280)
        self.lf_canvas.pack(fill="both", expand=True, pady=4)
        self.lf_plot = plots.CanvasPlot(self.lf_canvas, "25-300 Hz, 1/12 oct: L, R, power average, deviation from trend")
        self.modes_text = scrolledtext.ScrolledText(f, wrap="word", height=12, font=("TkFixedFont", 9))
        self.modes_text.pack(fill="both", expand=True)
        return f

    def _refresh_5(self):
        self.modes_text.delete("1.0", "end")
        self.lf_plot.clear()
        if not self.last:
            self.modes_text.insert("end", "No measurement yet.")
            self.lf_plot.draw()
            return
        m = self.last["ev"]["room_modes"]
        g = np.array(m["grid"])
        for k, lab in (("0", "Left"), ("1", "Right"), ("avg", "L/R avg")):
            if k in m["responses"]:
                self.lf_plot.add(g, np.array(m["responses"][k]) - np.mean(m["responses"][k]), lab)
        self.lf_plot.add(g, np.array(m["excess_db"]), "deviation from 1-oct trend", color="#333", dash=(4, 2))
        for p in m["peaks"]:
            self.lf_plot.mark(p["freq"], f"{p['freq']}Hz +{p['excess_db']}")
        self.lf_plot.draw()
        s = ["DETECTED PEAKS (candidates for EQ cuts)"]
        for p in m["peaks"]:
            s.append(f"  {p['freq']:>6.1f} Hz  +{p['excess_db']:.1f} dB  Q~{p['q_est']}  per-channel {p['per_channel_excess_db']}  {p['predicted_match'] or ''}")
        s.append("DEEP DIPS (cannot be EQ'd; move speakers / listening position)")
        for d in m["dips"]:
            s.append(f"  {d['freq']:>6.1f} Hz  -{d['depth_db']:.1f} dB")
        s.append("OCTAVE BANDS re 200-2000 Hz midband (dB): " + json.dumps(m.get("octave_band_levels_re_midband_db", {})))
        s.append("PREDICTED AXIAL MODES from room dimensions: " + ", ".join(f"{p['freq']} ({p['type']})" for p in m["predicted"]) if m["predicted"] else
                 "No room dimensions given - enter them in Setup to label modes.")
        s += ["", "CURRENT CORRECTIONS", eqexport.as_table(self.corrections)]
        if self.llm_json.get("modes"):
            s += ["", "LLM REVIEW", json.dumps(self.llm_json["modes"], indent=1)]
        self.modes_text.insert("end", "\n".join(s))

    def review_modes(self):
        if not self.last:
            return
        m = self.last["ev"]["room_modes"]
        ev = {k: v for k, v in m.items() if k not in ("grid", "responses", "excess_db")}
        # compact numeric table for the LLM: every 4th grid point
        g = m["grid"]
        ev["response_table_hz_db"] = [[g[i], m["responses"]["avg"][i], round(m["excess_db"][i], 1)] for i in range(0, len(g), 4)]
        prompt = prompts.MODES.format(setup=self.setup_text(), dims=self.cfg["room_dims"] or "unknown", mic_pos=self.cfg["mic_pos"] or "unknown",
                                      mic_comp=(self.session.state.get("mic_profile") or {}).get("note", "none"),
                                      evidence=prompts.j(ev), corrections=prompts.j(self.corrections),
                                      image_desc="(1) 25-300 Hz response per channel with detected peaks marked; (2) full-range response")
        imgs = [self.last["pngs"]["lf"], self.last["pngs"]["fr"]]

        def work():
            return self.llm().ask(prompts.SYSTEM, prompt, imgs, tag="room_modes")

        def done(res):
            text, data = res
            self.llm_json["modes"] = data
            self.log("LLM room-mode review:\n" + text)
            self._refresh_5()
        self.run_bg(work, done, "LLM reviewing room modes...")

    def use_llm_eq(self):
        eq = (self.llm_json.get("modes") or {}).get("eq")
        if not eq:
            messagebox.showinfo("No LLM EQ", "Run the LLM review first; it returned no EQ list.")
            return
        self.corrections = [{"type": "peaking", "freq_hz": float(e["freq_hz"]), "gain_db": float(e["gain_db"]), "q": float(e.get("q", 4.0)),
                             "reason": "LLM-reviewed"} for e in eq]
        self._refresh_5()

    # ------------------------------------------------------------------ step 7: export
    def _export_frame(self):
        f = ttk.Frame(self.content, padding=8)
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Button(row, text="Save EasyEffects/APO file...", command=self.save_apo).pack(side="left")
        ttk.Button(row, text="Install PipeWire 'Room EQ' sink", command=self.install_pw).pack(side="left", padx=4)
        ttk.Button(row, text="Remove PipeWire EQ", command=self.remove_pw).pack(side="left")
        ttk.Button(row, text="LLM hand-over guidance", command=self.final_guidance).pack(side="left", padx=4)
        ttk.Button(row, text="Open session folder", command=lambda: subprocess.Popen(["xdg-open", self.session.dir])).pack(side="right")
        self.exp_text = scrolledtext.ScrolledText(f, wrap="word", font=("TkFixedFont", 9))
        self.exp_text.pack(fill="both", expand=True, pady=4)
        return f

    def _refresh_6(self):
        self.exp_text.delete("1.0", "end")
        self.exp_text.insert("end", "FINAL CORRECTIONS\n" + eqexport.as_table(self.corrections) + "\n\nEqualizer APO / EasyEffects format:\n" +
                             eqexport.as_apo(self.corrections))
        if self.llm_json.get("final"):
            self.exp_text.insert("end", "\nLLM GUIDANCE\n" + json.dumps(self.llm_json["final"], indent=1))
        json.dump(self.corrections, open(self.session.path("corrections.json"), "w"), indent=1)
        open(self.session.path("corrections_apo.txt"), "w").write(eqexport.as_apo(self.corrections))

    def save_apo(self):
        p = filedialog.asksaveasfilename(defaultextension=".txt", initialfile="room_eq_apo.txt")
        if p:
            open(p, "w").write(eqexport.as_apo(self.corrections))
            self.log(f"Saved {p}.  EasyEffects: Equalizer -> Import APO preset.")

    def install_pw(self):
        snk = self.sink()
        if not self.corrections or not snk:
            messagebox.showinfo("Nothing to install", "No corrections / no sink selected.")
            return
        if not messagebox.askyesno("Install", f"Write {eqexport.PW_CONF_FILE} creating a 'Room EQ' sink that outputs to\n{snk['desc']}\n"
                                              "and restart PipeWire (audio will drop for a second)?"):
            return
        p = eqexport.install_pipewire(self.corrections, snk["name"])
        self.log(f"Installed {p}. Select 'Room EQ' as output device (or: wpctl set-default <id of aaa_room_eq_sink>).")

    def remove_pw(self):
        self.log("Removed PipeWire EQ." if eqexport.uninstall_pipewire() else "No PipeWire EQ installed.")

    def final_guidance(self):
        prompt = prompts.FINAL.format(corrections=prompts.j(self.corrections), setup=self.setup_text())

        def work():
            return self.llm().ask(prompts.SYSTEM, prompt, tag="final")

        def done(res):
            text, data = res
            self.llm_json["final"] = data
            self.log("LLM hand-over guidance:\n" + text)
            self._refresh_6()
        self.run_bg(work, done, "LLM writing guidance...")

    def export_wav(self, data, default_name):
        p = filedialog.asksaveasfilename(defaultextension=".wav", initialfile=default_name, filetypes=[("WAV", "*.wav")])
        if p:
            audio_io.write_wav(p, data, SR)
            self.log(f"Saved {p} (48 kHz stereo float WAV; convert with ffmpeg if your player needs 16-bit).")

    # ------------------------------------------------------------------ free question
    def ask_free(self):
        qtext = self.question.get().strip()
        if not qtext:
            return
        self.question.delete(0, "end")
        summary = {}
        if self.last:
            summary = {"polarity_verdict": self.last["ev"]["polarity"]["algorithmic_verdict"], "pairs": self.last["ev"]["polarity"]["pairs"],
                       "lf_peaks": self.last["ev"]["room_modes"]["peaks"], "dips": self.last["ev"]["room_modes"]["dips"],
                       "corrections": self.corrections, "setup": self.setup_text()}
        prompt = prompts.FREE.format(question=qtext, evidence=prompts.j(summary))

        def work():
            return self.llm().ask(prompts.SYSTEM, prompt, tag="question")

        def done(res):
            self.log(f"Q: {qtext}\nA: {res[0]}")
        self.log(f"Q: {qtext}")
        self.run_bg(work, done, "asking LLM...")


def main():
    App().mainloop()
