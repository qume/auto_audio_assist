"""Plotting: lightweight Tk-canvas line plots for the UI, matplotlib PNGs for the LLM / session."""
import math
import numpy as np

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


class CanvasPlot:
    """Simple multi-trace plot on a tkinter Canvas.  x may be log-scaled."""

    def __init__(self, canvas, title="", xlog=True, xlabel="Hz", ylabel="dB"):
        self.c, self.title, self.xlog, self.xlabel, self.ylabel = canvas, title, xlog, xlabel, ylabel
        self.traces, self.marks, self.spans = [], [], []
        self.c.bind("<Configure>", lambda e: self.draw())

    def clear(self):
        self.traces, self.marks, self.spans = [], [], []
        self.draw()

    def add(self, x, y, label="", color=None, width=1.5, dash=None):
        self.traces.append((np.asarray(x, float), np.asarray(y, float), label, color or COLORS[len(self.traces) % len(COLORS)], width, dash))

    def mark(self, x, text, color="#d62728"):
        self.marks.append((x, text, color))

    def span(self, x0, x1, color="#eeeeee"):
        self.spans.append((x0, x1, color))

    def _tx(self, x):
        return math.log10(max(x, 1e-9)) if self.xlog else x

    def draw(self):
        c = self.c
        c.delete("all")
        W, H = max(c.winfo_width(), 50), max(c.winfo_height(), 50)
        ml, mr, mt, mb = 52, 14, 24, 32
        if self.title:
            c.create_text(W / 2, 12, text=self.title, font=("TkDefaultFont", 10, "bold"))
        if not self.traces:
            c.create_text(W / 2, H / 2, text="(no data)", fill="#888")
            return
        xs = np.concatenate([t[0] for t in self.traces])
        ys = np.concatenate([t[1] for t in self.traces])
        ys = ys[np.isfinite(ys)]
        x0, x1 = self._tx(xs.min()), self._tx(xs.max())
        y0, y1 = float(np.floor(ys.min() / 5) * 5 - 5), float(np.ceil(ys.max() / 5) * 5 + 5)
        if y1 - y0 < 10:
            y1 = y0 + 10
        def px(x):
            return ml + (self._tx(x) - x0) / (x1 - x0 + 1e-12) * (W - ml - mr)
        def py(y):
            return mt + (y1 - y) / (y1 - y0) * (H - mt - mb)
        for a, b, col in self.spans:
            c.create_rectangle(px(a), mt, px(b), H - mb, fill=col, outline="")
        # grid
        if self.xlog:
            ticks = [f for f in [20, 30, 50, 100, 200, 300, 500, 1000, 2000, 5000, 10000, 20000] if xs.min() <= f <= xs.max()]
        else:
            step = _nice((xs.max() - xs.min()) / 8)
            ticks = np.arange(math.ceil(xs.min() / step) * step, xs.max() + 1e-9, step)
        for f in ticks:
            X = px(f)
            c.create_line(X, mt, X, H - mb, fill="#ddd")
            lab = (f"{f/1000:g}k" if f >= 1000 else f"{f:g}") if self.xlog else f"{f:g}"
            c.create_text(X, H - mb + 10, text=lab, font=("TkDefaultFont", 8))
        ystep = _nice((y1 - y0) / 6)
        yv = y0
        while yv <= y1 + 1e-9:
            Y = py(yv)
            c.create_line(ml, Y, W - mr, Y, fill="#ddd")
            c.create_text(ml - 6, Y, text=f"{yv:g}", anchor="e", font=("TkDefaultFont", 8))
            yv += ystep
        c.create_rectangle(ml, mt, W - mr, H - mb, outline="#666")
        c.create_text(W - mr, H - 6, text=self.xlabel, anchor="e", font=("TkDefaultFont", 8))
        c.create_text(4, mt, text=self.ylabel, anchor="nw", font=("TkDefaultFont", 8))
        for i, (x, y, label, col, w, dash) in enumerate(self.traces):
            pts = []
            for a, b in zip(x, y):
                if np.isfinite(b):
                    pts += [px(a), py(b)]
            if len(pts) >= 4:
                c.create_line(*pts, fill=col, width=w, dash=dash, smooth=False)
            if label:
                c.create_line(ml + 10, mt + 12 + 14 * i, ml + 30, mt + 12 + 14 * i, fill=col, width=3)
                c.create_text(ml + 36, mt + 12 + 14 * i, text=label, anchor="w", font=("TkDefaultFont", 8), fill=col)
        for x, text, col in self.marks:
            X = px(x)
            c.create_line(X, mt, X, H - mb, fill=col, dash=(3, 3))
            c.create_text(X + 3, mt + 4, text=text, anchor="nw", fill=col, font=("TkDefaultFont", 8))


def _nice(v):
    if v <= 0:
        return 1
    e = 10 ** math.floor(math.log10(v))
    for m in (1, 2, 5, 10):
        if m * e >= v:
            return m * e
    return 10 * e


# ------------------------------------------------------------------ PNG rendering for the LLM
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def png_frequency_response(path, grid, responses, title, marks=(), xlim=(20, 20000), labels=None):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=110)
    for i, (k, r) in enumerate(responses.items()):
        ax.semilogx(grid, r, label=(labels or {}).get(k, k), color=COLORS[i % len(COLORS)])
    for f, txt in marks:
        ax.axvline(f, color="#d62728", ls="--", lw=0.8)
        ax.text(f, ax.get_ylim()[1] if ax.get_ylim()[1] else 0, txt, rotation=90, va="top", fontsize=7, color="#d62728")
    ax.set_xlim(*xlim)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB (relative)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def png_impulse(path, irs, sr, title, window_ms=12.0, bands=None):
    """Overlay the direct-sound part of each channel IR, optionally in bands (for polarity)."""
    plt = _plt()
    bands = bands or {"full": None, "LF 80-1200 Hz": (80, 1200), "HF 3-10 kHz": (3000, 10000)}
    from .analysis import bandpass
    fig, axes = plt.subplots(len(bands), 1, figsize=(9, 2.6 * len(bands)), dpi=110, sharex=True)
    axes = np.atleast_1d(axes)
    n = int(window_ms / 1000 * sr)
    for ax, (bname, band) in zip(axes, bands.items()):
        for i, (k, ir) in enumerate(irs.items()):
            if k == "all":
                continue
            x = bandpass(ir, band[0], band[1], sr) if band else ir
            x = x[:n] / (np.abs(x[:n]).max() + 1e-12)
            t = np.arange(n) / sr * 1000
            ax.plot(t, x, label=f"ch {k}", color=COLORS[i % len(COLORS)], lw=1)
        ax.set_ylabel(bname, fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="k", lw=0.5)
    axes[0].set_title(title)
    axes[0].legend()
    axes[-1].set_xlabel("ms (each channel aligned to its own direct-sound onset window)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
