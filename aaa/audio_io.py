"""Device discovery and synchronous play+record via PipeWire (pw-play/pw-record) or raw ALSA (arecord).

Raw ALSA capture exists because on Asahi Linux the internal mic array is only reachable directly
(the PipeWire filter chain in front of it may be mis-linked and it applies beamforming plus a
120 Hz high-pass that ruins room-mode measurements).
"""
import json
import os
import re
import signal
import subprocess
import time
import numpy as np

SR = 48000


def _pw_nodes():
    try:
        out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=10).stdout
        return [o for o in json.loads(out) if o.get("type") == "PipeWire:Interface:Node"]
    except Exception:
        return []


def list_sinks():
    res = []
    for o in _pw_nodes():
        p = o.get("info", {}).get("props", {})
        if p.get("media.class") == "Audio/Sink":
            res.append({"id": o["id"], "name": p.get("node.name"), "desc": p.get("node.description", p.get("node.name"))})
    return res


def list_remote_sources(host, timeout=25):
    """ALSA capture devices on a remote box, as 'ssh:<host>:hw:CARD,DEV' sources."""
    if not host:
        return []
    try:
        out = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, "arecord -l"],
                             capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return []
    res = []
    for m in re.finditer(r"card (\d+): (\S+) \[([^\]]+)\], device (\d+): ([^\[]+)\[", out):
        card, cid, cname, dev, dname = m.groups()
        res.append({"id": None, "name": f"ssh:{host}:hw:{cid},{dev}", "kind": "ssh",
                    "desc": f"Remote mic on {host}: {cname.strip()} / {dname.strip()}"})
    return res


def list_sources(remote_host=""):
    """PipeWire sources, raw local ALSA devices ('alsa:'), and remote ALSA devices ('ssh:')."""
    res = []
    for o in _pw_nodes():
        p = o.get("info", {}).get("props", {})
        if p.get("media.class") == "Audio/Source":
            res.append({"id": o["id"], "name": p.get("node.name"), "desc": p.get("node.description", p.get("node.name")), "kind": "pw"})
    try:
        out = subprocess.run(["arecord", "-l"], capture_output=True, text=True, timeout=10).stdout
        for m in re.finditer(r"card (\d+): (\S+) \[([^\]]+)\], device (\d+): ([^\[]+)\[", out):
            card, cid, cname, dev, dname = m.groups()
            res.append({"id": None, "name": f"alsa:hw:{cid},{dev}", "desc": f"ALSA raw: {cname} / {dname.strip()}", "kind": "alsa"})
    except Exception:
        pass
    res += list_remote_sources(remote_host)
    return res


def default_sink_name():
    try:
        out = subprocess.run(["wpctl", "inspect", "@DEFAULT_AUDIO_SINK@"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r'node\.name = "([^"]+)"', out)
        return m.group(1) if m else None
    except Exception:
        return None


def set_default_sink(sink_id):
    subprocess.run(["wpctl", "set-default", str(sink_id)], check=False)


def set_sink_volume(sink_id, vol=1.0):
    subprocess.run(["wpctl", "set-volume", str(sink_id), f"{vol:.2f}"], check=False)


def write_wav(path, data, sr=SR):
    import scipy.io.wavfile as w
    if data.ndim == 1:
        data = data[:, None]
    w.write(path, sr, data.astype(np.float32))


ALSA_FORMAT = {"f32": "FLOAT_LE", "s32": "S32_LE", "s24_3": "S24_3LE", "s16": "S16_LE"}
SAMPLE_BYTES = {"f32": 4, "s32": 4, "s24_3": 3, "s16": 2}


def read_wav_robust(path, channels, fmt):
    """Read a possibly-truncated WAV or headerless raw capture written by arecord/pw-record.
    fmt: 'f32' | 's32' | 's24_3' (24-bit packed) | 's16'."""
    b = open(path, "rb").read()
    i = b.find(b"data")
    raw = b[i + 8:] if (i >= 0 and b[:4] == b"RIFF") else b
    fs = SAMPLE_BYTES[fmt] * channels
    raw = raw[: len(raw) - len(raw) % fs]
    if fmt == "s24_3":
        a = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = a[:, 0] | (a[:, 1] << 8) | (a[:, 2] << 16)
        v = np.where(v & 0x800000, v - (1 << 24), v)
        x = v.reshape(-1, channels).astype(np.float64) / 2 ** 23
    else:
        dt = {"f32": "<f4", "s32": "<i4", "s16": "<i2"}[fmt]
        x = np.frombuffer(raw, dtype=dt).reshape(-1, channels).astype(np.float64)
        if fmt == "s32":
            x /= 2 ** 31
        elif fmt == "s16":
            x /= 2 ** 15
    return x[np.isfinite(x).all(axis=1)]


class Recorder:
    """Background recorder.  source kinds:
      'pw'   local PipeWire node          -> pw-record
      'alsa' local ALSA device            -> arecord
      'ssh'  ALSA device on a remote box  -> ssh host arecord, streamed back to a local file

    Remote captures run for a fixed `duration` (seconds): killing an ssh client does not reliably
    kill the remote process, and every stage of this tool knows how long it needs to record.
    """

    def __init__(self, source, path, sr=SR, alsa_channels=None, alsa_format=None, duration=None):
        self.source, self.path, self.sr = source, path, sr
        self.kind = source.get("kind", "pw")
        self.channels, self.fmt = 1, "f32"
        self.duration = duration
        self.out = None
        if self.kind == "alsa":
            dev = source["name"].split("alsa:", 1)[1]
            self.channels = alsa_channels or probe_alsa_channels(dev)
            self.fmt = alsa_format or "s32"
            self.cmd = ["arecord", "-q", "-D", dev, "-c", str(self.channels), "-r", str(sr),
                        "-f", ALSA_FORMAT[self.fmt], path]
        elif self.kind == "ssh":
            host, dev = _ssh_parts(source["name"])
            self.channels = alsa_channels or 2
            self.fmt = alsa_format or "s24_3"
            if not duration:
                raise ValueError("remote (ssh) capture needs a duration")
            remote = (f"arecord -q -D {dev} -c {self.channels} -r {sr} -f {ALSA_FORMAT[self.fmt]} "
                      f"-t raw -d {int(np.ceil(duration))} -")
            self.cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote]
        else:
            self.cmd = ["pw-record", "--target", str(source.get("id") or source["name"]), "--rate", str(sr),
                        "--channels", "1", "--format", "f32", path]
        self.proc = None

    def start(self):
        if self.kind == "ssh":
            self.out = open(self.path, "wb")
            self.proc = subprocess.Popen(self.cmd, stdout=self.out, stderr=subprocess.PIPE)
            time.sleep(1.2)          # ssh handshake + arecord opening the device
        else:
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            time.sleep(0.4)
        if self.proc.poll() is not None:
            raise RuntimeError("recorder failed: " + self.proc.stderr.read().decode(errors="replace")[-400:])

    def stop(self):
        if self.proc and self.proc.poll() is None:
            if self.kind == "ssh":
                # the remote arecord stops on its own; wait for it, then fall back to killing ssh
                try:
                    self.proc.wait(timeout=(self.duration or 0) + 15)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            else:
                self.proc.send_signal(signal.SIGINT)
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
        err = self.proc.stderr.read().decode(errors="replace") if self.proc and self.proc.stderr else ""
        if self.out:
            self.out.close()
        time.sleep(0.2)
        x = read_wav_robust(self.path, self.channels, self.fmt)
        if len(x) < self.sr // 2:
            raise RuntimeError(f"capture produced only {len(x)} frames. {err[-400:]}")
        return x


def probe_alsa_channels(dev):
    try:
        out = subprocess.run(["arecord", "-D", dev, "--dump-hw-params", "-d", "1", "/dev/null"],
                             capture_output=True, text=True, timeout=10)
        m = re.search(r"CHANNELS:\s*\[?(\d+)", out.stdout + out.stderr)
        return int(m.group(1)) if m else 2
    except Exception:
        return 2


def probe_remote_alsa(host, dev, timeout=25):
    """(channels, fmt) for an ALSA capture device on a remote box, from arecord --dump-hw-params."""
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host,
                        f"arecord -D {dev} --dump-hw-params -d 1 /dev/null"],
                       capture_output=True, text=True, timeout=timeout)
    out = r.stdout + r.stderr
    m = re.search(r"CHANNELS:\s*\[?(\d+)", out)
    ch = int(m.group(1)) if m else 2
    fmts = re.search(r"FORMAT:\s*(.+)", out)
    have = fmts.group(1).split() if fmts else []
    for key, name in (("f32", "FLOAT_LE"), ("s32", "S32_LE"), ("s24_3", "S24_3LE"), ("s16", "S16_LE")):
        if name in have:
            return ch, key
    return ch, "s16"


def probe_alsa_format(source, sr=SR):
    """Some drivers (Apple AOP mics) deliver float32 regardless of the requested format.
    Record 0.5 s as S32 and check whether the bytes make more sense as float32."""
    dev = source["name"].split("alsa:", 1)[1]
    ch = probe_alsa_channels(dev)
    path = "/tmp/aaa_probe.wav"
    subprocess.run(["arecord", "-q", "-D", dev, "-c", str(ch), "-r", str(sr), "-f", "S32_LE", "-d", "1", path],
                   capture_output=True, timeout=15)
    b = open(path, "rb").read()
    i = b.find(b"data")
    raw = b[i + 8:]
    raw = raw[: len(raw) - len(raw) % (4 * ch)]
    as_i = np.frombuffer(raw, "<i4").astype(np.float64) / 2 ** 31
    as_f = np.frombuffer(raw, "<f4").astype(np.float64)
    f_ok = np.isfinite(as_f).all() and np.abs(as_f).max() <= 1.5
    i_rms = np.sqrt(np.mean(as_i ** 2))
    fmt = "f32" if (f_ok and i_rms > 0.2) else "s32"
    return ch, fmt


def record_only(source, seconds, workdir, sr=SR, alsa_channels=None, alsa_format=None):
    os.makedirs(workdir, exist_ok=True)
    rec = Recorder(source, os.path.join(workdir, "noise_floor.wav"), sr, alsa_channels, alsa_format,
                   duration=seconds)
    rec.start()
    time.sleep(seconds)
    x = rec.stop()
    x -= x.mean(axis=0)
    return x[:, 0]


def laptop_model():
    parts = []
    for f in ("/sys/devices/virtual/dmi/id/sys_vendor", "/sys/devices/virtual/dmi/id/product_name", "/proc/device-tree/model"):
        try:
            parts.append(open(f).read().strip("\x00\n "))
        except Exception:
            pass
    return " ".join(dict.fromkeys(parts)) or "unknown computer"


# ---------------------------------------------------------------- remote / manual outputs
def output_options(remote_host="", remote_dev="default"):
    """All ways to get sound to the speakers: PipeWire sinks, ssh->aplay on a remote box, manual file."""
    opts = [dict(s, kind="pw") for s in list_sinks()]
    if remote_host:
        opts.append({"id": None, "name": f"ssh:{remote_host}:{remote_dev or 'default'}", "kind": "ssh",
                     "desc": f"Remote via ssh: {remote_host} -> " + (f"pw-play --target {remote_dev[3:]}" if remote_dev.startswith("pw:") else f"aplay -D {remote_dev or 'default'}")})
    # remote-side tail: AirPlay/RAOP adds ~2 s of buffering, so keep recording longer after playback ends
    opts.append({"id": None, "name": "file:manual", "kind": "file",
                 "desc": "Manual / sneakernet: you play the exported WAV on any device, this app only records"})
    return opts


def _ssh_parts(name):
    _, host, dev = name.split(":", 2)
    return host, dev or "default"


def play(path, sink, blocking=True):
    """Start playback of a WAV through a sink option.  Returns a Popen (or None for manual mode)."""
    kind = sink.get("kind", "pw")
    if kind == "ssh":
        host, dev = _ssh_parts(sink["name"])
        if dev.startswith("pw:"):      # PipeWire node on the remote box (e.g. pw:raop-marantz)
            remote = f"pw-play --target {dev[3:]} -"
        else:                          # ALSA device on the remote box (e.g. default, hw:0,0)
            remote = f"aplay -q -t wav -D {dev} -"
        p = subprocess.Popen(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote],
                             stdin=open(path, "rb"), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    elif kind == "file":
        return None
    else:
        p = subprocess.Popen(["pw-play", "--target", str(sink.get("id") or sink["name"]), path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if blocking:
        p.wait()
    return p


def play_and_record(signal_stereo, sink, source, workdir, sr=SR, tail_s=1.5, alsa_channels=None, alsa_format=None,
                    progress=None, manual_wait_s=None):
    """Play a multichannel float32 array through `sink` while recording `source`.
    Manual mode ("file:manual"): just record for len(signal)+manual_wait_s seconds while the user
    plays the exported file themselves.  Returns (mono float64 recording, per-channel matrix)."""
    os.makedirs(workdir, exist_ok=True)
    stim = os.path.join(workdir, "stimulus.wav")
    recp = os.path.join(workdir, "recording_raw.wav")
    write_wav(stim, signal_stereo, sr)
    dur = len(signal_stereo) / sr
    # remote captures are fixed-length: cover playback plus network-renderer buffering and the tail
    extra = (manual_wait_s or 15.0) if sink.get("kind") == "file" else max(tail_s, 5.0) + 4.0
    rec = Recorder(source, recp, sr, alsa_channels, alsa_format, duration=dur + extra)
    rec.start()
    t0 = time.time()
    p = play(stim, sink, blocking=False)
    if p is None:  # manual mode
        total = dur + (manual_wait_s or 15.0)
        while time.time() - t0 < total:
            if progress:
                progress(min((time.time() - t0) / total, 1.0))
            time.sleep(0.2)
    else:
        while p.poll() is None:
            if progress:
                progress(min((time.time() - t0) / dur, 1.0))
            time.sleep(0.2)
        if p.returncode not in (0, None) and sink.get("kind") == "ssh":
            err = p.stderr.read().decode(errors="replace") if p.stderr else ""
            rec.stop()
            raise RuntimeError(f"remote playback failed ({p.returncode}): {err[-500:]}")
        # network renderers (AirPlay/RAOP, Chromecast) buffer ~2 s: record well past the end of playback
        time.sleep(max(tail_s, 5.0) if sink.get("kind") == "ssh" else tail_s)
    x = rec.stop()
    x -= x.mean(axis=0)
    # multi-capsule raw arrays (Apple: 3 capsules ~2 cm apart) are NOT averaged: the spacing would
    # smear the 3-10 kHz band used for the tweeter polarity check.  Capsule 0 only.
    mono = x[:, 0]
    write_wav(os.path.join(workdir, "recording_mono.wav"), mono.astype(np.float32), sr)
    return mono, x


# ---------------------------------------------------------------- system-wide sink -> remote ALSA box
PW_CONF_DIR = os.path.expanduser("~/.config/pipewire/pipewire.conf.d")
PW_SINK_CONF = os.path.join(PW_CONF_DIR, "50-aaa-remote-sink.conf")
SYSTEMD_DIR = os.path.expanduser("~/.config/systemd/user")
SERVICE = os.path.join(SYSTEMD_DIR, "aaa-remote-sink.service")
REMOTE_SINK = "aaa_remote_sink"


def install_remote_sink(host, dev="default", desc=None):
    """Create a persistent PipeWire sink whose audio is piped over ssh to `aplay` on `host`.
    Makes the remote box a system-wide output.  No restart needed (node created live as well)."""
    desc = desc or f"Amp via {host}"
    os.makedirs(PW_CONF_DIR, exist_ok=True)
    os.makedirs(SYSTEMD_DIR, exist_ok=True)
    node_args = (f'{{ factory.name=support.null-audio-sink node.name={REMOTE_SINK} node.description="{desc}" '
                 f'media.class=Audio/Sink object.linger=true audio.position=[FL FR] monitor.channel-volumes=true }}')
    with open(PW_SINK_CONF, "w") as f:
        f.write("# Generated by auto_audio_assist: virtual sink, streamed by aaa-remote-sink.service\n"
                f"context.objects = [\n  {{ factory = adapter args = {node_args} }}\n]\n")
    if not any(n["name"] == REMOTE_SINK for n in list_sinks()):
        subprocess.run(["pw-cli", "create-node", "adapter", node_args], capture_output=True)
    if dev.startswith("pw:"):
        remote_cmd = f"pw-play --target {dev[3:]} --raw --format s16 --rate 48000 --channels 2 -"
    else:
        remote_cmd = f"aplay -q -t raw -f S16_LE -r 48000 -c 2 -D {dev} --buffer-time=400000 -"
    pipe = (f"/usr/bin/sh -c 'pw-record --target {REMOTE_SINK} -P \"{{ stream.capture.sink = true }}\" --raw "
            f"--format s16 --rate 48000 --channels 2 --latency 100ms - | "
            f"ssh -o BatchMode=yes -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes {host} \"{remote_cmd}\"'")
    with open(SERVICE, "w") as f:
        f.write(f"""[Unit]
Description=auto_audio_assist: stream PipeWire sink {REMOTE_SINK} to {host} (aplay)
After=pipewire.service network-online.target
Requires=pipewire.service

[Service]
ExecStartPre=/usr/bin/sh -c 'for i in $(seq 30); do pw-cli info {REMOTE_SINK} >/dev/null 2>&1 && exit 0; sleep 1; done; exit 1'
ExecStart={pipe}
Restart=always
RestartSec=3

[Install]
WantedBy=default.target
""")
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    subprocess.run(["systemctl", "--user", "enable", "--now", "aaa-remote-sink.service"], check=False)
    return SERVICE


def remote_sink_status():
    r = subprocess.run(["systemctl", "--user", "is-active", "aaa-remote-sink.service"], capture_output=True, text=True)
    return r.stdout.strip()


def uninstall_remote_sink():
    subprocess.run(["systemctl", "--user", "disable", "--now", "aaa-remote-sink.service"], check=False)
    for p in (SERVICE, PW_SINK_CONF):
        if os.path.exists(p):
            os.remove(p)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
    for n in list_sinks():
        if n["name"] == REMOTE_SINK:
            subprocess.run(["pw-cli", "destroy", str(n["id"])], capture_output=True)
