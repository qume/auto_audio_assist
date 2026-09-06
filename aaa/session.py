"""Session directory: everything needed to audit or repeat a run lives here."""
import json
import os
import time

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sessions")
CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.json")
MICS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mics")


class Session:
    def __init__(self, path=None):
        self.dir = path or os.path.join(ROOT, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.state_path = os.path.join(self.dir, "state.json")
        self.state = json.load(open(self.state_path)) if os.path.exists(self.state_path) else {"runs": [], "log": []}

    def path(self, *p):
        return os.path.join(self.dir, *p)

    def save(self):
        json.dump(self.state, open(self.state_path, "w"), indent=1, default=str)

    def log(self, msg):
        self.state["log"].append(f"{time.strftime('%H:%M:%S')} {msg}")
        self.save()

    def new_run(self):
        n = len(self.state["runs"]) + 1
        d = self.path(f"run{n:02d}")
        os.makedirs(d, exist_ok=True)
        self.state["runs"].append({"dir": d, "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.save()
        return d, n


def load_config():
    if os.path.exists(CONFIG):
        try:
            return json.load(open(CONFIG))
        except Exception:
            pass
    return {}


def save_config(cfg):
    json.dump(cfg, open(CONFIG, "w"), indent=1)


def mic_profile_path(key):
    os.makedirs(MICS, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in key)[:80]
    return os.path.join(MICS, safe + ".json")
