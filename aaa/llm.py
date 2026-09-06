"""LLM backends: `claude` CLI (default on this machine), Anthropic Messages API, OpenAI-compatible.

Every call is logged (prompt, images, response) into the session directory so a run can be audited
and repeated.  The model is asked for a fenced ```json block at the end of every answer; we parse it.
"""
import base64
import json
import os
import re
import subprocess
import time
import urllib.request
import urllib.error

DEFAULTS = {
    "claude-cli": {"model": "fable", "url": "", "key": ""},
    "anthropic": {"model": "claude-opus-5", "url": "https://api.anthropic.com", "key": ""},
    "openai": {"model": "gpt-4o", "url": "https://api.openai.com/v1", "key": ""},
}


class LLM:
    def __init__(self, backend="claude-cli", model=None, url=None, key=None, log_dir=None, timeout=600):
        self.backend = backend
        d = DEFAULTS.get(backend, DEFAULTS["claude-cli"])
        self.model = model or d["model"]
        self.url = (url or d["url"]).rstrip("/")
        self.key = key or d["key"] or os.environ.get("ANTHROPIC_API_KEY" if backend == "anthropic" else "OPENAI_API_KEY", "")
        self.log_dir = log_dir
        self.timeout = timeout
        self.n = 0

    # ------------------------------------------------------------------ public
    def ask(self, system, prompt, images=(), tag="call"):
        self.n += 1
        t0 = time.time()
        if self.backend == "claude-cli":
            text = self._claude_cli(system, prompt, images)
        elif self.backend == "anthropic":
            text = self._anthropic(system, prompt, images)
        else:
            text = self._openai(system, prompt, images)
        data = extract_json(text)
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            with open(os.path.join(self.log_dir, f"{self.n:02d}_{tag}.md"), "w") as f:
                f.write(f"# {tag}  backend={self.backend} model={self.model} seconds={time.time()-t0:.1f}\n\n")
                f.write("## system\n\n" + system + "\n\n## prompt\n\n" + prompt + "\n\n## images\n\n" + "\n".join(images))
                f.write("\n\n## response\n\n" + text + "\n")
        return text, data

    # ------------------------------------------------------------------ backends
    def _claude_cli(self, system, prompt, images):
        full = prompt
        if images:
            full += "\n\nIMAGES: read each of these image files with the Read tool before answering:\n" + \
                    "\n".join(f"- {os.path.abspath(p)}" for p in images)
        cmd = ["claude", "-p", "--model", self.model, "--output-format", "json", "--no-session-persistence",
               "--system-prompt", system, "--tools", "Read", "--max-turns", str(4 + 2 * len(images))]
        r = subprocess.run(cmd, input=full, capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0 and not r.stdout.strip():
            raise RuntimeError(f"claude CLI failed ({r.returncode}): {r.stderr[-800:]}")
        try:
            j = json.loads(r.stdout)
            if isinstance(j, dict) and j.get("is_error"):
                raise RuntimeError("claude CLI error: " + str(j.get("result"))[:800])
            return j.get("result", "") if isinstance(j, dict) else str(j)
        except json.JSONDecodeError:
            return r.stdout

    def _http(self, url, headers, body):
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} from {url}: {e.read().decode(errors='replace')[:800]}")

    def _anthropic(self, system, prompt, images):
        content = []
        for p in images:
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": base64.b64encode(open(p, "rb").read()).decode()}})
        content.append({"type": "text", "text": prompt})
        body = {"model": self.model, "max_tokens": 8000, "system": system,
                "messages": [{"role": "user", "content": content}]}
        j = self._http(self.url + "/v1/messages", {"x-api-key": self.key, "anthropic-version": "2023-06-01"}, body)
        if j.get("stop_reason") == "refusal":
            raise RuntimeError("model refused: " + str(j.get("stop_details")))
        return "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")

    def _openai(self, system, prompt, images):
        content = [{"type": "text", "text": prompt}]
        for p in images:
            content.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                                                               base64.b64encode(open(p, "rb").read()).decode()}})
        body = {"model": self.model, "messages": [{"role": "system", "content": system},
                                                  {"role": "user", "content": content}], "temperature": 0}
        j = self._http(self.url + "/chat/completions", {"Authorization": f"Bearer {self.key}"}, body)
        return j["choices"][0]["message"]["content"]


def extract_json(text):
    """Return the last fenced ```json block (or last {...}) parsed, else None."""
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    for b in reversed(blocks):
        try:
            return json.loads(b)
        except json.JSONDecodeError:
            continue
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None
