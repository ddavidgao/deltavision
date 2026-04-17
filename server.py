"""
DeltaVision HTTP sidecar.

Exposes the Observer as a local HTTP service so non-Python CU frameworks
(OpenClaw / Node-based agents / Rust / Go bots) can POST screenshots and
get back the packaged observation in whatever format they consume.

Run:
    python server.py [--port 9000] [--host 127.0.0.1]

API:
    GET  /health
        → {"status": "ok", "version": "1.0.0"}

    POST /observe
        Request (one of):
            - multipart/form-data: file=<png>, url=<str>, last_action=<str>, format=<str>
            - application/json: {"screenshot_b64": <b64>, "url": ..., "last_action": ...,
                                 "format": "anthropic" | "browser_use" | "skyvern" | "openai" | "stagehand" | "raw"}
        Response: JSON matching the chosen format adapter.

    POST /reset
        Drops the observer's internal state (t0, anchor). Call at the start
        of a new agent run.

    GET  /state
        Introspect: returns current step count, last classification, etc.

The server holds ONE observer instance per process. For multi-session usage,
run multiple servers on different ports, or add session=<id> query parameter
support (not yet implemented — let us know if you need it).

This sidecar is intentionally minimal. No auth, no rate limiting, no TLS.
Meant to run on localhost only.
"""

import argparse
import base64
import io
import json
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from observer import DeltaVisionObserver

log = logging.getLogger("dv-server")


# Global observer — one per process. For multi-session, run multiple processes.
OBSERVER = DeltaVisionObserver()


# ============================================================= format dispatch

def render_format(obs, fmt: str, call_id: str | None = None) -> dict:
    """Render the observation in the requested format."""
    fmt = (fmt or "raw").lower()

    if fmt == "anthropic":
        return {"content": obs.to_anthropic_tool_result_content()}

    if fmt == "openai_cua":
        if not call_id:
            raise ValueError("format=openai_cua requires call_id")
        return obs.to_openai_computer_call_output(call_id)

    if fmt == "openai_vision":
        return {"content": obs.to_openai_vision_content()}

    if fmt == "browser_use":
        return {"screenshot_b64": obs.to_browser_use_screenshot_b64()}

    if fmt == "skyvern":
        return {"screenshots_b64": [
            base64.b64encode(s).decode() for s in obs.to_skyvern_screenshots_list()
        ]}

    if fmt == "stagehand":
        return {"parts": obs.to_stagehand_middleware_parts()}

    if fmt == "raw":
        raw = obs.to_raw()
        # Convert PIL images to base64 PNG so the response is JSON-safe
        def _pil_to_b64(img):
            if img is None:
                return None
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            return base64.b64encode(buf.getvalue()).decode()

        raw["frame_b64"] = _pil_to_b64(raw.pop("frame"))
        raw["thumbnail_b64"] = _pil_to_b64(raw.pop("thumbnail"))
        raw["crops_b64"] = [_pil_to_b64(c) for c in raw.pop("crops")]
        return raw

    raise ValueError(f"unknown format: {fmt!r}")


# ============================================================= handler

class Handler(BaseHTTPRequestHandler):
    server_version = "DeltaVision-Server/1.0"

    def log_message(self, fmt, *args):
        log.info("%s - %s", self.address_string(), fmt % args)

    def _respond_json(self, code: int, body: Any):
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    # -------- routes --------

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            return self._respond_json(200, {"status": "ok", "version": "1.0.0"})
        if parsed.path == "/state":
            cls = OBSERVER.last_classification
            return self._respond_json(200, {
                "step": OBSERVER.step,
                "no_change_streak": OBSERVER.no_change_streak,
                "last_classification": {
                    "transition": cls.transition.value if cls else None,
                    "trigger": cls.trigger if cls else None,
                    "diff_ratio": cls.diff_ratio if cls else None,
                    "phash_distance": cls.phash_distance if cls else None,
                    "anchor_score": cls.anchor_score if cls else None,
                } if cls else None,
            })
        return self._respond_json(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/reset":
            OBSERVER.reset()
            return self._respond_json(200, {"status": "reset"})
        if parsed.path == "/observe":
            return self._handle_observe()
        return self._respond_json(404, {"error": "not found"})

    def do_OPTIONS(self):
        # CORS preflight — keep it permissive, localhost-only use
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # -------- /observe body parsing --------

    def _handle_observe(self):
        content_type = (self.headers.get("Content-Type") or "").lower()
        body = self._read_body()

        screenshot = None
        url = None
        last_action = None
        fmt = "raw"
        call_id = None

        if "application/json" in content_type:
            try:
                data = json.loads(body.decode() or "{}")
            except json.JSONDecodeError as e:
                return self._respond_json(400, {"error": f"invalid json: {e}"})
            b64 = data.get("screenshot_b64")
            if not b64:
                return self._respond_json(400, {"error": "missing screenshot_b64"})
            screenshot = b64
            url = data.get("url")
            last_action = data.get("last_action")
            fmt = data.get("format", "raw")
            call_id = data.get("call_id")

        elif "multipart/form-data" in content_type:
            # Minimal multipart parser — enough for a file upload + fields.
            # For production traffic use a real ASGI/WSGI server. For local
            # sidecar traffic this is fine.
            import cgi  # noqa: E402  (stdlib)
            env = {"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type}
            fs = cgi.FieldStorage(
                fp=io.BytesIO(body), environ=env, keep_blank_values=True,
            )
            if "file" not in fs:
                return self._respond_json(400, {"error": "missing file field"})
            screenshot = fs["file"].file.read()
            url = fs.getfirst("url") or None
            last_action = fs.getfirst("last_action") or None
            fmt = fs.getfirst("format") or "raw"
            call_id = fs.getfirst("call_id") or None
        else:
            return self._respond_json(415, {
                "error": "use application/json or multipart/form-data"
            })

        try:
            obs = OBSERVER.observe(screenshot, url=url, last_action=last_action)
        except Exception as e:
            log.exception("observe failed")
            return self._respond_json(500, {"error": f"observe failed: {e}"})

        try:
            rendered = render_format(obs, fmt, call_id=call_id)
        except ValueError as e:
            return self._respond_json(400, {"error": str(e)})

        return self._respond_json(200, {
            "obs_type": obs.obs_type,
            "trigger": obs.trigger,
            "diff_ratio": obs.diff_ratio,
            "phash_distance": obs.phash_distance,
            "anchor_score": obs.anchor_score,
            "estimated_tokens": obs.estimated_image_tokens(),
            "format": fmt,
            "payload": rendered,
        })


# ============================================================= main

def main():
    p = argparse.ArgumentParser(description="DeltaVision HTTP sidecar")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("DeltaVision sidecar listening on http://%s:%d", args.host, args.port)
    log.info("POST /observe  GET /state  POST /reset  GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("shutting down")


if __name__ == "__main__":
    main()
