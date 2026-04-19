"""
deltavision selftest — staged E2E verification.

Runs the full "as intended" user journey as a sequence of independent
stages. Each stage reports ✓ or ✗ with what it checked. If any stage
fails, the script stops with exit code 1 so CI + dogfood scripts can
distinguish "works" from "broken" without reading prose.

Usage:
    deltavision selftest          # via console-script entrypoint
    python -m deltavision.selftest
    python -c "from deltavision.selftest import main; main()"

What the stages cover:

    S1  import deltavision + namespace access
    S2  observer construction
    S3  first observation returns full_frame (initial bootstrap)
    S4  second observation (small delta) is cheaper than full frame
    S5  third observation (whole-frame change) triggers the coverage
        guard and falls back to full_frame — DV token cost must
        not exceed FF baseline on any single step
    S6  adapter output (Anthropic tool_result shape) is well-formed
    S7  (optional) HTTP sidecar: boot server, /health reports a
        matching version, /observe round-trips a DVObservation,
        /reset clears state. Skipped if --no-http is passed.

No API keys, no network, no authenticated endpoints — everything runs
on localhost with synthetic frames.
"""
from __future__ import annotations

import argparse
import io
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request

# ANSI color helpers — degrade gracefully if piped.
_ISATTY = sys.stdout.isatty()
def _c(code: str, s: str) -> str:
    return f"\x1b[{code}m{s}\x1b[0m" if _ISATTY else s
def green(s):  return _c("32", s)
def red(s):    return _c("31", s)
def yellow(s): return _c("33", s)
def dim(s):    return _c("2",  s)
def bold(s):   return _c("1",  s)


class StageError(Exception):
    pass


def _synthetic_frame(w: int = 1280, h: int = 800, fill=(255, 255, 255), mark=None):
    """A deterministic PNG used for the E2E stages.

    The frame is intentionally dense: dozens of blocks of content spread
    across the viewport. This anchors pHash to a stable signature so that
    a small local mutation (the `mark` param) registers as a *delta*, not
    a whole-page change. A sparse frame (white + one blob) causes pHash
    to classify tiny mutations as NEW_PAGE, which would make S4 unstable.
    """
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (w, h), fill)
    d = ImageDraw.Draw(img)
    # Grid
    for y in range(0, h, 40):
        d.line([(0, y), (w, y)], fill=(230, 230, 230), width=1)
    for x in range(0, w, 80):
        d.line([(x, 0), (x, h)], fill=(230, 230, 230), width=1)
    # Persistent "nav bar"
    d.rectangle([0, 0, w, 48], fill=(40, 40, 50))
    d.rectangle([16, 14, 180, 34], fill=(200, 200, 210))
    # A lot of "content blocks" throughout the frame — anchors pHash
    for row in range(4):
        for col in range(6):
            x0 = 40 + col * 200
            y0 = 80 + row * 160
            # Alternating block colors for visual diversity
            blk = (235, 238, 245) if (row + col) % 2 == 0 else (245, 240, 230)
            d.rectangle([x0, y0, x0 + 170, y0 + 130], fill=blk)
            d.rectangle([x0 + 8, y0 + 8, x0 + 100, y0 + 24], fill=(70, 110, 170))
            d.rectangle([x0 + 8, y0 + 36, x0 + 162, y0 + 44], fill=(150, 150, 155))
            d.rectangle([x0 + 8, y0 + 52, x0 + 140, y0 + 60], fill=(180, 180, 185))
            d.rectangle([x0 + 8, y0 + 68, x0 + 120, y0 + 76], fill=(180, 180, 185))
    # Mark (user-provided local mutation goes ON TOP of the stable layout)
    if mark:
        d.rectangle(mark["rect"], fill=mark["fill"])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ff_tokens(w: int = 1280, h: int = 800) -> int:
    return max(75, int((w * h) / 750))


def stage(name: str):
    """Decorator: prints stage header + catches StageError → red ✗."""
    def deco(fn):
        def wrapper(*args, **kwargs):
            header = f"  {bold(name):<52}"
            sys.stdout.write(header)
            sys.stdout.flush()
            try:
                detail = fn(*args, **kwargs)
                sys.stdout.write(green("✓") + f"  {dim(detail or '')}\n")
                return True
            except StageError as e:
                sys.stdout.write(red("✗") + f"  {red(str(e))}\n")
                return False
            except Exception as e:
                sys.stdout.write(red("✗") + f"  {red(f'unexpected: {type(e).__name__}: {e}')}\n")
                return False
        return wrapper
    return deco


@stage("S1  import deltavision")
def stage_1_import():
    import deltavision
    v = getattr(deltavision, "__version__", None)
    if not v:
        raise StageError("no __version__ attribute")
    if not hasattr(deltavision, "DeltaVisionObserver"):
        raise StageError("DeltaVisionObserver not in public API")
    return f"v{v}"


@stage("S2  observer construction")
def stage_2_observer():
    from deltavision import DeltaVisionConfig, DeltaVisionObserver
    obs = DeltaVisionObserver()
    if obs.config is None:
        raise StageError("config is None after construction")
    if not isinstance(obs.config, DeltaVisionConfig):
        raise StageError(f"config is {type(obs.config).__name__}, expected DeltaVisionConfig")
    return f"{type(obs).__name__} created with default config ({len(obs.config.__dataclass_fields__)} fields)"


_STATE = {}


@stage("S3  initial observation → full_frame")
def stage_3_initial():
    from deltavision import DeltaVisionObserver
    obs = DeltaVisionObserver()
    _STATE["obs"] = obs
    frame = _synthetic_frame()
    r = obs.observe(frame, url="http://test.local", last_action="initial")
    if r.obs_type != "full_frame":
        raise StageError(f"expected full_frame, got {r.obs_type}")
    if r.trigger != "initial":
        raise StageError(f"expected trigger=initial, got {r.trigger!r}")
    ff = _ff_tokens()
    tok = r.estimated_image_tokens()
    if tok != ff:
        raise StageError(f"initial token cost {tok} != FF baseline {ff}")
    return f"obs_type=full_frame trigger=initial tokens={tok}"


@stage("S4  small delta is cheaper than full frame")
def stage_4_delta():
    obs = _STATE["obs"]
    # Tiny change in the top-left area — only ~3% of the frame.
    frame = _synthetic_frame(mark={"rect": [150, 150, 340, 180], "fill": (90, 90, 255)})
    r = obs.observe(frame, url="http://test.local", last_action="type 'hello'")
    ff = _ff_tokens()
    tok = r.estimated_image_tokens()
    if tok >= ff:
        raise StageError(
            f"DV cost {tok} ≥ FF baseline {ff} on small delta — "
            f"compression failed. obs_type={r.obs_type} trigger={r.trigger!r}"
        )
    return f"tokens={tok} vs FF={ff} ({(1 - tok/ff) * 100:.1f}% saved) obs_type={r.obs_type}"


@stage("S5  whole-frame change → coverage guard fires")
def stage_5_guard():
    obs = _STATE["obs"]
    # Entire frame changes color — crops-cover-frame guard should fire.
    frame = _synthetic_frame(fill=(20, 20, 20))
    r = obs.observe(frame, url="http://test.local", last_action="scroll")
    ff = _ff_tokens()
    tok = r.estimated_image_tokens()
    if tok > ff:
        raise StageError(
            f"DV cost {tok} > FF baseline {ff} — invariant violated. "
            f"obs_type={r.obs_type} trigger={r.trigger!r} crops={len(r.crops or [])}"
        )
    # It should end up as full_frame (either via NEW_PAGE classification or the
    # coverage guard — both are correct responses to a whole-frame change).
    if r.obs_type != "full_frame":
        raise StageError(
            f"expected full_frame on whole-frame change, got {r.obs_type} "
            f"(trigger={r.trigger!r})"
        )
    return f"tokens={tok} = FF {ff} (guard trigger={r.trigger!r})"


@stage("S6  anthropic adapter output is well-formed")
def stage_6_adapter():
    obs = _STATE["obs"]
    # Use the observation we just made
    # Reuse S5's frame — which is already in the observer's state
    # Just call the adapter on the most recent observation.
    # We need to make another observation to have a fresh one, since
    # s5's return isn't stored. Re-observe.
    frame = _synthetic_frame()
    r = obs.observe(frame, url="http://test.local", last_action="reset")
    blocks = r.to_anthropic_tool_result_content()
    if not isinstance(blocks, list):
        raise StageError(f"adapter returned {type(blocks).__name__}, expected list")
    if not blocks:
        raise StageError("adapter returned empty list")
    for i, blk in enumerate(blocks):
        if not isinstance(blk, dict):
            raise StageError(f"block[{i}] is {type(blk).__name__}, expected dict")
        if blk.get("type") not in {"text", "image"}:
            raise StageError(f"block[{i}] has unknown type={blk.get('type')!r}")
    return f"{len(blocks)} content blocks, all valid"


# ---------- HTTP sidecar stages ----------

def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_server_in_thread(port: int):
    """Start server.py's HTTPServer on a background thread."""
    from http.server import ThreadingHTTPServer

    import server as srv
    # Reset the process-global observer so stage 7 sees fresh state
    srv.OBSERVER.reset()
    srv._server = ThreadingHTTPServer(("127.0.0.1", port), srv.Handler)

    def _serve():
        try:
            srv._server.serve_forever()
        except Exception:
            pass

    t = threading.Thread(target=_serve, daemon=True)
    t.start()
    # Wait up to 3s for the port to accept
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return srv
        except OSError:
            time.sleep(0.05)
    raise StageError(f"server didn't come up on :{port}")


def _http_get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read().decode())


def _http_post_multipart(url: str, png_bytes: bytes, fields: dict) -> dict:
    boundary = "----dvsttest"
    body = io.BytesIO()
    for k, v in fields.items():
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        body.write(str(v).encode())
        body.write(b"\r\n")
    body.write(f"--{boundary}\r\n".encode())
    body.write(b'Content-Disposition: form-data; name="file"; filename="f.png"\r\n')
    body.write(b"Content-Type: image/png\r\n\r\n")
    body.write(png_bytes)
    body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        url,
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def _http_post_empty(url: str) -> dict:
    req = urllib.request.Request(url, data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        txt = r.read().decode()
        return json.loads(txt) if txt else {}


@stage("S7  HTTP /health reports package version")
def stage_7_health():
    port = _find_free_port()
    srv = _start_server_in_thread(port)
    _STATE["http_srv"] = srv
    _STATE["http_port"] = port
    data = _http_get(f"http://127.0.0.1:{port}/health")
    if data.get("status") != "ok":
        raise StageError(f"health returned status={data.get('status')!r}")
    import deltavision
    if data.get("version") != deltavision.__version__:
        raise StageError(
            f"/health says version={data.get('version')!r} but "
            f"deltavision.__version__={deltavision.__version__}"
        )
    return f"version={data['version']} port={port}"


@stage("S8  HTTP /observe round-trips a DVObservation")
def stage_8_observe():
    port = _STATE["http_port"]
    png = _synthetic_frame()
    data = _http_post_multipart(
        f"http://127.0.0.1:{port}/observe",
        png,
        {"url": "http://test.local", "last_action": "initial", "format": "raw"},
    )
    if data.get("obs_type") not in {"full_frame", "delta"}:
        raise StageError(f"got obs_type={data.get('obs_type')!r}")
    if "estimated_tokens" not in data and "payload" not in data:
        raise StageError(f"missing expected keys; got keys={list(data)}")
    return f"obs_type={data.get('obs_type')} (HTTP round-trip ok)"


@stage("S9  HTTP /reset clears observer state")
def stage_9_reset():
    port = _STATE["http_port"]
    resp = _http_post_empty(f"http://127.0.0.1:{port}/reset")
    # /reset's success status is "reset" (not "ok" like the others)
    if resp.get("status") not in {"ok", "reset"}:
        raise StageError(f"reset returned {resp!r}")
    # Subsequent /observe should go back to step=0 / full_frame (bootstrap)
    png = _synthetic_frame()
    data = _http_post_multipart(
        f"http://127.0.0.1:{port}/observe",
        png,
        {"url": "http://test.local", "last_action": "after-reset", "format": "raw"},
    )
    if data.get("obs_type") != "full_frame":
        raise StageError(
            f"after /reset, next /observe should be full_frame; got {data.get('obs_type')!r}"
        )
    return "reset confirmed — observer re-bootstrapped"


# ---------- driver ----------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="DeltaVision staged end-to-end self-test."
    )
    parser.add_argument(
        "--no-http",
        action="store_true",
        help="Skip HTTP sidecar stages (S7-S9). "
             "Useful if ports are restricted or in sandboxed CI.",
    )
    args = parser.parse_args(argv)

    print()
    print(bold("DeltaVision self-test") + dim(" — staged E2E"))
    print(dim("  Each stage is independent. A ✗ locates the failure without ambiguity.") + "\n")

    stages = [stage_1_import, stage_2_observer, stage_3_initial,
              stage_4_delta, stage_5_guard, stage_6_adapter]
    if not args.no_http:
        stages += [stage_7_health, stage_8_observe, stage_9_reset]

    n_pass = 0
    for fn in stages:
        if fn():
            n_pass += 1
        else:
            # Don't continue past a failure — later stages depend on state.
            print()
            print(red(bold(f"  FAILED at {fn.__name__}")) + "  " + dim(f"({n_pass}/{len(stages)} stages passed)"))
            print()
            return 1

    print()
    print(green(bold(f"  All {len(stages)} stages passed.")))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
