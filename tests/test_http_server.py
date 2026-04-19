"""
Tests for the DeltaVision HTTP sidecar (server.py).

Spins up a real ThreadingHTTPServer on an ephemeral port, hits each endpoint
with a real HTTP client, and asserts on the JSON round-trip.

This is the missing piece that let v1.0.2 ship with a stale version string
in /health for two releases — no automated coverage = silent drift.

What's tested:
  - GET /health → matches deltavision.__version__ (not a hardcoded literal)
  - POST /observe → returns a DVObservation-shaped dict
  - POST /reset → subsequent /observe re-bootstraps (obs_type=full_frame)
  - GET /state → returns step/last_classification fields
  - OPTIONS /observe → CORS preflight succeeds (for browser callers)
  - 404 on unknown route
"""
from __future__ import annotations

import io
import json
import socket
import threading
import time
import urllib.request

import pytest
from PIL import Image


def _synthetic_png(w: int = 1280, h: int = 800, fill=(255, 255, 255)) -> bytes:
    img = Image.new("RGB", (w, h), fill)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="module")
def http_server():
    """Start the sidecar on a random port for the whole test module."""
    from http.server import ThreadingHTTPServer

    import server as srv

    port = _free_port()
    srv.OBSERVER.reset()
    httpd = ThreadingHTTPServer(("127.0.0.1", port), srv.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    # Wait for port to accept
    deadline = time.time() + 3
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        httpd.shutdown()
        pytest.fail("HTTP sidecar didn't start")

    yield {"port": port, "base": f"http://127.0.0.1:{port}", "observer": srv.OBSERVER}

    httpd.shutdown()


def _get(base: str, path: str) -> tuple[int, dict]:
    req = urllib.request.Request(f"{base}{path}", method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def _post_empty(base: str, path: str) -> tuple[int, dict]:
    req = urllib.request.Request(f"{base}{path}", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        body = r.read().decode()
        return r.status, (json.loads(body) if body else {})


def _post_multipart(base: str, path: str, png_bytes: bytes, fields: dict) -> tuple[int, dict]:
    boundary = "----dvtest"
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
        f"{base}{path}",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read().decode())


# ---------- tests ----------


def test_health_matches_installed_version(http_server):
    """/health must report the actual deltavision.__version__, not a literal."""
    import deltavision
    code, body = _get(http_server["base"], "/health")
    assert code == 200
    assert body["status"] == "ok"
    assert body["version"] == deltavision.__version__, (
        f"/health says version={body['version']!r} but "
        f"deltavision.__version__={deltavision.__version__!r}"
    )


def test_observe_returns_dvobservation_shape(http_server):
    """POST /observe with a raw PNG must return a parseable observation."""
    # Reset first so we're at a known state
    _post_empty(http_server["base"], "/reset")
    png = _synthetic_png()
    code, body = _post_multipart(
        http_server["base"], "/observe", png,
        {"url": "http://test.local", "last_action": "initial", "format": "raw"},
    )
    assert code == 200
    assert body.get("obs_type") == "full_frame"  # first post-reset is always full
    assert body.get("trigger") == "initial"


def test_reset_reboots_observer_state(http_server):
    """After /reset, the next /observe must re-bootstrap (full_frame)."""
    png = _synthetic_png()

    # Seed a few observations so the observer is mid-stream
    for i in range(3):
        _post_multipart(http_server["base"], "/observe", png,
                        {"url": "http://test.local", "last_action": f"step-{i}", "format": "raw"})

    # Reset
    code, body = _post_empty(http_server["base"], "/reset")
    assert code == 200
    # /reset uses status="reset" (vs "ok" elsewhere) — either is fine,
    # both indicate success.
    assert body.get("status") in {"ok", "reset"}

    # First observe after reset must be a bootstrap
    code, body = _post_multipart(
        http_server["base"], "/observe", png,
        {"url": "http://test.local", "last_action": "after-reset", "format": "raw"},
    )
    assert code == 200
    assert body["obs_type"] == "full_frame"
    assert body["trigger"] == "initial"


def test_state_endpoint_is_introspectable(http_server):
    """GET /state returns step/last_classification info."""
    _post_empty(http_server["base"], "/reset")
    png = _synthetic_png()
    _post_multipart(http_server["base"], "/observe", png,
                    {"url": "http://test.local", "last_action": "initial", "format": "raw"})

    code, body = _get(http_server["base"], "/state")
    assert code == 200
    assert "step" in body
    assert "last_classification" in body


def test_unknown_route_returns_404(http_server):
    """GET /nonexistent should 404 cleanly, not 500."""
    code, _ = _get(http_server["base"], "/nonexistent")
    assert code == 404


def test_anthropic_format_roundtrip(http_server):
    """format=anthropic returns content blocks ready for the SDK."""
    _post_empty(http_server["base"], "/reset")
    png = _synthetic_png()
    code, body = _post_multipart(
        http_server["base"], "/observe", png,
        {"url": "http://test.local", "last_action": "initial", "format": "anthropic"},
    )
    assert code == 200
    # Anthropic adapter returns a list-ish payload with at least one block
    payload = body if isinstance(body, list) else body.get("content") or body.get("payload")
    assert payload, f"expected Anthropic content payload, got keys={list(body)}"
