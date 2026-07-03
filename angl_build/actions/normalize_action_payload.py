import json
import os
import subprocess


def normalize_action_payload(*args):
    build_dir = os.path.dirname(__file__)
    payload = json.dumps({"args": list(args)})
    env = os.environ.copy()
    artifact = os.path.join(build_dir, 'normalize_action_payload.js')
    cmd = ['node', artifact]
    proc = subprocess.run(
        cmd,
        input=payload,
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    if not proc.stdout.strip():
        raise RuntimeError(proc.stderr.strip() or 'node' + f" exited with {proc.returncode}")
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    if out.get("ok"):
        return out.get("value")
    raise RuntimeError(out.get("error") or "unknown node artifact error")
