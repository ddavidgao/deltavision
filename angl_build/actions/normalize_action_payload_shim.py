import json, os, subprocess, sys

req = json.loads(sys.stdin.read() or "{}")
build_dir = os.path.dirname(__file__)
payload = json.dumps({"args": req.get("args", [])})
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
if proc.stdout.strip():
    print(proc.stdout.strip().splitlines()[-1])
else:
    error = proc.stderr.strip() or 'node' + f" exited with {proc.returncode}"
    print(json.dumps({"ok": False, "error": error}))
