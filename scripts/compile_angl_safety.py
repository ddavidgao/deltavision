"""Compile the DeltaVision safety policy from its Angl chapter."""

import os
import sys
from pathlib import Path


def _load_angl():
    repo_root = Path(__file__).resolve().parents[1]
    sibling = repo_root.parent / "angl"
    angl_repo = Path(os.environ.get("ANGL_REPO", str(sibling)))
    if not angl_repo.exists():
        raise SystemExit(
            "Could not find the Angl repo next to this repo. Set ANGL_REPO "
            "and add it to PYTHONPATH before running this script."
        )
    sys.path.insert(0, str(angl_repo))


def main() -> int:
    _load_angl()

    from angl.parse import parse
    from angl.run import compile_until_green

    repo_root = Path(__file__).resolve().parents[1]
    spec_path = repo_root / "specs" / "evaluate_action_safety.angl"
    build_dir = repo_root / "angl_build" / "safety"
    spec = parse(spec_path.read_text())
    build, result, attempts = compile_until_green(
        spec,
        str(build_dir),
        units={spec["name"]: spec},
        max_attempts=3,
    )

    print(f"{spec['name']}: {result['passed']}/{result['total']} cases green")
    print(f"attempts: {attempts}")
    print(f"artifact: {build['artifact']}")
    print(f"shim: {build['shim']}")
    for case in result["results"]:
        if not case["pass"]:
            print(f"FAIL {case['case']} -- {case['detail']}")
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
