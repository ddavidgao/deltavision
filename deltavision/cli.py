"""
deltavision CLI — tiny dispatcher.

    deltavision selftest [--no-http]
        Staged E2E self-test. See deltavision.selftest for details.

    deltavision run ...
        Runs the agent loop (see main.cli_entry for the full arg list).
        This is the same entrypoint that v1.0.2 shipped as `deltavision`.

    deltavision --help
"""
from __future__ import annotations

import sys


def _usage() -> int:
    print(__doc__.strip())
    return 2


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        return _usage()

    cmd = argv[0]
    rest = argv[1:]

    if cmd == "selftest":
        from deltavision.selftest import main as selftest_main
        return selftest_main(rest)

    if cmd == "run":
        import main as _main_module
        sys.argv = [sys.argv[0]] + rest
        _main_module.cli_entry()
        return 0

    # Back-compat: v1.0.2/v1.0.3 `deltavision` was a direct alias for
    # `main.cli_entry`. If the user passes flags (`--task ...`) without a
    # subcommand, dispatch to `run` automatically so old scripts keep working.
    if cmd.startswith("-"):
        print(
            "deltavision: treating legacy `deltavision --...` as `deltavision run --...`. "
            "Future scripts should use `deltavision run`.",
            file=sys.stderr,
        )
        import main as _main_module
        sys.argv = [sys.argv[0]] + argv
        _main_module.cli_entry()
        return 0

    print(f"deltavision: unknown command: {cmd!r}\n", file=sys.stderr)
    return _usage()


if __name__ == "__main__":
    sys.exit(main())
