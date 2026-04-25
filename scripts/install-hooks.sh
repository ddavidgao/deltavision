#!/bin/bash
# Install repo-versioned git hooks into .git/hooks/.
#
# Why versioned, not pre-commit framework: zero external dependency,
# everything lives in this repo, runs in <100ms. The pre-commit framework
# is more standard but pulls in a Python deps tree just to call ruff.

set -e

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_SRC="$REPO_ROOT/scripts/hooks"
HOOKS_DST="$REPO_ROOT/.git/hooks"

mkdir -p "$HOOKS_DST"

for hook in "$HOOKS_SRC"/*; do
    name=$(basename "$hook")
    cp "$hook" "$HOOKS_DST/$name"
    chmod +x "$HOOKS_DST/$name"
    echo "installed: .git/hooks/$name"
done

echo ""
echo "Done. Hooks will run on every commit. Bypass with: git commit --no-verify"
