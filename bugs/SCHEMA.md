# Bug-log schema

Source of truth: `bugs/log.jsonl` — one JSON object per line, append-only.

Render the human-readable index with `python bugs/render.py` → writes `BUGS.md`.

## Why JSONL, not markdown

Append-only means no merge conflicts when two sessions log a bug at once. Structured fields mean you can query (`jq 'select(.status=="open")' bugs/log.jsonl`), filter (`jq 'select(.tags[]=="classifier")'`), and regenerate the render any time the schema changes. The flat markdown render exists only for human skim and the GitHub project page.

## Required fields

| Field | Type | Notes |
|---|---|---|
| `id` | string | Stable slug, monotonic. Format: `BUG-NNNN`. Never reuse. |
| `title` | string | One-line imperative summary. <80 chars. |
| `discovered_at` | ISO date | `YYYY-MM-DD` |
| `discovered_in_commit` | string \| null | Short SHA of the commit where the bug was *introduced* (if known) or `null` |
| `discovered_during` | string | What were we doing when we hit it. Keeps narrative. |
| `severity` | enum | `blocker` \| `major` \| `minor` \| `cosmetic` |
| `status` | enum | `open` \| `fixed` \| `deferred` \| `wontfix` \| `cant_reproduce` |
| `tags` | array of strings | Free-form. Common: `classifier`, `proxy`, `video`, `packaging`, `ci`, `benchmark` |
| `symptom` | string | What goes wrong, observable. |
| `repro` | string | How to reproduce. Commands, file paths, expected vs actual. |
| `root_cause` | string \| null | Mechanism, not just location. `null` if status=`open` and unknown. |
| `fix` | object \| null | `{ "commit": "<sha>", "summary": "<one-line>" }` when status=`fixed` |
| `notes` | string \| null | Anything else worth preserving. |

## Adding a bug

```bash
python bugs/add.py \
  --title "Greedy bbox-merge runs before MAX_REGIONS truncation" \
  --severity major \
  --tags classifier,benchmark \
  --discovered-during "scripted-77 CI failure on v1.0.6 release" \
  --symptom "..." --repro "..." --root-cause "..."
```

The `add.py` helper enforces the schema, picks the next `BUG-NNNN`, and appends one line to `log.jsonl`. After that, `python bugs/render.py` updates `BUGS.md`.

## Marking fixed

```bash
python bugs/resolve.py BUG-0003 \
  --commit abc1234 \
  --summary "Run merger after MAX_REGIONS truncation"
```

Updates the entry in-place (rewrites the file) and re-renders.
