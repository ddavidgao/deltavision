"""
Benchmark site registry.

Each site tests different DeltaVision capabilities. The registry defines
what to test, what the expected transition patterns are, and what metrics
matter. Add sites as we discover interesting edge cases.

Difficulty tiers:
  easy   — standard HTML, URL changes on nav, DOM-accessible text
  medium — SPA, iframes, dynamic content, URL doesn't change
  hard   — canvas rendering, custom widgets, heavy JS, anti-bot
"""

SITES = {
    # ── Educational (David's primary use case) ──────────────────────────

    "mcgrawhill": {
        "name": "McGraw-Hill Connect SmartBook",
        "url": "https://connect.mheducation.com",
        "difficulty": "hard",
        "auth_required": True,
        "tests": [
            "Question transitions (same URL, full content swap)",
            "Fill-in-the-blank → type → submit cycle",
            "Multiple choice → select → feedback cycle",
            "Question ↔ Reading mode toggle",
            "Confidence button interaction",
        ],
        "deltavision_challenges": [
            "URL never changes (SPA) — relies on diff_ratio + pHash",
            "Nav bar persists perfectly (anchor score 1.0)",
            "White-dominated pages keep diff_ratio low even on full swaps",
            "pHash is the primary trigger (not diff_ratio)",
        ],
        "key_metrics": ["questions_per_minute", "token_cost_per_question", "delta_ratio"],
    },

    "brightspace": {
        "name": "Purdue Brightspace (D2L)",
        "url": "https://purdue.brightspace.com",
        "difficulty": "medium",
        "auth_required": True,
        "tests": [
            "Navigate course content",
            "Open/close content modules",
            "Quiz interactions",
            "Grade checking",
        ],
        "deltavision_challenges": [
            "Complex iframe nesting for embedded content",
            "Mix of SPA and traditional navigation",
        ],
        "key_metrics": ["navigation_accuracy", "steps_to_complete"],
    },

    # ── Reaction / Speed benchmarks ─────────────────────────────────────

    "humanbenchmark_reaction": {
        "name": "Human Benchmark - Reaction Time",
        "url": "https://humanbenchmark.com/tests/reactiontime",
        "difficulty": "easy",
        "auth_required": False,
        "tests": ["Red → green reaction", "5-round consistency"],
        "deltavision_challenges": [
            "Pure speed test — screenshot capture is the bottleneck",
            "Model-free: CV pipeline alone should handle this",
        ],
        "key_metrics": ["reaction_time_ms", "detection_to_click_ms", "capture_latency_ms"],
        "baseline": {"claude_cu_ms": 13491, "human_median_ms": 273, "deltavision_best_ms": 412},
    },

    "humanbenchmark_aim": {
        "name": "Human Benchmark - Aim Trainer",
        "url": "https://humanbenchmark.com/tests/aim",
        "difficulty": "medium",
        "auth_required": False,
        "tests": ["Click random targets as they appear"],
        "deltavision_challenges": [
            "Small target detection via bbox extraction",
            "Target position = bbox center = click coordinates",
            "Could be model-free: diff → find new bbox → click center",
        ],
        "key_metrics": ["avg_time_per_target_ms", "accuracy"],
    },

    # ── Complex web apps (SPA stress tests) ─────────────────────────────

    "google_docs": {
        "name": "Google Docs",
        "url": "https://docs.google.com",
        "difficulty": "hard",
        "auth_required": True,
        "tests": [
            "Type text and verify rendering",
            "Formatting toolbar interactions",
            "Scroll through document",
        ],
        "deltavision_challenges": [
            "Canvas-rendered text (not DOM)",
            "Cursor blink creates constant micro-diffs",
            "Collaborative editing = unpredictable changes",
        ],
        "key_metrics": ["typing_accuracy", "false_positive_rate"],
    },

    "github": {
        "name": "GitHub",
        "url": "https://github.com",
        "difficulty": "medium",
        "auth_required": True,
        "tests": [
            "Navigate repos, open files",
            "Create/view issues and PRs",
            "Code review interactions",
        ],
        "deltavision_challenges": [
            "Turbo/HTMX partial page updates",
            "Markdown rendering in issues",
            "Diff views in PRs",
        ],
        "key_metrics": ["navigation_accuracy", "steps_to_complete"],
    },

    # ── Static / baseline (easiest, for sanity checking) ────────────────

    "wikipedia": {
        "name": "Wikipedia",
        "url": "https://en.wikipedia.org",
        "difficulty": "easy",
        "auth_required": False,
        "tests": [
            "Search for article",
            "Navigate between articles",
            "Extract information from tables",
        ],
        "deltavision_challenges": [
            "Standard HTML, URL changes — should be trivial",
            "Good baseline to verify classifier isn't over-triggering",
        ],
        "key_metrics": ["delta_ratio", "false_new_page_rate"],
    },

    # ── E-commerce (complex dynamic content) ────────────────────────────

    "amazon": {
        "name": "Amazon",
        "url": "https://www.amazon.com",
        "difficulty": "hard",
        "auth_required": False,
        "tests": [
            "Search for products",
            "Filter results",
            "Navigate product pages",
            "Add to cart flow",
        ],
        "deltavision_challenges": [
            "Heavy ad injection = unpredictable diff regions",
            "Lazy-loaded images change the page after initial render",
            "A/B tested UI = different layouts between sessions",
            "Anti-bot detection",
        ],
        "key_metrics": ["navigation_accuracy", "false_positive_rate"],
    },
}


def get_site(name: str) -> dict:
    """Look up a benchmark site by name."""
    if name in SITES:
        return SITES[name]
    # Fuzzy match
    for key, site in SITES.items():
        if name.lower() in key.lower() or name.lower() in site["name"].lower():
            return site
    raise KeyError(f"Unknown site: {name}. Available: {list(SITES.keys())}")


def list_sites(difficulty: str = None, auth: bool = None) -> list:
    """List sites, optionally filtered."""
    results = []
    for key, site in SITES.items():
        if difficulty and site["difficulty"] != difficulty:
            continue
        if auth is not None and site["auth_required"] != auth:
            continue
        results.append({"key": key, **site})
    return results
