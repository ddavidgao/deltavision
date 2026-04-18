"""
DOM-level page state extraction: clickable elements + focus.

Why this exists
---------------
Two gaps the pure-CV pipeline can't fill:

1. **Small-UI targeting.** A 320x225 thumbnail shows a 20 px TodoMVC checkbox
   as 5 px — essentially invisible. The agent then guesses coordinates and
   gets them wrong. Fix: enumerate clickable DOM elements with exact bboxes.

2. **Focus state.** Clicking an input field focuses it, but the cursor
   blinker is ~1 px wide — below the diff threshold. The CV pipeline
   concludes `action_had_effect=False`, the agent thinks its click failed
   and tries something else. The DOM knows what's focused.

Both are extracted in a single `page.evaluate()` to avoid round-trips.
Kept in `vision/` alongside the rest of the page-state primitives, but
imports Playwright-style `page` so it's a browser-coupled module.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


_SELECTORS = ", ".join([
    "a[href]",
    "button",
    "input",
    "textarea",
    "select",
    "[role='button']",
    "[role='link']",
    "[role='checkbox']",
    "[role='menuitem']",
    "[role='tab']",
    "[role='option']",
    "[onclick]",
    "[tabindex]:not([tabindex='-1'])",
    "summary",
    "label",
])


# JS evaluated in the page: returns {elements: [...], focus: {...} | null}.
_EXTRACT_JS = r"""
([selectorsCsv, maxElements]) => {
  const out = { elements: [], focus: null };
  const seen = new WeakSet();
  const vw = window.innerWidth, vh = window.innerHeight;

  const labelOf = (el) => {
    let label = el.getAttribute('aria-label') || '';
    if (!label) {
      const llby = el.getAttribute('aria-labelledby');
      if (llby) {
        const lbl = document.getElementById(llby);
        if (lbl) label = (lbl.innerText || lbl.textContent || '').trim();
      }
    }
    if (!label) label = (el.innerText || el.textContent || '').trim();
    if (!label) label = el.getAttribute('placeholder') || '';
    if (!label) label = el.getAttribute('value') || '';
    if (!label) label = el.getAttribute('name') || '';
    if (!label) label = el.getAttribute('title') || '';
    return label.replace(/\s+/g, ' ').trim().slice(0, 80);
  };

  const nodes = document.querySelectorAll(selectorsCsv);
  for (const el of nodes) {
    if (seen.has(el)) continue;
    seen.add(el);
    const r = el.getBoundingClientRect();
    if (r.width < 4 || r.height < 4) continue;
    if (r.right < 0 || r.left > vw || r.bottom < 0 || r.top > vh) continue;
    const cs = getComputedStyle(el);
    if (cs.visibility === 'hidden' || cs.display === 'none') continue;
    if (parseFloat(cs.opacity) < 0.05) continue;
    if (el.disabled) continue;

    const tag = el.tagName.toLowerCase();
    if (tag === 'label' && el.querySelector('input, button, a')) continue;

    const role = el.getAttribute('role') || '';
    const inputType = tag === 'input' ? (el.getAttribute('type') || 'text') : '';

    out.elements.push({
      bbox: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
      center: [Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2)],
      tag, role, label: labelOf(el), type: inputType,
    });
    if (out.elements.length >= maxElements) break;
  }

  // Focus: what's the active element? If it's an input/textarea, also
  // grab its current value (the text the agent has already typed).
  const af = document.activeElement;
  if (af && af !== document.body && af.tagName) {
    const r = af.getBoundingClientRect();
    const tag = af.tagName.toLowerCase();
    out.focus = {
      tag,
      role: af.getAttribute('role') || '',
      type: tag === 'input' ? (af.getAttribute('type') || 'text') : '',
      label: labelOf(af),
      bbox: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
      center: [Math.round(r.left + r.width / 2), Math.round(r.top + r.height / 2)],
      value: (tag === 'input' || tag === 'textarea') ? (af.value || '').slice(0, 160) : '',
      selection_start: (tag === 'input' || tag === 'textarea') ? (af.selectionStart ?? 0) : null,
    };
  }

  return out;
}
"""


async def extract_page_state(page, max_elements: int = 40) -> dict:
    """
    Return {elements: [...], focus: {...} | None}.

    elements: list of clickable elements with bbox/center/label
    focus:    None if no focused element (or body); else a dict describing
              the focused element and (for inputs) its current value.

    Returns {"elements": [], "focus": None} on error.
    """
    try:
        return await page.evaluate(_EXTRACT_JS, [_SELECTORS, max_elements]) or {"elements": [], "focus": None}
    except Exception as e:
        log.debug("extract_page_state failed: %s", e)
        return {"elements": [], "focus": None}


# Backwards-compat wrapper: some code still imports extract_clickables.
async def extract_clickables(page, max_elements: int = 40) -> list[dict]:
    state = await extract_page_state(page, max_elements)
    return state.get("elements", [])


def format_page_state_for_prompt(state: dict, max_chars: int = 2200) -> str:
    """
    Render the DOM state (clickables + focus) as a compact prompt block.
    """
    if not state:
        return ""
    lines = []

    focus = state.get("focus")
    if focus:
        cx, cy = focus.get("center", [0, 0])
        kind = focus.get("role") or focus.get("tag", "")
        if focus.get("type"):
            kind = f"{kind}/{focus['type']}"
        label = focus.get("label") or ""
        value = focus.get("value") or ""
        parts = [f"[FOCUS] {kind} at ({cx}, {cy})"]
        if label:
            parts.append(f'labeled "{label}"')
        if value:
            parts.append(f'current value: "{value}"')
        lines.append("Currently focused element (ground truth from DOM):")
        lines.append("  " + " · ".join(parts))
        lines.append("  → typing right now will enter this field; no need to click it first.")

    elems = state.get("elements") or []
    if elems:
        if lines:
            lines.append("")
        lines.append("Clickable elements on this page (center coords, click these directly):")
        total = sum(len(l) + 1 for l in lines)
        for e in elems:
            kind = e.get("role") or e.get("tag", "")
            if e.get("type"):
                kind = f"{kind}/{e['type']}"
            cx, cy = e.get("center", [0, 0])
            label = e.get("label") or ""
            label_snippet = f' — "{label}"' if label else ""
            line = f"  [{kind}] at ({cx}, {cy}){label_snippet}"
            if total + len(line) + 1 > max_chars:
                lines.append(f"  ... ({len(elems) - (len(lines) - (2 if focus else 1))} more, truncated)")
                break
            lines.append(line)
            total += len(line) + 1

    return "\n".join(lines)


# Backwards-compat: some code may still call format_clickables_for_prompt.
def format_clickables_for_prompt(elements: list[dict], max_chars: int = 2000) -> str:
    return format_page_state_for_prompt({"elements": elements, "focus": None}, max_chars)
