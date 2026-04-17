"""
Action definitions and browser executor.
All actions are typed — no free-form strings.
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(Enum):
    CLICK = "click"
    TYPE = "type"
    SCROLL = "scroll"
    KEY = "key"
    WAIT = "wait"
    DONE = "done"


@dataclass
class Action:
    type: ActionType
    x: Optional[int] = None
    y: Optional[int] = None
    text: Optional[str] = None
    direction: Optional[str] = None
    amount: Optional[int] = None
    key: Optional[str] = None
    duration_ms: Optional[int] = None

    def __str__(self):
        match self.type:
            case ActionType.CLICK:
                return f"click({self.x}, {self.y})"
            case ActionType.TYPE:
                return f"type('{self.text}')"
            case ActionType.SCROLL:
                return f"scroll({self.direction}, {self.amount}px)"
            case ActionType.KEY:
                return f"key({self.key})"
            case ActionType.WAIT:
                return f"wait({self.duration_ms}ms)"
            case ActionType.DONE:
                return "done"
        return f"unknown({self.type})"


def parse_action(action_dict: Optional[dict]) -> Optional[Action]:
    """Parse model JSON output into a typed Action.

    Supports two formats:
    1. DeltaVision native: {"type": "click", "x": 100, "y": 200}
    2. UI-TARS / CogAgent: {"action": "left_click", "coordinate": [100, 200]}
    """
    if not action_dict:
        return None
    if isinstance(action_dict, str):
        return None
    try:
        # UI-TARS / CogAgent format: {"action": "left_click", "coordinate": [x, y]}
        if "action" in action_dict and "type" not in action_dict:
            raw_action = action_dict["action"]
            coord = action_dict.get("coordinate", [])

            # Map UI-TARS action names to our ActionType
            action_map = {
                "left_click": ActionType.CLICK,
                "click": ActionType.CLICK,
                "right_click": ActionType.CLICK,
                "double_click": ActionType.CLICK,
                "type": ActionType.TYPE,
                "scroll": ActionType.SCROLL,
                "key": ActionType.KEY,
                "press": ActionType.KEY,
                "wait": ActionType.WAIT,
                "finished": ActionType.DONE,
                "done": ActionType.DONE,
            }
            atype = action_map.get(raw_action.lower())
            if atype is None:
                return None

            return Action(
                type=atype,
                x=int(coord[0]) if len(coord) > 0 else None,
                y=int(coord[1]) if len(coord) > 1 else None,
                text=action_dict.get("text"),
                direction=action_dict.get("direction"),
                amount=action_dict.get("amount"),
                key=action_dict.get("key"),
            )

        # DeltaVision native format.
        # Coerce numeric fields — some local VLMs (MAI-UI-8B observed) emit
        # coordinates as strings like "551" instead of ints.
        def _to_int(v):
            if v is None:
                return None
            try:
                return int(v)
            except (ValueError, TypeError):
                return None

        return Action(
            type=ActionType(action_dict["type"]),
            x=_to_int(action_dict.get("x")),
            y=_to_int(action_dict.get("y")),
            text=action_dict.get("text"),
            direction=action_dict.get("direction"),
            amount=_to_int(action_dict.get("amount")),
            key=action_dict.get("key"),
            duration_ms=_to_int(action_dict.get("duration_ms")),
        )
    except (KeyError, ValueError, TypeError, IndexError):
        return None


async def execute_action(action: Action, page, config):
    """Execute action in Playwright browser page."""
    match action.type:
        case ActionType.CLICK:
            # Robust click: find DOM element at coordinates, walk to nearest
            # interactive element, dispatch full event sequence.
            # Falls back to Playwright mouse.click if JS path fails.
            try:
                result = await page.evaluate("""([x, y]) => {
                    const raw = document.elementFromPoint(x, y);
                    if (!raw) return 'miss';
                    // Walk up to nearest interactive element
                    const interactive = raw.closest(
                        'input, textarea, select, button, a, [role="button"], ' +
                        '[role="textbox"], [role="link"], [contenteditable="true"], ' +
                        '[tabindex], label'
                    );
                    const el = interactive || raw;
                    // Full event sequence (pointer + mouse + focus + click)
                    const rect = el.getBoundingClientRect();
                    const cx = rect.left + rect.width / 2;
                    const cy = rect.top + rect.height / 2;
                    const opts = {bubbles: true, cancelable: true, clientX: cx, clientY: cy, view: window};
                    el.dispatchEvent(new PointerEvent('pointerdown', opts));
                    el.dispatchEvent(new MouseEvent('mousedown', opts));
                    el.focus();
                    el.dispatchEvent(new PointerEvent('pointerup', opts));
                    el.dispatchEvent(new MouseEvent('mouseup', opts));
                    el.dispatchEvent(new MouseEvent('click', opts));
                    return el.tagName;
                }""", [action.x, action.y])
                # Verify: if targeting an input, check it got focus
                if result in ("INPUT", "TEXTAREA"):
                    import asyncio as _aio
                    await _aio.sleep(0.05)
                    focused = await page.evaluate("document.activeElement?.tagName")
                    if focused not in ("INPUT", "TEXTAREA"):
                        await page.mouse.click(action.x, action.y)
            except Exception:
                await page.mouse.click(action.x, action.y)

        case ActionType.TYPE:
            await page.keyboard.type(action.text, delay=30)

        case ActionType.SCROLL:
            dx, dy = 0, 0
            amt = action.amount or 300
            match action.direction:
                case "down":
                    dy = amt
                case "up":
                    dy = -amt
                case "right":
                    dx = amt
                case "left":
                    dx = -amt
            await page.mouse.wheel(dx, dy)

        case ActionType.KEY:
            # Normalize common key names — models often output lowercase
            key_map = {
                "enter": "Enter", "return": "Enter",
                "tab": "Tab", "escape": "Escape", "esc": "Escape",
                "backspace": "Backspace", "delete": "Delete",
                "arrowup": "ArrowUp", "arrowdown": "ArrowDown",
                "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
                # Models often emit bare directional names for arrow keys
                "up": "ArrowUp", "down": "ArrowDown",
                "left": "ArrowLeft", "right": "ArrowRight",
                "space": " ",
                "home": "Home", "end": "End",
                "pageup": "PageUp", "pagedown": "PageDown",
                # Bare modifier keys — pressing alone is nearly always wrong,
                # but at least don't crash: map to valid Playwright names.
                "ctrl": "Control", "control": "Control",
                "cmd": "Meta", "meta": "Meta",
                "shift": "Shift", "alt": "Alt",
            }
            raw = action.key if action.key else "Enter"
            # Normalize modifier keys: ctrl -> Control, cmd -> Meta, alt -> Alt
            raw = raw.replace("ctrl+", "Control+").replace("cmd+", "Meta+").replace("alt+", "Alt+").replace("shift+", "Shift+")
            key = key_map.get(raw.lower(), raw)
            await page.keyboard.press(key)

        case ActionType.WAIT:
            await asyncio.sleep((action.duration_ms or 1000) / 1000)
