"""
Named parameter sets — save the whole sidebar, load it back later.

The sidebar is the system. Forty-odd controls decide what qualifies, how it is
ranked, how much you buy and where you get out, and re-typing them from memory
every Monday is how a live system quietly drifts away from the one you tested.

So the sidebar is saved as a plain dict of widget values — `{"p_fresh": 1,
"p_entries": 10, ...}` — one JSON file per named set in `params/`. Nothing is
interpreted here: this module does not know what a rung or a screen is, it only
stores what the app hands it and gives it back. That way a control added to the
sidebar next month needs no change in this file, and an OLD set that predates
that control still loads — the missing key simply falls back to the widget's
own default.

`_last.json` remembers which set you used most recently, which is what the app
loads on start.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

SCHEMA = 1
LAST_FILE = "_last.json"


# --------------------------------------------------------------------------- #
# paths
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    safe = "".join(c for c in str(name) if c.isalnum() or c in " _-").strip()
    return safe.replace(" ", "_") or "set"


def set_path(directory: str, name: str) -> str:
    return os.path.join(directory, f"{_slug(name)}.json")


# --------------------------------------------------------------------------- #
# read / write
# --------------------------------------------------------------------------- #
def save_set(directory: str, name: str, values: dict) -> str:
    """Write one named set. `values` is the raw widget dict; dates are stringified."""
    os.makedirs(directory, exist_ok=True)
    path = set_path(directory, name)
    blob = {
        "name": str(name),
        "schema": SCHEMA,
        "saved": datetime.now().isoformat(timespec="seconds"),
        "values": dict(values),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(blob, f, indent=2, default=str, sort_keys=True)
    set_last_used(directory, name)
    return path


def load_set(directory: str, name: str) -> dict:
    """The stored widget dict, or {} if the set is missing or unreadable.

    A set that cannot be parsed returns {} rather than raising: a corrupt
    preferences file should cost you your presets, not your app.
    """
    path = set_path(directory, name)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:
        return {}
    vals = blob.get("values")
    return dict(vals) if isinstance(vals, dict) else {}


def delete_set(directory: str, name: str) -> bool:
    path = set_path(directory, name)
    if not os.path.isfile(path):
        return False
    os.remove(path)
    if last_used(directory) == name:
        _write_last(directory, "")
    return True


def list_sets(directory: str) -> list[str]:
    """Saved set names, alphabetical. `_last.json` is bookkeeping, not a set."""
    if not os.path.isdir(directory):
        return []
    out = []
    for fn in os.listdir(directory):
        if not fn.endswith(".json") or fn == LAST_FILE:
            continue
        try:
            with open(os.path.join(directory, fn), encoding="utf-8") as f:
                blob = json.load(f)
            out.append(str(blob.get("name") or os.path.splitext(fn)[0]))
        except Exception:
            out.append(os.path.splitext(fn)[0])
    return sorted(out, key=str.lower)


def describe(directory: str, name: str) -> str:
    """"saved 2026-09-05 18:40, 47 settings" — for the caption under the picker."""
    path = set_path(directory, name)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
    except Exception:
        return ""
    when = str(blob.get("saved", "")).replace("T", " ")
    n = len(blob.get("values") or {})
    return f"saved {when} · {n} settings" if when else f"{n} settings"


# --------------------------------------------------------------------------- #
# which one was used last
# --------------------------------------------------------------------------- #
def _write_last(directory: str, name: str) -> None:
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, LAST_FILE), "w", encoding="utf-8") as f:
        json.dump({"name": str(name)}, f, indent=2)


def set_last_used(directory: str, name: str) -> None:
    _write_last(directory, name)


def last_used(directory: str) -> str:
    """Name of the set to load on start — "" when there is none, or it is gone."""
    path = os.path.join(directory, LAST_FILE)
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as f:
            name = str(json.load(f).get("name") or "")
    except Exception:
        return ""
    return name if name and os.path.isfile(set_path(directory, name)) else ""


def load_last(directory: str) -> tuple[str, dict]:
    """(name, values) of the most recently used set. ("", {}) when there is none."""
    name = last_used(directory)
    return (name, load_set(directory, name)) if name else ("", {})
