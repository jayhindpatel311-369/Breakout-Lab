"""
Where your data lives — and how to move it somewhere safer.

By default the journal and the saved parameter sets sit inside the app folder,
which is fine until the day you download a new version of the app. Then your
whole trading history is one careless folder-replace away from being gone.

So the data folder is a setting. Point it at a Google Drive or OneDrive folder
and three things happen at once: every fill is backed up as you record it, the
same book opens on another machine, and upgrading the app becomes "replace the
app folder" with nothing of yours inside it.

    <data folder>/
        journal/   your books, one JSON each
        params/    your saved sidebar presets
        backups/   dated snapshots

The setting itself has to live somewhere that does NOT move, or the app could
not find the folder it was told about — so `app_settings.json` stays in the app
directory. It holds a path and nothing else.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime

SETTINGS_FILE = "app_settings.json"
SUBFOLDERS = ("journal", "params", "backups")


# --------------------------------------------------------------------------- #
# the setting
# --------------------------------------------------------------------------- #
def settings_path(app_dir: str) -> str:
    return os.path.join(app_dir, SETTINGS_FILE)


def load_settings(app_dir: str) -> dict:
    path = settings_path(app_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            blob = json.load(f)
        return blob if isinstance(blob, dict) else {}
    except Exception:                                          # noqa: BLE001
        return {}


def save_settings(app_dir: str, settings: dict) -> str:
    path = settings_path(app_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    return path


def data_dir(app_dir: str) -> str:
    """The folder in use. Falls back to the app folder if the saved one is gone.

    A path that no longer exists — an unplugged drive, a renamed Drive folder —
    must not take the app down with it, and must not silently create a new empty
    folder somewhere unexpected either. So it falls back, and `data_note()` says
    that it did.
    """
    want = str(load_settings(app_dir).get("data_dir") or "").strip()
    if want and os.path.isdir(want):
        return want
    return app_dir


def data_note(app_dir: str) -> str:
    """"" when all is well; an explanation when the saved folder is missing."""
    want = str(load_settings(app_dir).get("data_dir") or "").strip()
    if want and not os.path.isdir(want):
        return (f"The data folder **{want}** is not there right now — an unplugged drive, "
                "or a folder that was renamed or is still syncing. The app folder is being "
                "used instead, so anything you save now will NOT go to that folder.")
    return ""


def set_data_dir(app_dir: str, new_dir: str) -> None:
    s = load_settings(app_dir)
    s["data_dir"] = os.path.abspath(os.path.expanduser(str(new_dir).strip()))
    save_settings(app_dir, s)


def use_app_folder(app_dir: str) -> None:
    s = load_settings(app_dir)
    s.pop("data_dir", None)
    save_settings(app_dir, s)


# --------------------------------------------------------------------------- #
# the folders
# --------------------------------------------------------------------------- #
def ensure(root: str) -> str:
    for sub in SUBFOLDERS:
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    return root


def journal_dir(app_dir: str) -> str:
    d = os.path.join(data_dir(app_dir), "journal")
    os.makedirs(d, exist_ok=True)
    return d


def params_dir(app_dir: str) -> str:
    d = os.path.join(data_dir(app_dir), "params")
    os.makedirs(d, exist_ok=True)
    return d


def backup_dir(app_dir: str) -> str:
    d = os.path.join(data_dir(app_dir), "backups")
    os.makedirs(d, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# moving in
# --------------------------------------------------------------------------- #
def check_target(path: str) -> tuple[bool, str]:
    """Is this somewhere we can actually write? (ok, message)"""
    path = os.path.abspath(os.path.expanduser(str(path).strip()))
    if not path:
        return False, "Give a folder path."
    if os.path.isfile(path):
        return False, "That is a file, not a folder."
    parent = path if os.path.isdir(path) else os.path.dirname(path)
    if not os.path.isdir(parent):
        return False, f"**{parent}** does not exist. Create it first, or pick another folder."
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".breakout_lab_write_test")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
    except Exception as exc:                                   # noqa: BLE001
        return False, f"Cannot write there: {exc}"
    return True, f"**{path}** is writable."


def copy_data(src_root: str, dst_root: str, overwrite: bool = False) -> tuple[int, list[str]]:
    """Copy journal and params from one root to another. (copied, skipped)

    Nothing is deleted and, unless you ask, nothing is overwritten — a file that
    already exists at the destination is reported and left alone. Moving your
    journal is not the moment to discover that "move" meant "replace".
    """
    copied, skipped = 0, []
    ensure(dst_root)
    for sub in ("journal", "params"):
        s = os.path.join(src_root, sub)
        if not os.path.isdir(s):
            continue
        d = os.path.join(dst_root, sub)
        os.makedirs(d, exist_ok=True)
        for fn in os.listdir(s):
            if not fn.endswith(".json"):
                continue
            target = os.path.join(d, fn)
            if os.path.exists(target) and not overwrite:
                skipped.append(f"{sub}/{fn}")
                continue
            shutil.copy2(os.path.join(s, fn), target)
            copied += 1
    return copied, skipped


# --------------------------------------------------------------------------- #
# backups
# --------------------------------------------------------------------------- #
def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M")


def write_backup(app_dir: str, name: str, files: dict[str, bytes]) -> str:
    """Write a dated set of files into backups/ and return the folder used.

    `files` is {filename: content}. Kept dumb on purpose: what a backup should
    contain is the journal tab's business, not this file's.
    """
    folder = os.path.join(backup_dir(app_dir), f"{name}_{stamp()}")
    os.makedirs(folder, exist_ok=True)
    for fn, blob in files.items():
        with open(os.path.join(folder, fn), "wb") as f:
            f.write(blob)
    return folder


def list_backups(app_dir: str, limit: int = 20) -> list[tuple[str, str]]:
    """(folder name, when) for the most recent backups, newest first."""
    root = backup_dir(app_dir)
    if not os.path.isdir(root):
        return []
    out = []
    for fn in sorted(os.listdir(root), reverse=True)[:limit]:
        p = os.path.join(root, fn)
        if os.path.isdir(p):
            out.append((fn, datetime.fromtimestamp(os.path.getmtime(p))
                        .strftime("%d %b %Y, %H:%M")))
    return out
