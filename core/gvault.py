"""Google Drive vault — journal JSON in YOUR Drive, via a 2-minute Apps Script.

Streamlit Cloud's disk is not yours. Drive is. This module talks to the tiny
web app in google_sync/Code.gs: one GET lists every book, one POST writes one.
"""

from __future__ import annotations

import json
import math
import os

import requests

from . import storage as sg


def vault_url(app_dir: str) -> str:
    """Disk settings first, then Streamlit secrets / env (Cloud reboot-proof)."""
    u = str(sg.load_settings(app_dir).get("vault_url") or "").strip()
    if u:
        return u
    try:
        import streamlit as st
        u = str(st.secrets.get("VAULT_URL") or st.secrets.get("vault_url") or "").strip()
        if u:
            return u
    except Exception:
        pass
    return str(os.environ.get("VAULT_URL") or "").strip()


def set_vault_url(app_dir: str, url: str) -> None:
    s = sg.load_settings(app_dir)
    s["vault_url"] = str(url or "").strip()
    sg.save_settings(app_dir, s)


def _exec_url(url: str) -> str:
    u = (url or "").strip()
    if u.endswith("/"):
        u = u[:-1]
    return u


def _json_safe(obj):
    """NaN / Inf are not JSON. Drive / Apps Script reject them."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    try:
        import numpy as np
        if isinstance(obj, np.floating):
            x = float(obj)
            return None if math.isnan(x) or math.isinf(x) else x
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
        if obj is np.nan:
            return None
    except Exception:
        pass
    return obj


def pull_all(url: str, timeout: int = 25) -> dict:
    """{book_name: json_str}. Empty dict if nothing stored yet."""
    r = requests.get(_exec_url(url), timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "books" in data:
        out = {}
        for k, v in (data.get("books") or {}).items():
            out[str(k)] = v if isinstance(v, str) else json.dumps(v, allow_nan=False)
        return out
    if isinstance(data, dict) and data.get("name"):
        return {str(data["name"]): json.dumps(data, allow_nan=False)}
    return {}


def push_book(url: str, book_blob: dict, timeout: int = 25) -> dict:
    payload = _json_safe({"name": book_blob.get("name") or "book", "book": book_blob})
    body = json.dumps(payload, allow_nan=False, default=str)
    r = requests.post(
        _exec_url(url),
        data=body.encode("utf-8"),
        timeout=timeout,
        allow_redirects=True,
        headers={"Content-Type": "application/json"},
    )
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"ok": True}


def write_pulled(journal_dir: str, books: dict[str, str]) -> int:
    """Write vault JSON files into the local journal folder. Returns count."""
    if not books:
        return 0
    os.makedirs(journal_dir, exist_ok=True)
    n = 0
    for name, text in books.items():
        safe = "".join(c for c in str(name) if c.isalnum() or c in " _-").strip().replace(" ", "_")
        path = os.path.join(journal_dir, f"{safe or 'book'}.json")
        body = text if isinstance(text, str) else json.dumps(text, allow_nan=False, default=str)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        n += 1
    return n
