"""Google Drive vault — journal JSON in YOUR Drive, via a 2-minute Apps Script.

Streamlit Cloud's disk is not yours. Drive is. This module talks to the tiny
web app in google_sync/Code.gs: one GET lists every book, one POST writes one.
"""

from __future__ import annotations

import json
import os

import requests

from . import storage as sg


def vault_url(app_dir: str) -> str:
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


def pull_all(url: str, timeout: int = 25) -> dict:
    """{book_name: json_str}. Empty dict if nothing stored yet."""
    r = requests.get(_exec_url(url), timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "books" in data:
        out = {}
        for k, v in (data.get("books") or {}).items():
            out[str(k)] = v if isinstance(v, str) else json.dumps(v)
        return out
    if isinstance(data, dict) and data.get("name"):
        return {str(data["name"]): json.dumps(data)}
    return {}


def push_book(url: str, book_blob: dict, timeout: int = 25) -> dict:
    r = requests.post(
        _exec_url(url),
        json={"name": book_blob.get("name") or "book", "book": book_blob},
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
        body = text if isinstance(text, str) else json.dumps(text)
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        n += 1
    return n
