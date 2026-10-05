"""NSE source: quotes, retries, the close overlay — against a fake session."""
import os

import numpy as np
import pandas as pd
import pytest

from core import nse


class Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responder):
        self.responder, self.calls, self.headers = responder, [], {}

    def get(self, url, timeout=None, **kw):
        self.calls.append(url)
        return self.responder(url)


def _quote(last, prev, stamp="05-Oct-2026 11:42:10"):
    return {"priceInfo": {"lastPrice": last, "previousClose": prev},
            "metadata": {"lastUpdateTime": stamp}}


NOSLEEP = lambda s: None                                           # noqa: E731


def test_parse_quote():
    q = nse._parse_quote(_quote(1520.5, 1500.0))
    assert q["last"] == 1520.5 and q["prev_close"] == 1500.0
    assert q["session"] == pd.Timestamp("2026-10-05")
    assert nse._parse_quote({"priceInfo": {}}) is None


def test_live_quotes_ok_and_partial():
    def responder(url):
        if "INFY" in url:
            return Resp(200, _quote(1520.5, 1500.0))
        return Resp(404)
    sess = FakeSession(responder)
    df, rep = nse.live_quotes(["INFY", "NOPE"], sess=sess, sleep=NOSLEEP)
    assert df.loc["INFY", "last"] == 1520.5
    assert rep["ok"] == 1 and rep["failed"] == ["NOPE"] and not rep["blocked"]


def test_live_quotes_gives_up_when_blocked():
    sess = FakeSession(lambda url: Resp(403))
    df, rep = nse.live_quotes(["A", "B", "C", "D", "E", "F"], sess=sess, sleep=NOSLEEP,
                              give_up_after=3)
    assert df.empty and rep["blocked"]
    assert set(rep["failed"]) == set("ABCDEF")
    # stopped after 3 symbols (2 tries each + cookie re-primes), not all 6
    assert sum("quote-equity" in c for c in sess.calls) == 6


def test_get_retries_then_succeeds():
    seq = iter([Resp(429), Resp(503), Resp(200, text="ok")])
    sess = FakeSession(lambda url: next(seq))
    waits = []
    r = nse._get(sess, "https://x", tries=3, backoff=1.0, sleep=waits.append)
    assert r.status_code == 200 and waits == [1.0, 2.0]


def test_recent_404_is_not_cached_as_holiday(tmp_path, monkeypatch):
    from datetime import datetime
    monkeypatch.setattr(nse, "now_ist", lambda: datetime(2026, 10, 5, 17, 0))
    sess = FakeSession(lambda url: Resp(404))
    day = pd.Timestamp("2026-10-05")
    _, df, status = nse._fetch_day(sess, day, str(tmp_path))
    assert df is None and status == "not published yet"
    assert not os.path.exists(nse._day_path(day, str(tmp_path)))
    old = pd.Timestamp("2026-09-01")
    _, df, status = nse._fetch_day(sess, old, str(tmp_path))
    assert status == "holiday/404" and os.path.exists(nse._day_path(old, str(tmp_path)))


def test_wide_keeps_close():
    day = pd.Timestamp("2026-10-01")
    frames = {day: pd.DataFrame({"symbol": ["INFY", "TCS"], "close": [1500.0, 3000.0],
                                 "trades": [10.0, 20.0], "deliv_pct": [50.0, 60.0]})}
    w = nse._wide(frames, {"INFY"}, [day])
    assert w["close"].loc[day, "INFY"] == 1500.0
    assert w["deliv_pct"].loc[day, "INFY"] == 50.0


def test_overlay_close_keeps_adjustment_basis():
    idx = pd.to_datetime(["2026-09-30", "2026-10-01"])
    raw = pd.DataFrame({"A": [100.0, 102.0], "B": [50.0, 51.0]}, index=idx)
    adj = raw * pd.DataFrame({"A": [0.98, 1.0], "B": [1.0, 1.0]}, index=idx)  # past dividend
    nse_close = pd.DataFrame({"A": [100.5, np.nan]}, index=idx)
    panel, n = nse.overlay_close({"RawClose": raw, "Close": adj, "Open": raw}, nse_close)
    assert n == 1
    assert panel["RawClose"].loc[idx[0], "A"] == 100.5
    assert panel["Close"].loc[idx[0], "A"] == pytest.approx(100.5 * 0.98)
    assert panel["RawClose"].loc[idx[1], "A"] == 102.0                 # NSE gap: Yahoo kept
    assert panel["Close"]["B"].equals(adj["B"])
    assert nse.overlay_close({"RawClose": raw, "Close": adj}, pd.DataFrame())[1] == 0
