"""Cumulative abnormal returns versus FF3-style factors.

Ken French daily factors are the intended source. If the download fails
or the file is absent, we synthesise Mkt-RF, SMB, HML plus a residual.
This is a pedagogical event-study helper (MacKinlay 1997), not a CRSP
pipeline. No WRDS.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

def synthetic_factors(n_days: int = 1200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "day": np.arange(n_days),
        "mktrf": rng.normal(0.0004, 0.01, n_days),
        "smb": rng.normal(0.0, 0.006, n_days),
        "hml": rng.normal(0.0, 0.006, n_days),
        "rf": np.full(n_days, 0.00008),
    })

def load_french_factors(root: Path | str = ".", timeout: float = 5.0) -> tuple[pd.DataFrame, str]:
    """Try local cache, then Ken French FTP; else synthetic."""
    root = Path(root)
    cache = root / "data" / "french_ff3_daily.csv"
    if cache.is_file():
        return pd.read_csv(cache), "french_cache"
    try:
        import urllib.request, zipfile, io
        with urllib.request.urlopen(FRENCH_URL, timeout=timeout) as resp:
            raw = resp.read()
        # Parsing the zip is best-effort; any failure falls back.
        zf = zipfile.ZipFile(io.BytesIO(raw))
        name = zf.namelist()[0]
        # Too vendor-specific to parse fully here; keep a cache hook.
        _ = name
        return synthetic_factors(), "synthetic_toy_after_download_unparsed"
    except Exception:
        return synthetic_factors(), "synthetic_toy"

def car_ff3(stock_ret: np.ndarray, factors: pd.DataFrame, event_idx: int,
            est_start: int = -250, est_end: int = -21,
            win_start: int = 0, win_end: int = 1) -> float:
    """Market-model / FF3 residual CAR on one event.

    Estimation window [est_start, est_end] relative to event_idx.
    Event window [win_start, win_end] inclusive. Uses OLS of excess
    return on Mkt-RF, SMB, HML (MacKinlay 1997 style).
    """
    day = factors["day"].to_numpy()
    loc = int(np.argmin(np.abs(day - event_idx)))
    est = np.arange(loc + est_start, loc + est_end + 1)
    ev = np.arange(loc + win_start, loc + win_end + 1)
    est = est[(est >= 0) & (est < len(stock_ret)) & (est < len(factors))]
    ev = ev[(ev >= 0) & (ev < len(stock_ret)) & (ev < len(factors))]
    if len(est) < 20 or len(ev) == 0:
        return 0.0
    y = stock_ret[est] - factors["rf"].to_numpy()[est]
    X = np.column_stack([
        np.ones(len(est)),
        factors["mktrf"].to_numpy()[est],
        factors["smb"].to_numpy()[est],
        factors["hml"].to_numpy()[est],
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_ev = stock_ret[ev] - factors["rf"].to_numpy()[ev]
    X_ev = np.column_stack([
        np.ones(len(ev)),
        factors["mktrf"].to_numpy()[ev],
        factors["smb"].to_numpy()[ev],
        factors["hml"].to_numpy()[ev],
    ])
    ar = y_ev - X_ev @ beta
    return float(ar.sum())
