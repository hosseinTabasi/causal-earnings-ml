"""Event calendar: CSV if present, otherwise a labelled synthetic panel.

Treatment is standardised unexpected earnings (SUE) when a real file exists.
If we synthesise, the column is a noisy public PROXY and is named
`sue_proxy` so it cannot be mistaken for I/B/E/S SUE. No WRDS.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

@dataclass
class EventPanel:
    frame: pd.DataFrame
    source: str  # "csv" or "synthetic_toy"
    treatment_col: str  # "sue" or "sue_proxy"

def synthetic_events(n: int = 400, n_firms: int = 40, seed: int = 0,
                     true_ate: float = 0.0) -> EventPanel:
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, n_firms, size=n)
    # Event dates in a 4-year window, business-day-ish integers.
    day = rng.integers(0, 1000, size=n)
    size = rng.lognormal(mean=1.0, sigma=0.8, size=n)
    vol = rng.lognormal(mean=-1.0, sigma=0.5, size=n)
    # High-dimensional X: lagged return, bm, turnover, plus noise.
    x = rng.normal(size=(n, 8))
    x[:, 0] = np.log(size) - np.log(size).mean()
    x[:, 1] = np.log(vol) - np.log(vol).mean()
    sue_proxy = rng.normal(size=n)
    # Confounding: size and vol shift both treatment and outcome.
    sue_proxy = sue_proxy + 0.3 * x[:, 0] - 0.2 * x[:, 1]
    # Potential outcome: CAR depends on X and optional true ATE.
    car = 0.4 * x[:, 0] - 0.3 * x[:, 1] + 0.1 * x[:, 2] + rng.normal(0, 0.05, size=n)
    car = car + true_ate * sue_proxy
    size_bucket = pd.qcut(size, 3, labels=["small", "mid", "large"])
    vol_bucket = pd.qcut(vol, 3, labels=["low", "mid", "high"])
    df = pd.DataFrame({
        "firm": firm, "event_day": day, "sue_proxy": sue_proxy,
        "car": car, "size": size, "vol": vol,
        "size_bucket": size_bucket.astype(str),
        "vol_bucket": vol_bucket.astype(str),
    })
    for j in range(x.shape[1]):
        df[f"x{j}"] = x[:, j]
    return EventPanel(frame=df, source="synthetic_toy", treatment_col="sue_proxy")

def load_events(root: Path | str = ".", seed: int = 0, true_ate: float = 0.0,
                n: int = 400) -> EventPanel:
    root = Path(root)
    csv_path = root / "data" / "events.csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        tcol = "sue" if "sue" in df.columns else "sue_proxy"
        return EventPanel(frame=df, source="csv", treatment_col=tcol)
    return synthetic_events(n=n, seed=seed, true_ate=true_ate)
