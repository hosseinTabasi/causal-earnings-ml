"""Synthetic-null placebo should not reject at 5%; OLS sign matches true ATE."""
from __future__ import annotations
import numpy as np
from events.calendar import synthetic_events
from dml.ate import ols_ate, dml_ate, placebo_ate, cate_by_bucket
from returns.car import synthetic_factors, car_ff3

def _xy(panel):
    df = panel.frame
    xcols = [c for c in df.columns if c.startswith("x")]
    y = df["car"].to_numpy()
    d = df[panel.treatment_col].to_numpy()
    x = df[xcols].to_numpy()
    return y, d, x

def test_placebo_does_not_reject_on_null():
    panel = synthetic_events(n=500, seed=0, true_ate=0.0)
    y, d, x = _xy(panel)
    r = placebo_ate(y, d, x, seed=0)
    assert r["p"] > 0.05, r

def test_dml_on_null_not_tiny_p():
    panel = synthetic_events(n=500, seed=1, true_ate=0.0)
    y, d, x = _xy(panel)
    r = dml_ate(y, d, x, seed=1)
    # Null DGP: do not require p>0.05 always (finite sample), but
    # |t| should not explode. Bound the t-stat.
    assert abs(r["t"]) < 4.0, r

def test_ols_sign_matches_true_ate():
    panel = synthetic_events(n=600, seed=2, true_ate=0.08)
    y, d, x = _xy(panel)
    r = ols_ate(y, d, x)
    assert r["ate"] > 0.0, r

def test_cate_buckets_run():
    panel = synthetic_events(n=400, seed=3, true_ate=0.05)
    df = panel.frame
    y, d, x = _xy(panel)
    out = cate_by_bucket(y, d, x, df["size_bucket"].to_numpy())
    assert "small" in out and "large" in out

def test_car_ff3_finite():
    fac = synthetic_factors(n_days=400, seed=0)
    rng = np.random.default_rng(0)
    ret = fac["mktrf"].to_numpy() + rng.normal(0, 0.01, 400)
    val = car_ff3(ret, fac, event_idx=200)
    assert np.isfinite(val)
