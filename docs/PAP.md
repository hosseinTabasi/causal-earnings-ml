# Pre-analysis plan: DML for earnings-announcement effects

Hossein Tabasi, 2026. This PAP is registered in-repo *before* any real
event file is used. Toy simulations may run; they do not update the plan.

## 1. Question and estimand

Primary estimand: the average treatment effect of earnings surprise D on
post-announcement CAR Y, after controlling for a high-dimensional X,
under unconfoundedness Y(d) ⊥ D | X. D is SUE (actual minus consensus,
scaled by price or by the time-series SD of forecast errors) when those
fields exist. Until they exist, D is a public proxy and all output is
tagged TOY / `sue_proxy`. We do not claim that the proxy is SUE.

Secondary: heterogeneity of the effect by size tercile and by trailing
volatility tercile (CATE). These are pre-specified slices, not data-mined
interactions.

## 2. Sample inclusion (to be applied when data exist)

Universe: US common stocks with an earnings announcement date, a
non-missing surprise measure, and sufficient estimation-window returns
to fit FF3 (at least 20 days in [-250, -21]). Exclude dual-class
duplicates by keeping the primary permno. Winsorise CAR and SUE at
1%/99% within calendar year. No look-ahead: X is lagged relative to the
announcement date. No WRDS dump is committed to this repository.

## 3. Outcome

Y = CAR[0, +1] from a FF3 market model estimated on [-250, -21] using
Ken French daily factors (MacKinlay 1997). Alternative windows [0, 0]
and [+2, +60] (PEAD) are robustness, not primary. We will not convert
CAR into a long-short backtest in this paper.

## 4. Treatment

Primary: SUE. Fallback: `sue_proxy`. We will not mix the two in one
table without a column that names the variable. If only a binary beat
dummy is available, that is a different estimand and requires an
addendum to this PAP.

## 5. Confounders X

Pre-specified: log size, book-to-market, trailing 63-day volatility,
lagged 21-day return, share turnover, an earnings-yield proxy if public,
and industry dummies (Fama-French 12) when industry codes exist. On the
toy path X is eight Gaussian columns with size and vol in the first two.
No post-announcement variables in X.

## 6. Estimators (frozen)

- OLS: Y on 1, D, X. Reported as a biased-in-general baseline.
- DML: cross-fit residual-on-residual (Chernozhukov et al. 2018).
  Nuisance models: Ridge in the toy path; when real data exist, we will
  use cross-validated Ridge and, if `doubleml` installs, compare to
  DoubleMLIRM/PLR. The sklearn path is pedagogical DML.
- Standard errors: homoskedastic OLS on the second stage for the toy
  test; cluster by announcement date on real data (addendum if we
  switch to firm-level cluster).
- CATE: OLS inside size terciles and vol terciles, same X minus the
  slicing variable.
- Placebo: permute D, or shift event dates by a uniform draw from
  {+30,...,+80} trading days. The 5% rejection rate under the synthetic
  null should be near size; we test that the seeded placebo p-value
  exceeds 0.05.

## 7. Decision rules

We will report DML ATE with SE and p. We will **not** drop DML if it is
insignificant and keep OLS if it is significant. We will not hunt over
CAR windows after seeing estimates. If DML and OLS disagree in sign on
real data, we discuss confounding, we do not pick the nicer sign.

## 8. What would falsify the story

A significant placebo on the synthetic null after we fix the seed in
tests (that is a code bug, not economics). On real data: a DML ATE that
vanishes once X includes lagged return and size, which would mean the
earnings surprise is not adding information beyond characteristics.
That is an acceptable negative result.

## 9. What this PAP does not authorise

PnL, PEAD trading-rule Sharpe, or any claim that DML "beats the market".
Use of non-public I/B/E/S extracts committed to git. Changing the
primary window to [+2,+60] after seeing [0,+1] is weak. Adding extra
buckets beyond size and vol without an addendum is weak.

## 10. Software freeze

`src/dml/ate.py`, `src/returns/car.py`, `src/events/calendar.py`,
`configs/default.yaml`. Toy config may use a planted `true_ate` for
tests only. Seeds in tests are part of the freeze.
