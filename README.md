# Double Machine Learning for Earnings-Announcement Effects

**Author:** Hossein Tabasi (2026). MIT licence. Junior research scaffold.

## Question

After double/debiased machine learning (DML) controls for a high-dimensional
confounder vector X, does earnings surprise still have an average effect on
post-announcement cumulative abnormal returns (CARs), and is that effect
heterogeneous by size and volatility? Treatment is SUE when a real event
file exists; otherwise a noisy public **proxy** labelled `sue_proxy`.
No WRDS.

## Why it matters

Ball and Brown (1968) documented that earnings news is associated with
contemporaneous returns. Bernard and Thomas (1989, 1990) documented
post-earnings-announcement drift (PEAD). Standard event studies
(MacKinlay 1997) residualise returns on a factor model and then regress
CAR on surprise. When X is high-dimensional (size, book-to-market,
turnover, lagged returns, industry), naive OLS is biased if those
controls are selected with the same sample. Chernozhukov et al. (2018)
DML is a Neyman-orthogonal, cross-fit estimator for this setting.
The question is identification of an earnings-surprise effect *after*
flexible controls, not a trading rule.

## Data

- Events: `data/events.csv` if present, else `synthetic_events` (TOY).
- Factors: Ken French daily FF3 (see `scripts/download_french.md`);
  synthetic factors if the download fails.
- Returns/CAR: FF3 residual CAR on a one- or two-day window after the
  event (MacKinlay). On the toy path, CAR is generated in the DGP.

## Method

1. OLS of CAR on treatment and X (baseline, likely biased).
2. Pedagogical DML: K-fold Ridge for E[Y|X] and E[D|X], then residual
   on residual. If `doubleml` is installed it is detected but the
   tested path remains the sklearn cross-fit so CI stays light.
3. CATE: OLS inside simulated size and vol buckets.
4. Placebo: permute treatment; on a null DGP the p-value should not
   reject at 5% (see tests).

## Baselines

OLS with linear X vs DML with Ridge nuisance. No lasso-logit treatment
model in the toy path (continuous proxy).

## Results

**NO FULL RESULTS YET.** There is no I/B/E/S SUE file and no CRSP CAR.
`results/tables/toy_dml.csv` is a **TOY** simulation: the DGP plants a
known `true_ate` on `sue_proxy`. Those numbers check the estimator, they
are not an earnings-announcement finding. Placebo p-values on the
synthetic null are a specification test, not evidence about PEAD.

## Limitations

`sue_proxy` is not SUE. Synthetic FF3 is not Ken French. Pedagogical DML
is not DoubleML.jl. Heterogeneity buckets are simulated. Do not cite TOY
rows as empirical CARs.

## Reproduce (toy)

PYTHONPATH=src python -m run_toy
PYTHONPATH=src python -m pytest -q

## References

Chernozhukov et al., Econometrics Journal 2018. Ball and Brown, JAR 1968.
Bernard and Thomas, JAR 1989 / JFE 1990. MacKinlay, JEL 1997.
