# Workshop outline: Double ML for earnings-announcement effects

Hossein Tabasi, 2026. Methods outline. **No real earnings file has been
used. No WRDS. All runnable numbers are TOY simulations.**

## 1. Motivation

The association between earnings news and equity returns is one of the
oldest results in empirical accounting (Ball and Brown 1968). The
subsequent literature split into two programmes that are often conflated.
The first is a measurement programme: define surprise, define an abnormal
return, and document a contemporaneous move. The second is a drift
programme: Bernard and Thomas (1989, 1990) showed that signed surprise
predicts returns for weeks after the announcement. Both programmes
typically residualise returns on a factor model (MacKinlay 1997) and then
run a low-dimensional regression of CAR on SUE plus a handful of
characteristics.

That last step is where identification is least discussed. Size, book-to-
market, turnover and lagged returns are high-dimensional once industry
dummies and nonlinearities are allowed, and they are correlated with both
the coverage process that produces consensus forecasts and with expected
returns. Selecting those controls with the same sample that estimates the
surprise coefficient is the textbook setting for regularisation bias.
Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey and Robins
(Econometrics Journal, 2018) provide a Neyman-orthogonal, cross-fit
estimator — double/debiased machine learning — that is designed for
exactly this problem: a low-dimensional treatment of interest and a
high-dimensional nuisance function.

The question this project asks is therefore not "does PEAD still exist
in 2026" and not "can we trade earnings". It is: after DML controls
for a pre-specified X, does surprise still have an average effect on
post-announcement CARs, and is that effect heterogeneous by size and
volatility? Until a real event file exists, the question is answered only
on a simulated DGP whose true ATE we planted.

## 2. Estimand

We want theta = E[Y(1) - Y(0)] for a continuous treatment under a
partially linear model Y = theta D + g(X) + U, D = m(X) + V, with
E[U|X,D]=0 in the toy DGP by construction. On real data the same
equation is an assumption: unconfoundedness given X, overlap, and
SUTVA across firms on a given day (we will cluster by date). D is SUE
when available and `sue_proxy` otherwise. The proxy is named so that a
reader cannot miss the substitution. CATE is E[Y(1)-Y(0)|bucket] for
pre-specified size and vol terciles.

## 3. Related work we actually rely on

Ball and Brown (1968) is the existence result. Bernard and Thomas is
the drift result; we cite it to locate the project in accounting, not
because we will estimate [+2,+60] as primary. MacKinlay (1997) is the
event-study algebra for CAR. Fama and French (1993) supply the three
factors we residualise on; Ken French's data library is the public
source. Chernozhukov et al. (2018) is the DML theorem. We do not cite
Kaggle earnings notebooks. We do not treat a sklearn Ridge cross-fit as
if it were the full DoubleML.jl / R `DoubleML` stack; the code path is
labelled `pedagogical_dml`.

## 4. Data plan and current status

Intended: announcement dates, SUE, and daily returns for a US common-stock
universe, plus Ken French daily FF3. Status: none of that is in the
repo. `src/events/calendar.py` synthesises firms, days, a noisy proxy,
and a CAR that equals a linear function of X plus `true_ate * proxy`
plus Gaussian noise. `src/returns/car.py` will try the Ken French zip
URL and, on failure, synthesise Mkt-RF, SMB, HML. Tests never require
the network. This is honest: a CAR computed on synthetic factors is not
an event study.

## 5. Methods (what the code does)

OLS is the baseline: Y on an intercept, D, and X. DML is K-fold Ridge
for E[Y|X] and E[D|X], out-of-fold residuals, then a no-intercept OLS
of residual Y on residual D. Placebo permutes D. CATE repeats OLS
inside buckets. A download hook for French factors is best-effort.
The pre-analysis plan in `docs/PAP.md` freezes windows, winsorisation,
and the rule that we will not drop DML if it is noisier than OLS.

## 6. What has been run

Unit tests generate a null DGP (`true_ate=0`) and check that a placebo
permutation does not reject at 5% under a fixed seed, and that OLS on a
DGP with `true_ate=0.08` has a positive coefficient. Those tests are
about software, not about earnings. A toy entry `src/run_toy.py` writes
`results/tables/toy_dml.csv` with OLS, DML, placebo, and bucket CATEs,
all tagged TOY. **NO FULL RESULTS YET** on any actual announcement.

## 7. What we will not say in a workshop

We will not say that DML "proves causality" of earnings. Unconfoundedness
is assumed, not tested, except via placebo dates/permutations that can
only detect some forms of misspecification. We will not say that a
significant theta is a tradable alpha. We will not show a cumulative
PnL chart. We will not replace SUE with a social-media sentiment score
without an addendum. We will not claim the pedagogical Ridge DML is
Chernozhukov's forest/lasso IRM.

## 8. Identification discussion (for the discussant)

The threat is that X is incomplete. Analyst coverage, short interest, and
options-implied events are not in the toy X and may not be in the first
public extract either. If those omitted variables jointly move surprise
and CAR, DML on the observed X is not causal. The PAP's answer is to
name the assumption and to pre-specify X rather than to claim a design.
A regression discontinuity in surprise around zero is a different paper.
An instrument for SUE (for example, a weather shock to a retailer's
quarter) is also a different paper. This project is DML under
unconfoundedness, or it is nothing.

Overlap is the second threat. Extreme SUE names may have no comparable
untreated units in X-space. We will report a simple histogram of the
fitted E[D|X] and drop the 2% tails as robustness, pre-specified in
spirit if not yet coded as a flag.

## 9. Heterogeneity

PEAD is often described as stronger among small, neglected names. That
is a CATE statement. We slice by size and by vol because those are the
two buckets that the accounting literature already argues about. We do
not slice by 12 industries in the first pass (power). If size CATE and
vol CATE tell opposite stories, we will report both, not the one that
matches Bernard and Thomas.

## 10. Placebo and specification tests

On the synthetic null, permuting D should yield p > 0.05 at the test
seed. That is in `tests/test_dml_placebo.py`. On real data we will shift
announcement dates forward by a random 30-80 trading days and expect
the DML ATE to collapse. If it does not, the code is picking up a
calendar or factor residual, not surprise.

## 11. Computational status and empty tables

Table 1 (planned): OLS vs DML ATE on SUE, real sample — empty.
Table 2: CATE by size and vol — empty except TOY.
Table 3: placebo dates — empty except TOY permutation.
Table 4: robustness windows [0,0] and [+2,+60] — not coded as a full
grid yet; CAR helper supports custom windows.

The workshop talk should show Table 1 as a framed empty box. Filling it
with the toy 0.05 would be a lie: that 0.05 was the DGP parameter.

## 12. Limitations

No I/B/E/S. No CRSP. No clustering in the toy SE. Ridge is a weak
learner for nonlinear confounding; the DGP is linear, so DML is not
stress-tested. Continuous treatment with a linear theta is a PLR, not
an IRM with binary D. The French download may fail in CI; tests use
synthetic factors. Copyright of Ken French files remains with that
library; we do not vendor them.

## 13. Next actions

1. Obtain a public or licensed event file; document the licence.
2. Parse Ken French daily factors into `data/french_ff3_daily.csv`.
3. Freeze PAP; run OLS/DML once; write Table 1 with SEs.
4. Run placebo dates on the real calendar.
5. Only then discuss PEAD windows.

## 14. Workshop 12-minute arc

Slide 1: Ball/Brown vs PEAD vs this paper's narrower DML question.
Slide 2: estimand and unconfoundedness.
Slide 3: pedagogical DML vs OLS.
Slide 4: empty empirical table, TOY disclaimer.
Slide 5: what would change our mind (placebo, omitted coverage).
Closing sentence: a planted ATE recovered from a linear DGP is a unit
test, not an earnings result.

## 15. Conclusion of the outline

The contribution at this stage is a PAP, a CAR helper, and a
cross-fit estimator with a placebo test that passes on a synthetic
null. The empirical contribution is reserved until SUE and returns
exist. This report does not fill that gap with invented t-statistics.


## 16. Notes for a CSE PhD reader

A computer-science reader may ask why this is not "just another
gradient-boosted CAR prediction". Prediction of Y from (D, X) is a
different target: it can be excellent while theta is biased. DML spends
degrees of freedom on orthogonalising D and Y with respect to X so that
the coefficient on D is less sensitive to the nuisance fit. That is why
we report theta and its SE, not out-of-sample R^2, as the headline.
It is also why a Kaggle-style leaderboard would be the wrong artefact
for this repository. The right artefact is a PAP, a toy recovery of a
planted ATE, and later a real table that may well be a precise zero.

If the discussant prefers a binary beat/miss treatment, the IRM path in
DoubleML is the natural extension; we will not switch after seeing PLR
results. If they prefer a panel TWFE of returns on surprise, that
estimator does not by itself solve the high-dimensional X problem and is
out of scope for v0.1. The codebase is intentionally small so that a
lab can audit the residual-on-residual step in one file.
