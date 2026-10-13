# Phase 6: controlled cadence, sample budget and phase experiments

## Question and protocol

Does accurate shedding frequency imply accurate future fields, and how do cadence and finite training windows affect that conclusion? All experiments use the same settled Re100 float64 trajectory, complete P2 coefficients and consistent mass M. They do not establish cross-Re or cross-mesh robustness.

There are 36 configurations and 72 model fits, using both linear and affine regressions. Pseudoinverse cutoff is fixed at 1e-10. Ranks and experiment budgets are predeclared; diagnostic prediction errors are not used for hyperparameter selection. Some times overlap the Phase5 evaluation data, so this is not a new independent test set.

| Study | Fixed controls | Variable |
|---|---|---|
| A | Rank 6; training27.2-28.8; common future times28.96/29.12/29.28/29.44 | Cadence0.01/0.02/0.04/0.08/0.16; samples 161/81/41/21/11 |
| B | Rank 6;11 snapshots; training endpoint28.8; same future times | Cadence and observation duration 0.1-1.6 |
| C | Rank 10 or20;121 snapshots; duration 1.2; forecast horizon1.2 | Eight approximately eighth-cycle window shifts |
| D | Rank 10; cadence 0.01; endpoint28.8; same future times |34-180 snapshots, approximately 0.986-5.348 cycles |

All means, bases and operators use the respective training window only. Predictions seed the final training frame and never reset. Lift period arranges phase shifts but does not select a DMD eigenvalue. Since N-1=T/dt, duration, count and cadence cannot all remain fixed when cadence changes. Four common evaluation times are fair across cadence but miss within-cycle errors. Native-time errors are also retained. Fluctuation denominators depend on each training mean; common-time total-field errors use an identical truth denominator and are available for comparison.

## Results

| Cadence | Fixed-duration affine fluctuation error | Relative frequency discrepancy from CFD | Fixed-11-sample affine error |
|---|---:|---:|---:|
|0.01|3.8489%|0.001900%|55.4434%|
|0.02|3.8649%|0.001851%|28.1894%|
|0.04|3.8975%|0.001664%|4.1579%|
|0.08|4.1314%|0.002169%|9.5170%|
|0.16|13.3490%|0.016670%|13.3490%|

StudyA removes the changing-rank confound in Phase5. Sample count still changes, so the final discrepancy cannot be attributed exclusively to aliasing. Frequency remains accurate while field prediction degrades. The roughly3.85% fine-cadence error also reflects rank 6 spatial truncation.

With11 snapshots, dense cadence 0.01 covers only about 0.30 cycles. The affine design condition number is 2491.5, versus 48.2 at cadence 0.04. Dense sampling at fixed sample budget is not always informative. All designs remain full row rank at the chosen cutoff; rank alone does not guarantee reliability. Condition numbers and errors are associated here, not a complete causal proof. Cadence0.04 is not a universal optimum.

| Rank in phase study | Linear future fluctuation error range | Affine range |
|---|---:|---:|
|10|11.5406%-11.6389%|0.75861%-0.77282%|
|20|11.7942%-11.9170%|0.03197%-0.04897%|

Overlapping windows on one trajectory are not independent repeats; no confidence intervals or significance tests are used. Phase shifts also change absolute time, potentially incorporating residual drift.

StudyD linear rank 10 errors near 1.494/2.480/3.495/4.481 observed cycles are 27.98%/17.31%/12.37%/9.71%. Near2.002 and 4.003 cycles they are 2.31% and 1.38%. Affine error stays around 0.744%-0.815%. Longer windows do not monotonically improve this regression. These observations are consistent with finite-window centering, not a theorem that integer cycles are universally optimal.

## Mechanism and prior art

For a reference linear system a(k+1)=R a(k), subtracting a finite training mean m gives z(k+1)=R z(k)+(R-I)m. A zero-intercept regression may miss the resulting translation. Exact affine recovery in this reference system requires a full-row-rank augmented source matrix and exact data. This elementary identity is explanatory, not a new theorem or an exact description of nonlinear CFD.

The incomplete-cycle analytic ROM check confirms this reference mechanism. Mean handling and affine terms have existing theoretical treatment; sampling range, resolution and truncation are also established research topics:

- Hirsh et al., Centering Data Improves the Dynamic Mode Decomposition: https://arxiv.org/abs/1906.05973 ; DOI 10.1137/19M1289881.
- Seenivasaharagavan et al., Clarifying the effect of mean subtraction on Dynamic Mode Decomposition,2025 revision: https://arxiv.org/abs/2105.03607 . Its companion-matrix results should not be indiscriminately applied to this coordinate regression.
- Li et al., A Parametric and Feasibility Study for Data Sampling of the Dynamic Mode Decomposition: https://arxiv.org/abs/2110.06577 ; related DOI 10.1063/5.0082640.

Decision: do not claim the affine correction or general sampling sensitivity as original contributions. This release is a reproducible verified case study. Numerical-error versus forecasting-error criteria remain a possible future research direction needing further evidence and literature review.

## Checks and reproduction

All 36 configurations/72 fits pass finite-error, numerical design-rank and control-budget checks. Three analytic ROM tests passed again; the solver and ROM algorithm were unchanged. `summary.json` records frequency, both norms, conditions, spectral radii and time boundaries. `audit.json` records checks. `phase6-controls.png/.svg/.pdf` presents the studies.

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python experiments/analyze_phase6.py
PYTHONPATH=src python -m unittest discover -s tests -p test_rom.py -v
```

Python plotting suffices here; ParaView is optional for the saved spatial fields. See the final report for release status and future work.
