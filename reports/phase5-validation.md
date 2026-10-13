# Phase 5: mass-weighted POD and affine DMD

## Protocol

The question is whether a small number of velocity modes preserve wake structure and frequency, and whether that conclusion survives sampling, parameter and numerical changes. This stage establishes a single-Re baseline only.

The float64 coarse-half P2 velocity data are restricted to time 27-30: 301 snapshots, interval 0.01. Chronological training spans 27-28.79, validation 28.8-29.39, test 29.4-30. The mean, consistent-mass POD basis and regression operator use training data only. Candidate ranks are 1,2,3,4,6,8,10,12,16,20. POD selects the smallest candidate with at least 99% training fluctuation energy and at most 1% validation fluctuation error. DMD selects the candidate with minimum validation fluctuation error. Test curves are diagnostic and do not reselect the models.

POD uses the Gram matrix X^T M X. Reconstruction uses observed projection coefficients. DMD fits c(k+1)=A c(k)+b and compares b=0; forecasting starts at the final training frame and recursively traverses validation and test without truth resets. Dominant positive frequency is selected by training modal activity, not proximity to a reference frequency. It is a coordinate regression model, not a Galerkin Navier-Stokes ROM.

Error is the mass-weighted norm over the complete evaluation interval, divided either by the total true field norm or by the true field minus training mean. Both are reported to expose mean-flow dominance.

## Results

| Measurement | Value |
|---|---:|
| Selected POD rank | 10 |
| Retained training fluctuation energy | 99.99434% |
| POD test total-field error | 0.19099% |
| POD test fluctuation error | 0.75443% |
| Selected DMD rank | 20 |
| DMD test total-field error | 0.01097% |
| DMD test fluctuation error | 0.04333% |
| Same-rank linear DMD test fluctuation error | 7.57276% |
| Mean baseline fluctuation error | 100% |
| Last-training-frame persistence error | 139.925% |
| DMD frequency | 2.98750464 |
| CFD lift frequency | 2.98756477 |
| Relative frequency discrepancy | 0.002013% |

Rank 20 is the candidate upper bound, not a globally optimal rank. More retained spatial modes explain much of the advantage over rank 10 reconstruction. At rank 10, DMD fluctuation error is 0.76558%, close to POD's 0.75443%. The leading two modes retain 95.414% of training fluctuation energy but have 21.393% test reconstruction error.

The maximum forecast horizon is 1.21, approximately 3.6 shedding periods. This is a settled, same-condition, short-horizon test. The CFD Strouhal discrepancy from the official reference is still 1.0369%; close DMD/CFD frequency agreement does not eliminate that error.

## Sensitivity

| Snapshot interval | Rank | Test points | Forecast fluctuation error |
|---|---:|---:|---:|
| 0.01 | 20 | 61 | 0.04333% |
| 0.02 | 20 | 31 | 0.04316% |
| 0.04 | 20 | 16 | 0.04551% |
| 0.08 | 20 | 8 | 1.31757% |
| 0.16 | 10 | 4 | 26.6010% |

Physical split boundaries remain fixed but actual sampled endpoints shift slightly. The final case also reduces rank because fewer training samples are available; it does not isolate aliasing. Nyquist frequency 3.125 is near the fundamental 2.988 and cannot resolve higher harmonics. Fixed-cadence training windows of 0.5,1.0,1.5 yield fluctuation errors 0.01237%,0.01524%,0.03306%; this is nonmonotonic exploratory sensitivity, not test-based window selection.

## Verification and figures

All 16 numerical/ROM checks passed. Analytic checks cover nonuniform mass weighting, held-out harmonic forecasts and incomplete-cycle affine regression. Maximum mass-orthogonality discrepancy across all retained modes is 6.61e-6; tiny-energy Gram modes are conditioning-sensitive. Maximum reconstructed B u norm is 3.28e-16, a weak algebraic continuity check rather than pointwise divergence.

`phase5-rom` contains spectrum, rank errors, forecast coefficients and cadence sensitivity. `phase5-flow-comparison` displays time 29.70 CFD, rank 10 reconstruction, rank 20 prediction and velocity error. The first three panels share a speed scale. Display uses P2 vertex velocities with linear interpolation; quantitative errors retain complete P2 coefficients and M. NPZ includes mean, basis, spectrum, operator, bias, reconstruction and forecasts.

## Research decision and reproduction

Weighted POD, affine DMD and finite-window centering are established techniques, not new contributions here. Further controlled sampling studies and numerical/parameter robustness are needed. See Phase6 and the final report for current release status.

```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python experiments/analyze_phase5.py
python experiments/visualize_phase5.py
PYTHONPATH=src python -m unittest discover -s tests -v
```

The full suite requires optional cylinder dependencies and Gmsh shared libraries. Analysis alone needs NumPy, SciPy and Matplotlib. Background: Tu et al., On Dynamic Mode Decomposition: Theory and Applications, DOI 10.3934/jcd.2014.1.391.
