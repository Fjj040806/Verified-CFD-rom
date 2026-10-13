# Phase 1 — Periodic MAC and pressure solver verification

Completed 2026-10-13. Actual numerical results, not full CFD trajectories.

## Implemented
- Uniform rectangular periodic MAC grid with separate p/u/v coordinates.
- Unique periodic face DOFs: all arrays (ny,nx), unlike the uncompressed closed-wall layout in Phase 0.
- Compatible gradient G, divergence D and Laplacian L=DG.
- Discrete Fourier Poisson solver with exact discrete eigenvalues; zero-mean gauge.
- Matrix-free conjugate gradients with constant-mode identity augmentation: SPD full-space operator, no asymmetric pinning. No preconditioner implemented yet.
- Velocity projection and absolute/relative residual, energy and divergence diagnostics.
- RHS compatibility checks reject nonzero forcing means; projection uses input flux scale to account for cancellation when projecting an already nearly solenoidal field.

## Numerical evidence
Analytic pressure sin(x)cos(2y) solves Laplacian p=-5p on [0,2pi]^2.

| Grid | RMS pressure error | Observed order |
|---|---:|---:|
| 16² | 2.23809255e-2 | — |
| 32² | 5.49465746e-3 | 2.0262 |
| 64² | 1.36747742e-3 | 2.0065 |
| 128² | 3.41484197e-4 | 2.0016 |

Gradient, divergence and Laplacian truncation errors also decrease approximately fourfold per refinement; values are in results/phase1-convergence.csv. These are periodic smooth-field results only.

Projection experiment: fixed seeded random face fields on 64², dt=0.02. These are synthetic operator stress tests, not physical flow snapshots.

| CG rtol | Iterations | Corrected divergence RMS |
|---|---:|---:|
| 1e-3 | 54 | 1.91458161e-2 |
| 1e-5 | 96 | 1.93239077e-4 |
| 1e-7 | 119 | 1.56234585e-6 |
| 1e-9 | 135 | 1.87886195e-8 |
| 1e-11 | 148 | 1.88773333e-10 |

FFT projection divergence RMS: 8.07564e-15. Identity max error ||D u_new-dt*(b-Lphi)||_infinity < 1.6e-14. Discrete kinetic energy decreases from 1.00623307 to 0.51988408 under the orthogonal projection; this does not establish time-integrator energy behaviour.

## Tests and bug resolved
Four unittest cases passed: rectangular-grid discrete adjoint relation and Poisson inversion; projection residual identity, near-idempotence and energy decrease; incompatible/zero RHS; second-order Poisson convergence. Repeated projection exposed overly strict compatibility checks based only on the small residual RHS. The check now uses the scale of input face fluxes for projection, while direct Poisson calls remain strict. Significant incompatibility is still rejected.

## Research implications
The residual/divergence trend confirms a known discrete identity, not a novelty claim. It creates a controlled baseline for studying whether pressure-solver errors alter later POD/DMD diagnostics. No ROM or frequency effect has been measured.

## Scope and next gate
Phase 1 is complete for the explicitly planned periodic foundation. Walls/ghost cells, cavity boundaries, viscous/convection predictor, time integration, cylinder, ROM and performance timing are not implemented. Closed-wall Poisson requires separate boundary and nullspace verification before cavity calculations.
Next Phase 2: integrate Navier-Stokes on the periodic grid, validate against exact Taylor-Green decay, measure temporal order independently of spatial error, and check the full method's energy evolution.

Runtime: NumPy/SciPy/Matplotlib; SciPy 1.17.0. Reproduction commands in README. Raw CSV/JSON and regenerated PNG/SVG/PDF included.
