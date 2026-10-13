# Project summary

**Verified Incompressible Flow and Reduced-Order Analysis**

A scientific-computing study of how numerical verification and data-driven reduction interact in two-dimensional incompressible flow. The project connects compatible staggered-grid operators, pressure correction, fitted-mesh Taylor-Hood finite elements, nonlinear BDF2 integration, benchmark force diagnostics, mass-weighted POD and affine DMD.

The central question is whether a compact velocity representation preserves wake structure and frequency, and whether accurate frequency recovery implies accurate forecasting. The work includes analytic operator/time-stepping checks, a lid-driven cavity, confined Re20/Re100 cylinder benchmarks, chronological held-out evaluation and controlled sampling experiments.

Ten POD modes reconstruct held-out Re100 velocity fluctuations with 0.7544% error. A validation-selected 20-mode affine DMD gives 0.04333% short-horizon fluctuation prediction error. Under fixed-rank sparse sampling, frequency remains close to the numerical flow while field error rises to 13.35%. This illustrates why spatial prediction and frequency diagnostics must be reported together.

The project's mathematical content includes discrete adjointness and nullspaces, elliptic solves, finite-element weak forms, stability/accuracy distinctions, weighted eigenproblems and linear dynamical regression. Its engineering content includes reproducible experiments, checks of numerical contracts, state/metadata export and visualization of actual computed fields.

This is a completed computational case study, not a claimed new algorithm or published paper. Cross-Re/cross-mesh ROM generalization and long-horizon performance remain future work. Project-owner application statements should describe actual personal contributions and understanding; the technical summary alone does not assign authorship.
