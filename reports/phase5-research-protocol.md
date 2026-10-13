# Locked research protocol for POD/DMD (next stage)

Core question: In verified two-dimensional cylinder shedding, how many velocity
modes preserve the dominant flow structures and shedding frequency, and how
robust are those conclusions to Reynolds number, sampling and numerical error?

Primary current case: DFG Re100. A single Re100 case does not establish
cross-Re robustness; additional Re cases remain a later experiment.

1. Confirm a settled interval with repeated cycles and quantified drift.
2. Choose a single mesh and time-step baseline. Keep complete P2 coefficients,
   consistent mass M and weak divergence B. Subtract the training mean only.
3. Contiguous split: first 60% training, next 20% validation, last 20% test.
   No randomly shuffled neighbouring frames and no fitting on held-out data.
4. Weighted POD via snapshot Gram matrix X^T M X. Choose rank on training and
   validation, then report held-out mass-norm errors and mean-flow baseline.
5. Report energy content, held-out reconstruction, B-mode/field diagnostics,
   and frequency agreement. High energy retention alone is not success.
6. DMD: uniform snapshots, rank and window sensitivity. Separate reconstruction
   from genuine forecasting. Compare against persistence and mean baselines.
7. A decimation study keeps physical observation duration fixed; a window study
   keeps sample cadence fixed. Report Nyquist limits and finite-window spectral
   resolution. Do not interpret zero padding as new frequency information.
8. Numerical-error study: current FEM baseline uses nonlinear-solve tolerances,
   temporal refinement and polygonal geometry / force-estimator audits. The
   earlier pressure-Poisson tolerance hypothesis applies to the MAC branch;
   it does not describe the mixed monolithic FEM solver.
9. Reynolds dependence requires matched geometry, boundary scaling, viscosity
   and state mapping. No claim based on comparing unrelated cylinder domains.

Force postprocessing is a validated diagnostic and candidate sensitivity study,
not an invented new method. Weighted POD, DMD and cached Newton solves are
established methods. A novelty claim needs prior-art review and reproducible
results beyond those methods.

Phase 4B creates data for these tests; it does not already answer the rank,
forecasting or cross-Re questions. One published comparison sequence is also
not statistical evidence of universally robust behaviour.
