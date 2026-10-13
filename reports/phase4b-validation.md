# Phase 4B — periodic cylinder shedding and ROM-ready data

Completed: Re100 transient implementation, late-period verification, local geometry/time sensitivity, controlled nonlinear-tolerance audit, and scientific visualization. POD/DMD analysis is the next stage; no mode-count or forecast result is asserted.

## Governing numerical problem

Same DFG channel and cylinder as Phase 4A; nu=.001, rho=1, inlet maximum 1.5 and mean 1, diameter .1. No-slip cylinder/walls and natural Laplacian traction outlet. Mixed P2/P1 FEM on affine triangles; circular CAD is approximated by polygonal facets. The monolithic constraint is weak continuity, not a pressure-Poisson projection.

Fully implicit BDF2: (3 M u_new - 4 M u_current + M u_previous)/(2 dt) + A u_new + N(u_new) - B^T p_new = 0; B u_new=0. First step and each restart use backward Euler. Nonlinear Newton corrections use cached exact Jacobians with periodic refresh and refresh on slowdown; acceptance checks the true residual. Caching is a solver optimization, not a changed PDE.

The transient force is the negative cylinder-DOF momentum reaction including M du/dt. Omitting inertia would incorrectly reuse a steady formula. Pressure probes and force scaling follow Phase 4A, with mean velocity 1.

## Actual comparison

| Run | h max | h circle | dt | St | St reference discrepancy | CD min | CD max | CL min | CL max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coarse | 0.065 | 0.009 | 0.01 | 0.29584995 | 1.9997% | 3.147752 | 3.211860 | -1.007978 | 0.973797 |
| refined | 0.065 | 0.006 | 0.01 | 0.29549963 | 2.1157% | 3.155309 | 3.220844 | -1.028836 | 0.994888 |
| coarse-half | 0.065 | 0.009 | 0.005 | 0.29875648 | 1.0369% | 3.153191 | 3.219490 | -1.033239 | 0.998546 |

Reference uses the official updated 20141124 archive, file bdforces_q2_lv6_dt4. Its time spacing is .000625 and St=.30188679245 from repeated lift minima. Do not silently substitute old tables for the updated force reference. The downloaded reference is not bundled as a copied full publication.

Source: https://wwwold.mathematik.tu-dortmund.de/~featflow/media/dfg_bench2new_2d/20141124_dfg_bench2_official.zip

Local circular-boundary refinement improves the drag extrema here, but shedding frequency does NOT improve monotonically. Maximum mesh size remains .065 in both grids; this is local refinement, not global uniform refinement or a complete mesh/time factorial. Temporal refinement reduces St discrepancy from 2.00% to 1.04%. The remaining discrepancy is real; no fully mesh-converged frequency claim is made.

## Periodicity and initialization

| Run | Analysis interval | Full cycles | Period std | Extrema drift / lift range | Settled gate |
|---|---|---:|---:|---:|---|
| coarse | 25.00–30.00 | 14 | 1.298e-05 | 4.411e-06 | True |
| refined | 28.00–33.00 | 13 | 2.952e-06 | 3.332e-05 | True |
| coarse-half | 27.00–30.00 | 8 | 1.067e-06 | 8.074e-05 | True |

Settled gate requires relative period std and early/late cycle-extrema drift below .001, with at least five periods. Quadratic three-sample interpolation of extrema reduces sample-phase bias. FFT plots retain finite observation-window resolution; zero padding is not additional information. Cycle curves are aligned at lift minima, not compared at unaligned clock times.

Coarse run: zero interior velocity with inlet prescribed at t=0, propagated to t=30. Refined mesh: coarse t=25 checkpoint, P2 interpolation with fluid-side boundary queries, mass-metric projection to enforce the new weak continuity constraint, then propagation to t=33; analysis starts t=28. Time-refined same mesh: same t=25 checkpoint, propagate to t=30; analysis starts t=27. Restart uses BE for one step before BDF2. Warmup is excluded from the stated analysis intervals.

## Verification and tolerance control

13 tests passed, including nonlinear-vector equivalence to library assembly, finite-difference Jacobian agreement, first-step residual and constrained boundary values, alongside all earlier numerical tests. Independent semidiscrete manufactured trajectory u(t)=exp(-t) u_bar, p(t)=exp(-t) p_bar uses the exact assembled forcing, including time derivative and quadratic convection; it isolates time error, not continuum spatial accuracy.

| dt | Relative mass-norm velocity error | Order |
|---|---:|---:|
| 0.04 | 2.252007e-04 | — |
| 0.02 | 5.570183e-05 | 2.0154 |
| 0.01 | 1.385655e-05 | 2.0072 |
| 0.005 | 3.455846e-06 | 2.0035 |

Coarse long-run nonlinear tolerance: 1e-8. Refined/time-refined runs: 1e-6. To quantify this difference, a matched same-mesh/same-dt/same-state .33-time-unit trajectory compared both tolerances. These are near-converged solver settings, but the grid/time comparison is not strictly tolerance-identical. The short audit does not prove long-time phase insensitivity.

- Maximum relative mass velocity difference: 5.055429e-06
- Maximum CD difference: 2.329628e-06
- Maximum CL difference: 5.971140e-06

## Snapshot integrity and visualization

Selected ROM baseline: coarse-half (dt=.005), complete P2 velocity coefficients in float64. Pressure remains separate; M and B are saved. Raw snapshots include startup after restart, so Phase 5 must restrict to the validated interval [27,30] before splitting. The coarse exploratory run saved float32 snapshots; it is not the ROM baseline. Metadata records that difference.

Selected saved-snapshot weak divergence coefficient norm maximum: 1.812361e-16. This does not imply pointwise divergence is zero.

Python GIF links actual vorticity evolution to lift with a time cursor. The visualization L2-projects the discontinuous velocity-derived vorticity into continuous P1 for display; raw fields and all numerical diagnostics are unchanged. Display colour range is clipped to ±25 to reveal the wake; it is not the true full range. Pressure/velocity visualization uses nodal interpolation and does not preserve the full P2 representation in VTU. Complete coefficients remain in NPZ. Cell-mean vorticity and display-projected vorticity are both exported. All exported frames are checked using native VTK; no ParaView GUI inspection is claimed.

## Research implications

We now have periodic baseline data and an approximately 1% numerical frequency bias. A ROM may reproduce the computed frequency to better than 1%, but that does not make it physically more accurate. Phase 5 will separate mass-weighted reconstruction error, frequency recovery and held-out prediction. Numerical and sampling sensitivity remain core tests; cross-Re robustness requires additional matched-Re cases and is not established by this Re100 dataset.

The observed nonmonotonic local-mesh frequency response, force-estimator differences from Phase 4A, and numerical-tolerance audit are leads for controlled robustness studies, not novelty claims. Standard BDF2, Newton, weighted POD and DMD are established methods.

## Reproduce

Install requirements-cylinder.txt plus optional VTK for export checks; Gmsh OS-library guidance is in Phase 4A. Run commands from project root with PYTHONPATH=src and OPENBLAS_NUM_THREADS=1.
```
python experiments/run_phase4b.py --name coarse --dt .01 --end 30
python experiments/run_phase4b.py --name refined --h .065 --hc .006 --dt .01 --end 33 --restart results/phase4b/coarse-checkpoint25.npz --transfer --tol 1e-6
python experiments/run_phase4b.py --name coarse-half --dt .005 --end 30 --restart results/phase4b/coarse-checkpoint25.npz --tol 1e-6 --snapshot-dt .01
python experiments/verify_bdf2.py
python experiments/audit_transient_tolerance.py
python experiments/analyze_phase4b.py
python experiments/visualize_phase4b.py
```
The updated official-reference summary is bundled; its source and extraction method are recorded. Newly reproduced coarse snapshots default to float64 rather than the initial run’s float32; force histories are unaffected by the storage conversion.

## Next stage

Phase 5: restrict baseline to settled snapshots; perform weighted POD with contiguous train/validation/test windows; assess DMD frequency/forecasting, then cadence and window-length sensitivity. Do not claim rank selection or prediction performance before those tests.
