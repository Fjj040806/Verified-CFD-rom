# Phase 3 — closed-wall cavity, Re=100

Completed 2026-10-13. Actual MAC computations, not generated artwork.

## Method

Unit square, lid speed 1, viscosity .01; u[Ny,Nx+1], v[Ny+1,Nx], p[Ny,Nx]. Normal face velocities are zero on walls. Tangential ghost reflections impose no slip, with u_top=1 at top face midpoints. Corners keep side-wall normal velocity zero; the discontinuous lid is not regularized.

Incremental pressure correction, first-order in time, centred explicit convection, implicit diffusion. DCT-II diagonalizes the compatible cell-centred Neumann pressure Laplacian. DST-I/DST-II diagonalize the two component-specific homogeneous wall diffusion operators; nonhomogeneous lid forcing is added to the RHS. The old pressure is included in the predictor. Pressure mean is zero. This closed-wall solver does not reuse periodic wrapping.

Steady criterion: RMS velocity change over an approximately .5-time-unit window, divided by the window duration, below 1e-7 for four consecutive windows. Max allowed time is 60; runs fail rather than silently label unconverged results steady.

## Results

| Grid | dt | Stopping time | Change rate | Divergence RMS | u benchmark RMS | v benchmark RMS |
|---|---:|---:|---:|---:|---:|---:|
| 32² | 0.006 | 26.892 | 3.745e-08 | 5.387e-16 | 0.003220 | 0.002531 |
| 64² | 0.003 | 26.553 | 4.311e-08 | 1.075e-15 | 0.001385 | 0.003659 |
| 128² | 0.0015 | 26.973 | 3.393e-08 | 2.059e-15 | 0.002158 | 0.004476 |
| 64² | 0.0015 | 26.973 | 3.415e-08 | 1.034e-15 | 0.001385 | 0.003659 |

Benchmark RMS uses 17 tabulated sample points per centreline, linear interpolation with exact wall values appended, and velocity normalized by lid speed. It is a sample RMS absolute discrepancy, not relative percentage error.

## Grid and temporal checks

Centreline differences on a common 501-point coordinate grid:

| Refinement pair | u difference RMS | v difference RMS |
|---|---:|---:|
| 32 to 64 | 0.0043813 | 0.0047599 |
| 64 to 128 | 0.0010606 | 0.0011592 |

64² dt halving: native velocity difference RMS 1.445109e-08. This is a steady-solution sensitivity check, not a transient order estimate. Starting from zero is retained for all runs.

Benchmark discrepancies are NOT monotone with grid refinement. The solutions approach one another, but do not approach every Ghia sample monotonically. Possible contributors include finite-resolution reference, centreline interpolation, convection discretization and lid-corner treatment. Their relative contributions have not been isolated. Do not claim the present result is more accurate than the published reference, or infer global second-order convergence from this cavity alone. A modern independent high-accuracy reference and/or smooth-wall manufactured solution is the next validation extension if high-precision force or corner claims are pursued.

## Checks and visualization

Eleven tests passed: three new wall tests check Neumann inversion, projection/normal-wall invariance and implicit diffusion against the explicit stencil; all eight previous tests retained. Steady flow VTI was checked by native VTK reader. ParaView GUI not available. Python figures include reference curves, convergence history, computed streamlines, a dark-palette flow view, and a startup animation. Display velocity is interpolated to centres; native u/v/p retained in NPZ. Display vorticity uses a cell-centred gradient approximation; it is not a force calculation.

## Innovation register

The corner discontinuity and nonmonotone reference discrepancy motivate a controlled comparison of discontinuous vs smooth lids, interpolation error and independent references. This is a reproducibility/verification lead, not established novelty. No novel boundary method has been implemented. The pressure-tolerance-to-ROM study remains pending; this DCT pressure solve supplies a tightly solved reference.

## Reference provenance

Ghia, U.; Ghia, K.N.; Shin, C.T. (1982), Journal of Computational Physics 48, 387–411, DOI 10.1016/0021-9991(82)90058-4. Re100 centreline values transcribed from Tables I and II. Checked against the downloaded original paper, pages 398–399; OCR split digits and minus signs were reconciled with the table values.

Primary paper hosted at: https://www.ljll.fr/~frey/papers/Navier-Stokes/Ghia%20V.%2C%20High-Resolutions%20for%20incompressible%20flow%20using%20the%20Navier-Stokes%20equations%20and%20a%20multi-grid%20method.pdf
Readable original-paper mirror: https://www.scribd.com/document/221152349/Ghia-v-High-Resolutions-for-Incompressible-Flow-Using-the-Navier-Stokes-Equations-and-a-Multi-grid-Method

## Reproduce / next stage

Run from project root:
```
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python experiments/verify_phase3.py
PYTHONPATH=src python experiments/visualize_phase3.py
```
Optional VTK is required by visualization export inspection.

Next Phase 4: cylinder backend/geometry feasibility, benchmark boundary matching and low-Re verification before vortex shedding; no lift/drag or cylinder claims are supported by this cavity stage.
