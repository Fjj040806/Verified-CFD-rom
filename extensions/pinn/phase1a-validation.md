# Phase 1A: saved-mesh reduced physics and exact sensor audit

Date: 26 October 2026. Status: numerical operator/probe audit completed. **No PINN has been trained.** Automatic differentiation and reference-trajectory consistency still require a Phase 1B check before the pilot.

## Scope and inputs

The baseline is the original Re100 data at repository commit `25f9dca3b068b92b9c3d904001ff9978f2f7ab10`. Input data, M, B, mesh and solver-source hashes were checked against the saved release manifest. All dense-field calculations in this checkpoint use only the 180 offline frames at 27-28.79. No complete online or future evaluation field enters an operator diagnostic or a fitted model.

The saved mesh was reconstructed directly; Gmsh was not rerun. scikit-fem 12.0.2, the original P2 velocity basis, quadrature order 5, and the original nonlinear vector/Jacobian forms were reused. Weak-form pressure elimination was checked through homogeneous mode boundary traces and B-mode residuals. The computed reduced mass was retained rather than replaced by identity.

## Operator and probe checks

| Check | Measured value |
|---|---:|
| Velocity-coordinate ordering maximum difference | 0 |
| Relative Frobenius discrepancy of reassembled M | 0 |
| Relative Frobenius discrepancy of reassembled B | 0 |
| Mean prescribed-boundary maximum error | 6.66e-15 |
| Relative mode boundary trace | 1.54e-22 |
| Matrix-scaled weak-divergence residual of modes | 2.90e-15 |
| Eight-sensor polynomial-field maximum error | 6.66e-16 |
| Hidden-probe polynomial-field maximum error | 8.88e-16 |
| Maximum entry discrepancy M_r versus identity at rank 20 | 9.48e-10 |

The 8 measured locations provide 16 interleaved velocity channels. Probe evaluation uses the exact scalar P2 basis and verified vector-component index maps, not the P1 interpolation used in display figures. The 16 hidden locations are disjoint from the measured locations. Nominal sensor coordinates were fixed before looking at these conditioning results.

The symmetric quadratic convection tensor, constant term and linear operator were assembled to reproduce the original projected finite-element vector field. Ten fixed-seed, offline-scale coefficient vectors per rank compared tensor evaluation with direct full-order assembly. Maximum scaled discrepancies were 1.15e-15, 1.66e-15 and 1.25e-15 at ranks 6, 10 and 20. Tensor directional derivatives agreed with centered finite differences within 6.21e-12. This derivative check does not substitute for a future neural-library automatic-differentiation test.

## Sensor information and spatial closure

| Rank | Rank of H*Phi | Instantaneous nullity | Full-column condition number | Offline spatial closure / projected rate |
|---|---:|---:|---:|---:|
| 6 | 6 | 0 | 4.86 | 1.14376% |
| 10 | 10 | 0 | 21.73 | 0.54684% |
| 20 | 16 | 4 | Not finite | 0.02231% |

The 20-state sensor map is necessarily underdetermined at one time with only 16 channels. This does not prove that a temporal inverse problem is unobservable: dynamics and multiple observation times may supply additional information. It does establish why a per-time gappy least-squares model needs a declared minimum-norm or regularized convention.

The closure indicator is the difference of full-state and reconstructed-state projected operators, with the same mean, basis and reduced mass. Its norm is divided by the norm of the full-state projected rate. It is a spatial diagnostic evaluated on offline data, not total model error, a PINN result, or a bound on future solution error.

## Continuous-rate consistency: an additional pretraining issue

On 174 common offline interior frames (spatial closure above uses all 180 frames), 3-, 5- and 7-point centered snapshot derivatives were compared with the full-state projected semidiscrete rate. This comparison deliberately avoids attributing all discrepancy to spatial truncation.

| Rank | 7-point derivative/rate mismatch | 5-versus-7-point derivative sensitivity |
|---|---:|---:|
| 6 | 0.94443% | 0.09639% |
| 10 | 1.25850% | 0.21036% |
| 20 | 1.35916% | 0.26535% |

These are combined consistency diagnostics. They include differentiation of sampled data, the saved BDF2 trajectory, nonlinear-solve effects and residual projection effects; they are **not pure temporal-discretization errors**. In particular, the saved interval is 0.01 but the original BDF2 step is 0.005. A BDF2 residual using the larger saved interval is not the original solver residual.

At rank 20 this mismatch is appreciably larger than the measured spatial closure indicator. A continuous-time PINN residual should therefore not be presumed exactly compatible with the saved numerical trajectory. The cause and loss-design consequences require a matched reference/time-resolution investigation. The present result does not establish that the continuous model, the data, or a specific derivative stencil is uniquely responsible.

## Pure Galerkin diagnostic before neural modeling

Each reduced ODE was initialized from the complete first offline state and integrated over 27-28.79. This is an intentionally privileged **offline model diagnostic**, not a sparse-observation assimilation experiment or a fair comparison with the earlier test-window DMD results.

| Rank | Offline fluctuation trajectory error | Relative trajectory change under tighter adaptive tolerances |
|---|---:|---:|
| 6 | 10.6033% | 8.30e-9 |
| 10 | 7.8378% | 2.87e-9 |
| 20 | 8.8637% | 1.74e-9 |

RK45 tolerances changed from rtol=1e-8/atol=1e-10 to rtol=1e-10/atol=1e-12. Agreement supports negligible sensitivity to those tolerance settings here; it is not a fixed-step order proof. Lower spatial closure did not produce monotonically lower trajectory error. Error accumulation, conditioning, the continuous/discrete trajectory mismatch and other model discrepancies remain possible contributors.

## Deliverables and reproducibility

- `audit_phase1.py`: saved-mesh reconstruction, tensor/probe tests, offline closure, temporal stencils and Galerkin diagnostic.
- `phase1-results/reduced-operators.npz`: mean, modes, computed reduced mass, constant/linear/symmetric-quadratic coefficients, scales and offline times.
- `phase1-results/sensor-probes.npz` and `hidden-probes.npz`: exact interleaved observation maps.
- `phase1-results/summary.json`: complete measured quantities and software versions.
- `phase1-results/operator-audit.png/.svg/.pdf`: the numerical diagnostics.

From the repository root:

```bash
python -m pip install -r extensions/pinn/requirements-operators.txt
OPENBLAS_NUM_THREADS=1 python extensions/pinn/audit_phase1.py
```

Gmsh and Torch are not required for this checkpoint. Figures and checks use Python/Matplotlib; ParaView is unnecessary for these low-dimensional diagnostics. The audit can also use `--project-root PATH` for a separate verified baseline-data checkout.

## Next checkpoint and originality status

Phase 1B must implement and test automatic differentiation/time normalization, then examine the reference-rate mismatch before attributing a PINN advantage or failure to truncation. Only then should the predeclared nine-fit pilot proceed. The frozen Phase 0 pilot should not be silently retuned using online/future complete fields.

The existing POD-PINN and ROM-closure literature prevents treating the coupling itself as new. A possible research lead is how numerical-reference consistency, truncation discrepancy and sensor information jointly affect the apparent benefit of physics regularization. This is a hypothesis, not an established original contribution. No new empirical claims have been added to the original paper.
