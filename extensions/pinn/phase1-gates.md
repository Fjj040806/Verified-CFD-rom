# Phase 1 acceptance gates: operators and sparse observations

Phase 0 is complete; every item below is still pending. Do not train a PINN until the applicable gates pass.

1. Load the baseline from `data/demo/re100-settled.npz`, consistent M, saved B, and saved mesh/coefficient coordinates. Restrict the offline POD construction to 27-28.79.
2. Reconstruct the saved mesh directly from points/triangles, rather than regenerating a nominally similar Gmsh mesh. Rebuild boundary tags from its boundary facets. Compare velocity/pressure ordering and saved coefficient coordinates. A different ordering requires an explicit verified map.
3. Reassemble M using the original P2 vector basis and order-5 quadrature. Require relative Frobenius discrepancy from the saved matrix at most 1e-12. Assemble stiffness and convection on the identical mesh.
4. Audit the mean, modes, boundary traces, reduced mass matrix, and weak continuity. Record absolute and matrix-scaled B residuals rather than claiming a universal physical meaning for the Euclidean B norm. Confirm the assumptions needed for pressure elimination; stop or retain pressure if they fail.
5. Construct exact P2 sensor probes at the declared 8 locations and 16 disjoint hidden probes. Check row ordering on analytic vector polynomial fields to tolerance 1e-12*(1+field magnitude). Report singular values/rank of H*Phi; sensor coordinates being inside the analytic channel is not a probe-matrix validation.
6. Assemble c, L, Q and M_r for ranks 6, 10 and 20. For ten predetermined offline-scale coefficient vectors, require reduced tensor evaluation and direct FEM projection to agree within 1e-10*(1+direct norm). Use the same mass-corrected coordinates for every extension model.
7. Compute an offline-only instantaneous spatial closure indicator by subtracting projected full and reconstructed-state operators. Also measure snapshot temporal-difference consistency separately; do not label differencing error as closure error.
8. Integrate a pure Galerkin baseline before adding a neural model, checking convergence under ODE-step refinement. Whether it is stable and useful is an experiment, not an assumption.
9. Verify normalized-time automatic differentiation and tensor gradients using float64 once the training dependency is installed. The network's derivative with respect to normalized time must include the physical-time chain-rule factor.
10. Save operator/probe metadata and checks, including source commit, mesh hash, exact data windows, basis scales, dimensions, quadrature and software versions. New metadata must distinguish computed results from configured intentions.

Expected deliverables: reduced operators, probe matrices, basis/constraint diagnostics, offline closure and sensor-conditioning figures, and a validation report. This is the next checkpoint before the nine-fit pilot.
