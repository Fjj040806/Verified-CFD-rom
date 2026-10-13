# Phase 2 — actual validation results

Completed: periodic first-order Navier–Stokes implementation and verification. See phase2-method.md for the equations and limitations.

## Spatial convergence

T=0.02; dt scales with h². Velocity RMS is sqrt(mean((eu²+ev²)/2)); pressure RMS uses a zero-mean gauge.

| N | dt | Velocity RMS | Pressure RMS | Velocity order | dt-halving change / error |
|---|---:|---:|---:|---:|---:|
| 16 | 0.0005 | 2.556598e-05 | 0.005211798 | — | 0.190% |
| 32 | 0.000125 | 6.416436e-06 | 0.001315326 | 1.9944 | 0.193% |
| 64 | 3.125e-05 | 1.605672e-06 | 0.0003296083 | 1.9986 | 0.194% |
| 128 | 7.8125e-06 | 4.015158e-07 | 8.245069e-05 | 1.9996 | 0.194% |

Pressure refinement orders: 1.9864, 1.9966, 1.9991

## Time convergence

N=64, T=0.5. Semidiscrete exact stationary-TG velocity separates temporal error from the spatial floor.

| dt | Semidiscrete velocity error | Order |
|---|---:|---:|
| 0.02 | 9.011471e-05 | — |
| 0.01 | 4.511503e-05 | 0.9982 |
| 0.005 | 2.257197e-05 | 0.9991 |
| 0.0025 | 1.12896e-05 | 0.9995 |

## Additional evidence

- Translated TG background velocity (0.3,0.2), T=0.2: velocity errors N=32: 0.0002357, N=64: 6.06869e-05, N=128: 1.62738e-05. These combined h/dt refinements are not an independent spatial-order claim.
- N=64, dt=.005, T=2: final energy relative error 0.104198%; divergence RMS maximum 4.48373e-16. Energy decreases at every recorded sample for this case.
- Eight tests passed including convection truncation, uniform-flow invariance, discrete TG amplitude, Phase 1 contracts, and VTK roundtrip.
- All 21 generated VTI frames verified with native VTK reader: 4096 points and four data arrays each. ParaView GUI not available; interactive rendering not inspected.

## Research register

Pressure spatial and temporal errors can partially cancel in a fixed-grid refinement test. Treat as a controlled-error-analysis lead, not novelty. The solver-tolerance / ROM question remains pending. Stationary TG is a low-complexity field and cannot establish robustness on generic high-Re flows.

## Next stage

Phase 3: closed-wall MAC layouts, ghost-cell no-slip treatment, Neumann pressure solve, cavity Re=100, steady-state and centreline validation. Cylinder and ROM remain later phases.

Raw results, snapshots, runtime metadata, Python figures/GIF and ParaView files are included. Original face DOFs remain in snapshots.npz; visualization interpolation does not replace diagnostics.
