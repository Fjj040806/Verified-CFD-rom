# Phase 4A — steady cylinder verification

Status: COMPLETE for DFG 2D-1, Re20. Phase 4B unsteady Re100 shedding is pending. No shedding or ROM claims are made.

## Geometry, boundaries and method

Channel [0,2.2] x [0,.41]; cylinder centre (.2,.2), radius .05. rho=1, nu=.001, parabolic inlet maximum .3, mean .2, Re20 based on diameter .1. Zero velocity on the cylinder and horizontal walls. Natural outlet nu*(grad u)n-pn=0; pressure is determined by this outlet, not pinned arbitrarily.

The earlier from-scratch MAC modules remain independent. New backend uses Gmsh for CAD/mesh generation and scikit-fem for P2 velocity / P1 pressure mixed finite-element assembly. Project code specifies the forms, boundary conditions, relaxed Picard iteration, residual diagnostics and force estimators. This is not a from-scratch FEM library or FEniCSx implementation.

Gmsh CAD is circular, but the calculation uses linear triangular geometry and polygonal boundary facets. P2 velocity does not mean curved geometry. Geometry error is included in refinement. Quadrature order 5 is used for all volume forms. The pressure coupling uses -B^T and B; no pressure penalty is introduced.

Picard under-relaxation .7; relative whole-state iterate change below 1e-9; a failed max-iteration check raises an error. Final free-equation residual is checked separately.

## Reference comparison

| Mesh | Triangles | Cylinder facets | CD | CD relative error | CL relative error | pressure difference relative error |
|---|---:|---:|---:|---:|---:|---:|
| coarse | 1970 | 35 | 5.56188336 | 0.3164% | 2.9759% | 1.4573% |
| medium | 4157 | 53 | 5.57185962 | 0.1376% | 0.6709% | 0.6451% |
| fine | 8700 | 79 | 5.57615435 | 0.0606% | 0.3556% | 0.3010% |

Official references: CD=5.57953523384, CL=.010618948146, dp=.11752016697. Front/rear pressure probes approach (.15,.2) and (.25,.2) from the fluid side with 1e-8 offsets. Reference website: https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html

## Force and incompressibility audit

Primary force estimator is the negative sum of uneliminated momentum residual entries on constrained cylinder velocity DOFs, component by component. The negative sign converts fluid-domain boundary traction to force on the body. Normalize by 2/(Umean² D)=500. Independent surface estimate integrates -(nu grad(u)n_fluid - p n_fluid) on polygonal cylinder facets. Both use the benchmark Laplacian/traction convention consistently.

| Mesh | Reaction CD | Surface CD | Strong divergence L2 | Weak divergence coefficient norm |
|---|---:|---:|---:|---:|
| coarse | 5.56188336 | 5.49852339 | 0.032990 | 9.981e-17 |
| medium | 5.57185962 | 5.54352023 | 0.017438 | 1.637e-16 |
| fine | 5.57615435 | 5.56015997 | 0.008923 | 1.548e-16 |

Fine surface-CD relative error: 0.3473%; surface-CL relative error: 2.5535%. Keep both estimates; do not cherry-pick the more accurate one without explaining the method.

Taylor-Hood satisfies continuity weakly. The small ||B u|| does NOT imply pointwise divergence near machine zero. Strong divergence decreases under refinement, but is still nonzero. Net boundary flux audit: inlet -.082, outlet .082 to roundoff; wall and cylinder flux near zero. Essential velocity boundary max error is zero at constrained DOFs. Fine free-equation relative residual is about 1.56e-10.

## Tests and visualization

All 12 tests passed in the configured environment, including fitted geometry perimeter, weak continuity, momentum residual and essential boundary constraints. Optional FEM test skips when dependencies are unavailable; such a skip is not a numerical pass. All three VTU exports passed native VTK reader cell-count checks.

Python/Matplotlib plots include mesh, speed, pressure, refinement of diagnostics and a dark blue/teal computed-flow view. Streamlines and VTU velocity use vertex interpolation for viewing; complete P2 coefficient arrays and DOF coordinates are retained in NPZ. VTU therefore does not preserve the full P2 field. No ParaView GUI inspection performed.

## Candidate research / limitations

Estimator sensitivity is measured: surface stress and discrete reaction converge differently, especially for small lift. A future controlled experiment should separate geometric refinement, solution degree and force postprocessing; follow with a focused prior-art review. Existing force methods and Taylor-Hood are not innovations. Pressure-tolerance-to-ROM study remains a separate hypothesis.

No formal spatial order is inferred from these three non-nested meshes. No transient stability, shedding frequency or force-history validation is supported by this steady solve. Before a final high-precision benchmark claim: extend geometry refinement, compare probe offset sensitivity and assess quadrature changes if differences justify it.

## Reproduction

Install requirements-cylinder.txt. Optional vtk is required for native export checks. Gmsh may need OS shared libraries, including libXft.so.2 on Linux. This execution used task-local extracted Ubuntu libXft/OpenGL dependencies through LD_LIBRARY_PATH because OS package installation was unavailable. Those binary libraries are not included as project source; install the equivalent supported OS packages when reproducing.

From project root:
```
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 python experiments/verify_phase4.py
PYTHONPATH=src python experiments/audit_phase4.py
PYTHONPATH=src python experiments/visualize_phase4.py
```

Backend versions: scikit-fem 12.0.2, gmsh 4.15.2, meshio 5.3.5.

## Next

Phase 4B: implement transient mixed FEM with explicit startup protocol, validate Re100 time-dependent shedding and temporal/grid sensitivities, then retain settled snapshots for Phase 5 POD/DMD. Steady initialization may be used only with documented inflow and perturbation choices matching the selected benchmark.
