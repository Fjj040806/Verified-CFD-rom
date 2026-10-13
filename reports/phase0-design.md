# Phase 0 — Numerical design and research protocol

Status: design complete; no numerical solver results yet. Date: 2026-10-13.

## Objective and ownership
Build an independently implemented MAC-grid incompressible-flow solver, verify it, then study vortex shedding and low-dimensional representations. Cylinder calculations may use a separately identified fitted-mesh FEM backend. Do not describe third-party FEM assembly as a from-scratch implementation. This is a new, independent project.

## Equations and grid
Use nondimensional velocity u=(u,v), kinematic pressure p, viscosity nu=1/Re:
du/dt + N(u) = -grad(p) + nu Laplacian(u) + f; div(u)=0.
For uniform spacing hx,hy: pressure at cell centres (i+1/2,j+1/2); horizontal velocity on vertical faces (i,j+1/2); vertical velocity on horizontal faces (i+1/2,j). Storage without ghosts: p[Ny,Nx], u[Ny,Nx+1], v[Ny+1,Nx]. Add ghost layers explicitly for wall stencils; periodic boundary faces represent the same degree of freedom, not two independent values.

D(u,v)[j,i]=(u[j,i+1]-u[j,i])/hx+(v[j+1,i]-v[j,i])/hy.
Gx(phi)[j,i]=(phi[j,i]-phi[j,i-1])/hx on interior vertical faces, and analogously Gy on horizontal faces. Define Lp=D G using the SAME boundary conventions. Never assemble an unrelated pressure Laplacian.
Interior scalar Laplacian uses centred second differences. Interpolate cross-component velocities consistently to the target face for convection. Begin with centred advective convection for smooth, resolved validation cases; it is not generally energy conserving. Audit discrete energy production before choosing a skew-symmetric/flux formulation for production.

## First implementation: first-order incremental pressure correction
1. Predictor: (I-dt*nu*Lu) u_star = u_n + dt*(-N(u_n)-G p_n+f_n). Diffusion implicit; convection explicit.
2. Correction: Lp phi = D u_star/dt.
3. Velocity: u_new=u_star-dt*G phi.
4. Pressure: p_new=p_n+phi; remove its mean where the pressure gauge is free.
The predictor is first order in time. Do not claim second-order temporal accuracy. Later implement and verify BDF2/explicit extrapolation or an alternative second-order method, including startup and pressure boundary treatment.
Use conservative advective step selection dt <= C/(max|u|/hx+max|v|/hy), initially C=0.25 as a heuristic, not a stability theorem. Check by dt refinement. Implicit diffusion removes the explicit diffusive step restriction but not temporal accuracy requirements.

## Boundary and linear-system contract
Periodic validation: periodic operators, zero-mean correction RHS and pressure. Closed cavity: no penetration at faces; tangential no-slip via ghost values. Top lid u=1 except corner convention stated explicitly; other walls u=v=0. Corner discontinuities may limit global regularity, so cavity is NOT the main order-verification case.
Derive correction normal derivative from final normal velocity g: dphi/dn=(u_star dot n-g)/dt. When predictor satisfies g, this becomes homogeneous Neumann. This is a boundary condition for the CORRECTION, not a claim of physical zero normal pressure gradient. Apply ghost conditions consistently; indiscriminate velocity overwriting after projection can reintroduce divergence.
Pure Neumann/periodic Poisson systems have a constant nullspace. Measure RHS compatibility before removing roundoff-scale means; significant incompatibility is a boundary-flux bug, not something to hide. Work on the zero-mean subspace using the positive operator A=-Lp. For CG, use an SPD operator/preconditioner on that subspace; avoid asymmetric row pinning. At modest size use a constrained direct solve as a reference.
With residual r=b-Lp phi, corrected divergence obeys D u_new=dt*r when operators and boundary updates are consistent. Record relative AND absolute residuals, iterations, divergence, and the norm of D u_new-dt*r. A relative tolerance alone is misleading when b is small.

## Verification protocol
V1: periodic sine/cosine Poisson problem and manufactured face fields; verify D/G/L consistency and expected interior second-order truncation error.
V2: Taylor-Green vortex on [0,2pi]^2 with periodic boundaries:
u=sin(x)cos(y)exp(-2nu*t), v=-cos(x)sin(y)exp(-2nu*t),
p=(cos(2x)+cos(2y))*exp(-4nu*t)/4, f=0.
Kinetic energy averaged over the domain is K(t)=exp(-4nu*t)/4. Sample exact velocities at their own face coordinates; pressure at centres after gauge alignment.
V3: spatial grids N=16,32,64,128; choose sufficiently small dt and tight linear tolerance, and demonstrate temporal contamination is negligible with another dt halving. Time study at fixed fine grid with dt,dt/2,dt/4,dt/8 and report the spatial error floor. Observed order log(E_h/E_h/2)/log(2); weighted L2 and Linfinity errors for u,v,p separately.
V4: cavity Re=100 first; 32,64,128 grids, centreline velocity and steady-state change rate, divergence and mass flux. Advance until declared convergence; do not equate a preset final time with steady state. Match reference Reynolds number, coordinates and corner conventions.
Promotion gates: successful analytic verification; compatible Poisson RHS; projection residual identity to floating-point/linear-solve precision; stable refinement trends; documented energy behaviour. Investigate failed gates before expanding physics.

## Cylinder phase: gated design
Use the DFG cylinder benchmark definitions, including geometry, mean inlet speed, viscosity, outlet treatment and force conventions; never combine open-domain cylinder reference values with confined-channel results. Start with Re=20 steady and Re=100 periodic benchmark configurations after separately verifying each specification.
Cartesian immersed boundary is optional research, not the default promised backend. Conduct a small feasibility trial on boundary accuracy and forces. If inadequate, use fitted-mesh Taylor-Hood FEM via FEniCSx for the cylinder and retain the original MAC solver for analytic/cavity work. Backend comparisons require matched geometry and boundary conditions.
Quantities: drag/lift from surface traction with stated normal convention; pressure difference; shedding period after removing startup; St=f_s D/U_mean. Require several settled cycles and report frequency resolution and window sensitivity.

## ROM protocol
Train on settled snapshots only. Remove the training mean, use spatial quadrature-weighted velocity POD, and use the same grid/mask/weights at evaluation. Do not combine pressure and velocity without a declared scaling. Report eigenvalue energy, held-out relative velocity reconstruction error, and divergence of reconstructed fields. Energy fraction alone is insufficient.
DMD: uniform time sampling, truncation sensitivity, eigenvalue-to-frequency conversion, held-out time forecasting and comparison to the lift-spectrum frequency. Distinguish POD reconstruction, DMD forecasting, and a true dynamical reduced-order solver. No POD-Galerkin prediction claim unless that model is actually implemented and validated.
Use contiguous training/validation/test time windows; avoid randomly mixing adjacent snapshots. For cross-Re studies evaluate a separate parameter value with consistent domain and state mapping. Compare with mean-flow and persistence baselines.

## Innovation register: hypotheses, not discoveries
H1 (priority): How does pressure-solve tolerance propagate through incompressibility and energy errors into POD rank estimates and DMD frequencies? Controlled factorial tolerance x dt x grid; evaluate against tightly converged flow. Claim only if effects exceed discretization/window uncertainty. RHS scaling and absolute residual are essential controls.
H2: How do snapshot cadence and window length bias mode rank and shedding-frequency recovery at a fixed verified flow? Control total observation duration and avoid aliasing. Useful reproducibility study; novelty must be checked against literature.
H3 (optional): Divergence-preserving projection of reduced reconstructions vs unconstrained/approximate modes. Exact POD of identically discretized divergence-free snapshots already preserves linear divergence, so establish whether a nontrivial issue exists before developing a correction.
For every hypothesis: record prior art, proposed distinction, negative results, compute budget and go/no-go decision. Standard projection methods, POD and DMD themselves are not innovations. A publishability assessment requires a focused literature review and actual results.

## Plotting and output
Python/Matplotlib: MAC grid schematic, analytic validation field, convergence plots, energy/divergence histories, cavity profiles, force spectra and POD/DMD figures. Save vector PDF/SVG plus PNG with units, nondimensional conventions, time and parameters. Use common colour limits across comparable flow plots; symmetric limits for vorticity. Do not fabricate convergence curves.
ParaView: optional for larger meshes and FEM cylinder output via VTK/XDMF; not necessary for initial 2D work. Gmsh: fitted cylinder mesh generation if FEM route is used. Browser display later uses computed snapshots; avoid representing interpolation as a new PDE solution.
Current figures are DESIGN and ANALYTIC-reference figures, not solver output.

## Reproducibility and next step
Every experiment records configuration, software versions, commit when available, stopping criterion, wall-clock time and diagnostics. Test numerical contracts instead of merely mirroring implementation. Next authorized stage: Phase 1 MAC grid/operators/Poisson with analytic checks; no cylinder or ROM before the verification gates.

## Primary references
- Projection implementation and pressure boundary caveats: https://dealii.org/developer/doxygen/deal.II/step_35.html
- Pressure nullspace: https://petsc.org/release/tutorials/physics/guide_to_stokes/
- DFG benchmark definitions: https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow.html
- Re100 periodic cylinder benchmark: https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark2_re100.html

Sources checked during Phase 0. Dedicated ROM/novelty literature review remains pending; this report does not assert novel or validated results.
