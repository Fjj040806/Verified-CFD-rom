# PINN extension - Phase 0 research protocol

Date: 26 October 2026. Status: design and analytic illustration complete; no PINN has been trained. Baseline repository commit: `25f9dca3b068b92b9c3d904001ff9978f2f7ab10`.

Working title: **Physics-Informed Reduced-Order Modeling under Sparse Observations: Truncation Error, Sensor Information, and Prediction Reliability**.

## 1. Locked question and claim boundary

For a fixed, verified Re100 cylinder-wake discretization and a declared offline POD basis, when does a reduced physics constraint improve spatial reconstruction from sparse velocity observations, and when does unmodeled truncation bias outweigh that improvement?

Primary task: reconstruct unobserved velocity information within an observation window. Secondary diagnostics: time-coordinate-network extrapolation and physics-model rollout initialized from estimated states. These are different tasks. A time-to-coefficient PINN is a solution-function approximator, not automatically a learned vector field or a neural ODE.

Known viscosity comes first. Unknown-viscosity identification is a later gated extension. Cross-Re transfer, superiority to Galerkin, and a new algorithm are not claimed. The old 14-page manuscript remains a record of the completed CFD/POD/DMD study; these plans are not inserted as new empirical results.

## 2. Prior work and originality register

| Source and review scope | Established overlap | Decision for this extension |
|---|---|---|
| Hijazi, Freitag and Landwehr (2023), DOI 10.1186/s40323-023-00242-2; abstract and introductory methodological discussion checked | POD-Galerkin PINNs, Navier-Stokes inverse problems, viscosity estimation, and cylinder examples; truncated-mode effects are discussed | Do not claim POD+PINN, viscosity inversion, or recognizing truncation as new |
| Krishnapriyan et al. (NeurIPS 2021), arXiv:2109.01050; proceedings abstract checked | Physics penalties can create difficult optimization, with convection/reaction/diffusion examples | Separate optimization failure from model discrepancy; no assumption that larger physics weight helps |
| Kaya et al. (2026), arXiv:2603.01998v1; abstract checked, full method comparison pending | VMS-based ROM closure modeled through PINNs | Do not claim neural ROM closure itself as new; review this method before designing a closure module |

Sources:
- https://link.springer.com/article/10.1186/s40323-023-00242-2
- https://papers.neurips.cc/paper_files/paper/2021/hash/df438e5206f31600e6ae4af72f2725f1-Abstract.html
- https://arxiv.org/abs/2603.01998

Candidate H1: map the interaction of modal rank, measured closure discrepancy, sensor conditioning, and physics weight under a fixed numerical reference. Status: a controlled empirical question; originality unestablished.

Candidate H2: test whether discretization mismatch biases viscosity inference differently under informative and poorly conditioned sensor layouts. Status: later-stage candidate; needs independent parameter cases and a deeper prior-art comparison.

Negative results remain valid outputs. Do not change the benchmark or tune against evaluation fields until PINN appears superior. This bounded review is a positioning check, not an exhaustive novelty audit.

## 3. Information budget and data split

The existing dataset contains 301 float64 P2 velocity snapshots over 27-30, sampled at 0.01; the solver time step is 0.005. Geometry, mesh, full velocity ordering, mass matrix, and weak divergence must remain consistent.

| Information | Time interval | Permitted use |
|---|---|---|
| Offline complete CFD fields | 27.00-28.79; 180 snapshots | Compute training mean, mass-weighted POD, coefficient scales, offline DMD, and offline diagnostics |
| Online sparse observations | 28.80-29.39; 60 snapshot times available | Fit solution coefficients or infer reduced initial state; use only declared sensor components/times |
| Complete online fields | 28.80-29.39 | Evaluation-only mass errors and closure diagnostics after model configurations are frozen |
| Future complete fields | 29.40-30.00; 61 snapshots | Evaluation only; never used for stopping, loss weighting, architecture choice, or sensor selection |

The offline basis encodes dense CFD information and is supplied to every reduced model. The experiment is **conditional on an offline full-field basis**, not learning a flow from a few sensors without a prior. These windows were already examined in the previous study; the extension is an exploratory secondary analysis, not a fresh blind or statistically independent test. A confirmatory claim later needs new data.

Distinguish 8 sensor locations from 16 scalar velocity channels when both components are measured. Record scalar observation count, measured times, noise realization, and offline cost separately. Do not compare a sparse-observation PINN with a baseline given full online coefficient truth, or conversely with a baseline denied the offline basis.

## 4. Reduced equations derived from the existing FEM backend

Write the semidiscrete equations on the original coefficient ordering as

$$M\dot u+K(\nu)u+N(u)-B^Tp=f,\qquad Bu=0,$$

where `K(nu)=nu*K0`; the current solver attribute `A` already includes viscosity. Fix the offline mean `ubar` and basis `Phi`, and represent `u_r=ubar+Phi*a`. Keep the computed reduced mass `M_r=Phi^T M Phi` rather than silently replacing it by identity in floating point. Throughout this extension, projection coefficients are `a=M_r^{-1} Phi^T M (u-ubar)`. New DMD and other baselines use those same coordinates; the old `WeightedPOD.coefficients` output needs this mass correction, and an old fitted operator must not be silently mixed with the new convention.

Provided the modes have homogeneous prescribed-velocity trace and `B Phi=0`, they are admissible tests and

$$M_r\dot a=F_r(a;\nu),\qquad
F_r=\Phi^T[f-K(\nu)(\bar u+\Phi a)-N(\bar u+\Phi a)].$$

The pressure contribution is `(B Phi)^T p`, which vanishes under that hypothesis. This is weak discrete pressure elimination, not pointwise incompressibility. Check the hypothesis numerically before using a pressure-free residual. If it fails, repair/verify the basis or explicitly model pressure; do not simply drop the term.

For quadratic convection, write

$$F_r(a;\nu)=c(\nu)+L(\nu)a+Q(a,a),$$

$$c=\Phi^T[f-K\bar u-N(\bar u)],\qquad
L_{ij}=-k(\phi_j,\phi_i)-c_{adv}(\bar u;\phi_j,\phi_i)-c_{adv}(\phi_j;\bar u,\phi_i),$$

$$Q_{ijk}=-c_{adv}(\phi_j;\phi_k,\phi_i),\qquad
c_{adv}(w;v,z)=\int_\Omega ((w\cdot\nabla)v)\cdot z\,d\Omega.$$

`k(w,v)=nu*integral(grad(w):grad(v))`. The finite-window mean is not assumed steady, so the constant vector `c` is retained. Direct full-order and tensor evaluations must agree on the same quadrature before training. Existing natural outlet and fixed boundary conventions are retained.

## 5. Network, scaling, and observation operator

Let `tau=(t-t_A)/T_A` on the online window, with `t_A=28.8`, `T_A=0.59`. Define `a_theta(t)=D z_theta(tau)`, where `D_jj=max(s_j,1e-3*s_1)` and `s_j` is the offline standard deviation of POD coefficient j. No online complete fields enter these scales.

The dimensionless reduced residual is

$$r_z=\frac{d z_\theta}{d\tau}-T_A D^{-1}M_r^{-1}F_r(Dz_\theta;\nu).$$

The factor `T_A` is required by the chain rule. Use automatic differentiation for the time derivative. A coefficient-function PINN cannot be assumed to extrapolate causally merely because it is queried at a later time.

For sparse velocity samples, `y_k=H u(t_k)+epsilon_k`. Construct `H` using exact P2 finite-element point evaluation, not rounded mesh vertices or P1 display interpolation. Its row order is declared as `u(sensor0), v(sensor0), u(sensor1), v(sensor1), ...` and must be checked against analytic vector fields.

The first loss is

$$L=L_d+\lambda_f L_f,\quad
L_d=\frac{\sum_k\|H(\bar u+\Phi a_\theta(t_k))-y_k\|_2^2}{N_{scalar}\,U_{mean}^2},\quad
L_f=\frac{1}{rN_c}\sum_j\|r_z(\tau_j)\|_2^2.$$

Here `U_mean=1`. There is **no full-state initial-condition label** in the primary experiment: initial coefficients are unknown and inferred from the same sparse data. Do not impose a hidden exact initial state. No physics collocation point lies beyond the online observation window in the extrapolation diagnostic. New constraints and loss terms require a recorded protocol revision.

## 6. Sensor design and identifiability

Pilot locations, in domain coordinates:

`(0.35,0.12), (0.35,0.29), (0.65,0.12), (0.65,0.29), (1.00,0.12), (1.00,0.29), (1.40,0.12), (1.40,0.29)`.

These are planned locations, not an optimized layout or an observability result. Confirm that every probe lies in the saved fluid mesh and away from boundaries. Hidden diagnostic probes use x in `{0.50,0.85,1.20,1.70}` and y in `{0.08,0.16,0.25,0.33}`, giving 16 locations disjoint from the measured set.

At one time, `H Phi` maps coefficients to observations. Report its singular values, rank, and conditioning. For local reduced dynamics `dot a=g(a)` with linearized state-transition matrix `Psi(t,t0)`, temporal information is represented by the stacked map

$$\mathcal{O}=\begin{bmatrix}H\Phi\Psi(t_0,t_0)\\H\Phi\Psi(t_1,t_0)\\\vdots\end{bmatrix}.$$

Its numerical conditioning is a diagnostic, not a global observability or nonlinear parameter-identifiability theorem. Do not select sensors by future reconstruction errors. Optimized layouts, if later added, use only offline information and have their design cost reported.

## 7. Baselines with the same online information

1. Gappy POD: per-observation-time least squares/ridge using `H Phi` and the same mean. Record rank deficiency; regularization is fixed or calibrated from offline data only. Native gappy POD does not define time extrapolation, so do not invent a forecast metric without specifying an added model.
2. Data-only time-to-coefficient MLP: identical architecture, normalization, seeds, observation loss, optimizer, and update budget; set physics weight to zero.
3. POD-Galerkin state assimilation: optimize its initial coefficients over the online window using only the same sensor observations, then integrate its known reduced equations. This directly tests whether a neural representation adds value beyond the available ODE.
4. Affine DMD with sensor-based state estimation: fit dynamics only on the offline complete fields; infer the initial reduced state from online sparse observations and propagate without future truth resets.
5. Mean-flow baseline. Full-field POD projection is an offline/evaluation representation floor, not a sparse-data competitor.

For secondary prediction, report separately (a) native time-network extrapolation, (b) native DMD/Galerkin evolution, and optionally (c) the same Galerkin rollout initialized from each estimated final online state. The common-rollout comparison isolates state-estimation quality; it is not the same as comparing native forecasting algorithms.

## 8. Truncation discrepancy: the central control

For evaluation-only complete states, set `a*=M_r^{-1} Phi^T M (u-ubar)` and `u_r*=ubar+Phi*a*` to account for finite-precision nonidentity of `M_r`. Define the instantaneous spatial closure indicator

$$d_r(t)=M_r^{-1}\Phi^T\{\mathcal{F}(u(t))-\mathcal{F}(u_r^*(t))\},\qquad
\mathcal{F}(u)=f-Ku-N(u).$$

Under the pressure-elimination assumptions, this is the unresolved spatial contribution in the corresponding semidiscrete projected dynamics. Computing the difference of assembled operators avoids differentiating snapshots. It is not, by itself, a measurement of total continuous-physics discrepancy: BDF2 time error, quadrature/geometry error, and solver tolerance remain separate.

Only after a configuration is frozen may full online fields be used to explain its discrepancy. Do not feed `d_r` computed from evaluation truth into the PINN or select physics weights with it. Offline closure diagnostics may guide a clearly labeled revised development protocol.

A small projected residual can coexist with a poor full field. Report `L_f`, mass-weighted field errors, sensor/hidden-probe errors, and an error/closure discrepancy diagnostic together; do not turn a residual into a claimed solution-error bound.

## 9. First pilot and later controlled matrix

The initial pilot is deliberately small: rank 10; 8 locations measuring both components at all 60 online times; noiseless observations; physics weights `{0,0.1,1}`; paired seeds `{0,1,2}`. This is 9 MLP/PINN fits plus the nonneural baselines, with no choice of winner on future truth. Collocation uses 128 fixed equally spaced points on `[0,1]`. The candidate network has 3 hidden layers of width 32 with tanh, float64; Adam, learning rate 0.001, 2000 updates, fixed final iterate. Record actual wall time and all finite-loss/gradient failures. Torch installation and AD implementation are Phase 1/2 work, not completed Phase 0 work.

Pilot settings are a starting diagnostic budget, not a tuned or guaranteed-converged recipe. Any training-budget extension applies equally to paired zero/positive-weight networks and is recorded before rerunning comparisons. No full-field evaluation data are used for early stopping. Stalled optimization is a reported outcome, not silently discarded.

After operator and pilot gates pass, separately vary rank `{6,10,20}`, noise `{0,1% of U_mean}`, sensor budget/layout, and physics weight. Keep measurement counts and noise realizations paired across methods. Report seed ranges/medians; three optimization seeds are not independent physical trajectories and are not strong statistical confidence evidence.

## 10. Metrics, figures, and software

Primary: complete-field mass norm on the online window, fluctuation and total-field errors, sensor RMSE, hidden-probe RMSE, retained-space representation error, residual norms, training/runtime cost, and sensor-map conditioning. Point samples have zero continuum area; do not fabricate an off-sensor mass norm by arbitrarily zeroing rows of M. Use full-domain mass error plus disjoint hidden probes.

Secondary: future errors with zero future-data/collocation access, frequency/phase, stable-rollout duration, and the distinction between field accuracy and physics residual. Later inverse work reports parameter bias and identifiability, not just a single estimated value.

Python/Matplotlib: principle diagram, sensor locations, error versus physics weight/rank/noise, conditioning diagnostics, and consistent-scale computed/reconstructed fields. Actual flow animations come only after training. ParaView remains optional for inspecting mesh/field exports and is unnecessary for the initial low-dimensional plots.

## 11. Stage gates and remaining distance

| Phase | Deliverable | Gate |
|---|---|---|
| 0 - current | Positioning, equations, information budget, pilot configuration, principle diagram, analytic counterexample | Documentation/config/diagram consistency checked; no trained-model claims |
| 1 | Reconstruct saved FEM ordering/operators, POD audit, exact probe matrix, tensor and closure checks | Saved/reassembled M and coordinate ordering agree; boundary/weak-continuity and AD/normalization tests pass |
| 2 | Nine-fit pilot plus gappy, Galerkin, DMD, and mean controls | Finite training runs; equal-information evaluation; model discrepancy separated from optimization failure |
| 3 | Rank, noise, sensor-layout, and physics-weight controls | Paired budgets; all failures and negative results retained |
| 4 | Numerical-resolution checks and focused interpretation | Effects distinguished from time/mesh/basis uncertainty; prior-art comparison updated |
| 5 - conditional | Unknown-viscosity identification and new held-out parameter trajectories | Avoid true-parameter basis leakage and same-discretization inverse-crime claims |
| 6 | Revised academic manuscript and graphical release | Claims reflect measured results and unresolved limitations |

The new extension has completed its design checkpoint only. Operators, PINN training, controlled evidence, and the revised empirical paper remain. The analytic illustration below verifies a mathematical mechanism, not PINN performance on the cylinder.

## 12. Analytic illustration supplied at Phase 0

Consider the three-state system `xdot=-x+y`, `ydot=-z`, `zdot=y`, initialized at `(1/2,1,0)`. Its periodic solution has `y=cos(t)`, `z=sin(t)`, and `x*=(cos(t)+sin(t))/2`. Truncating the y/z coordinates leaves the misspecified scalar model `xdot=-x`, omitting a time-varying closure with zero cycle mean.

Within the restricted trial family `x_theta=alpha*x*`, normalize the observation and reduced-physics losses over one cycle as `L_d=4*mean((x_theta-x*)^2)` and `L_f=2*mean((xdot_theta+x_theta)^2)`. Then

$$J(alpha)=(alpha-1)^2+lambda\,alpha^2,\qquad alpha^*=\frac{1}{1+lambda}.$$

The normalized data error grows while the reduced residual shrinks as weight increases. This is a coordinate-truncation illustration, not POD learned from this trajectory, a CFD experiment, a trained network, or a universal PINN failure curve. It motivates measuring the competing errors in the cylinder study. Reproduce it and the principle diagram with `python extensions/pinn/plot_phase0.py` after installing the core plotting dependencies.
