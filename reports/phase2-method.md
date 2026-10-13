# Phase 2 numerical method and limitations

Periodic, first-order incremental pressure correction:
(I-dt nu L) u_star = u_n-dt N(u_n)-dt G p_n.
L phi = D u_star/dt; u_new=u_star-dt G phi; p_new=p_n+phi.
Diffusion solves use FFT diagonalization of the discrete Laplacian. Convection
is explicit centred advective form. Cross-component velocities use four-point
interpolation at the target face. The energy behaviour of this convection
form is not generally conservative; the present TG checks do not prove a
universal energy-stability property.

Scope: nu=.1, periodic [0,2pi]^2, smooth TG and translated TG. Configured CFL
limit .5 is an operational guard, NOT a proof of centred explicit-advection
stability. Do not use this method uncritically for high-Re general flows.

Temporal verification compares velocities against the exact semidiscrete
stationary TG amplitude exp(-nu lambda_h t), where
lambda_h=8 sin²(h/2)/h² on a square grid. For this special field the projected
nonlinear term cancels, so it isolates first-order implicit diffusion timing.
Translated TG, plus a separate convection-consistency test, exercises transport;
broader manufactured multiscale flows and second-order advancement remain
future verification extensions.

Spatial verification uses dt proportional to h² with dt halving at every
grid, to limit temporal contamination. The first exploratory fixed-dt test
had a finest-grid halving change near 10% of total error and was replaced by
this controlled experiment. Pressure must be mean-aligned and assessed
separately. Nonmonotonic pressure error under dt refinement can result from
opposing temporal and spatial error contributions; do not report anomalously
small error as higher accuracy without controlled refinement.

Candidate research question: interaction of pressure splitting error, Poisson
tolerance and downstream mode extraction. Error cancellation observed here
is a diagnostic lead, not a discovery or evidence of publishable novelty.
No POD/DMD results exist yet.

Sources: projection-method comparison https://dealii.org/developer/doxygen/deal.II/step_35.html
VTK output format https://docs.vtk.org/en/v9.6.1/vtk_file_formats/vtkxml_file_format.html
