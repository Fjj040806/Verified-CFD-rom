# Scope, attribution and release status

Version 1.0 completes the implemented single-Re wake analysis and scientific-computing demonstration. It is not a completed original-method paper.

The project contains independently specified MAC operators, finite-difference solvers, verification experiments, FEM weak forms and boundary choices, nonlinear stepping logic, force diagnostics, ROM analysis and plotting scripts. Gmsh generates meshes and scikit-fem supplies finite-element bases and assembly infrastructure. SciPy supplies linear algebra and transforms. This is not a from-scratch FEM library; FEniCSx was considered in the design but is not the implemented backend.

Established methods used here include pressure correction, Taylor-Hood elements, BDF2, Newton iteration, weighted POD and affine DMD. Official DFG cylinder benchmarks supply reference values. Source links are recorded in the final/stage reports; the downloaded full benchmark publication is not redistributed.

Results produced during this collaboration should be reviewed and understood by the project owner before use in an application. Descriptions of personal contributions should reflect the owner's actual work and understanding, and should distinguish assisted implementation from independent derivation.

No author name, institutional affiliation, paper acceptance, or external repository URL is asserted. The repository preserves the license supplied by its owner in the initial commit. Dependency licenses remain separate.

Uncompleted research extensions: cross-Re prediction, cross-mesh mode transfer, a global spatial-refinement study, long-horizon robustness, and novelty assessment beyond the initial related-work review. These are future work, not missing pieces required to run the completed demonstration.
