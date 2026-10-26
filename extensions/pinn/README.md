# Physics-informed reduced-order modeling extension

**Current checkpoint: Phase 0 completed. No PINN has been trained.**

Question: when do reduced physics constraints improve sparse flow reconstruction, and when does truncation discrepancy make the constraint misleading?

- [Research and mathematical protocol](phase0-protocol.md)
- [Predeclared pilot configuration](pilot-config.json)
- [Configuration and analytic check results](phase0-checks.json)
- [Next-stage acceptance gates](phase1-gates.md)

![Planned sensors, information split, and reduced-physics loss](figures/phase0-principle.png)

![Analytic illustration of a misspecified reduced physics constraint](figures/phase0-analytic-mismatch.png)

The second figure is a closed-form three-state coordinate-truncation example within a restricted one-parameter trial family. It is not a learned result or empirical evidence for the cylinder wake. Its exact normalization and solution are recorded in the protocol and JSON data.

Reproduce the diagrams and checks from the repository root:

```bash
python extensions/pinn/plot_phase0.py
```

Only the existing NumPy/Matplotlib dependencies are needed. PNG, SVG, and vector PDF figures are generated. Torch training, FEM sensor probing, and intrusive reduced-operator assembly belong to the next stages.

The original CFD/POD/DMD study and its manuscript retain their existing claims. The extension will update the empirical paper only after new models and fair controls are actually evaluated.
