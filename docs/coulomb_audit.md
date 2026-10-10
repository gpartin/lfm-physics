# Coulomb claim audit and falsifiable dynamics protocol

## Scope and provenance

Audited public baseline: `e899b6c067a4b5ada46e7bf40a7e02842ba93943`.
Historical filenames and initial overlap measurements are retained. No governing
physics or particle construction has been changed by this audit.

| Calculation | What it measures | What it does not establish |
|---|---|---|
| Experiment 03 | Initial sum of squared field magnitude; no evolution | Energy, potential, force, charge, inverse-square scaling |
| Example 05 | Initial/final density-based separation | Acceleration, reciprocal force, force-distance law |
| Example 28 | Initial/final distance between chi minima | Stable particle identity, acceleration, signed charge |
| Historical validation test | chi minimum / initial norm | Interaction Hamiltonian or Coulomb potential |
| Experiment 14 | Local-density centroid acceleration minus isolated drift | Validated signed sources, inertial force, Maxwell closure |

For real Gaussian envelopes g1 and g2 with phases 0 and pi, the initial norm
difference is exactly `4 sum(g1*g2)`. In the unbounded continuum with
`g=A exp(-r^2/(2 sigma^2))`, this is
`4 A^2 pi^(3/2) sigma^3 exp(-d^2/(4 sigma^2))`.
This is Gaussian overlap, not an inverse-square force. Finite-grid and float32
rounding modify the numerical expression. Norm must not be called total energy.

Example 05's original `delta_opp < delta_same` is compatible with both pairs
approaching, both separating, or one of each. It does not alone show repulsion
and attraction. Well merging and deformation can also change separation.

## Research boundary

Primary evidence, inspected during this audit:

- [Maxwell Vacuum Completion V1.0](https://github.com/gpartin/Papers/blob/main/paper_experiments/electron_first_principles_2026/LFM_MAXWELL_VACUUM_COMPLETION_REPORT_V1_0.md),
  blob `895f560f5ec939c0cf392410578f0c5d9d82b12e`:
  `NO_EXTENSION_MAXWELL_VACUUM_COMPLETION_FAILS_TESTED_ENTRANCES`.
- [Effective Maxwell V8.4](https://github.com/gpartin/Papers/blob/main/paper_experiments/electron_first_principles_2026/LFM_COLOR_EFFECTIVE_MAXWELL_REPORT_V8_4.md),
  blob `ddcadca57855f02afa02b750ead43b1ebe39208e`:
  `EFFECTIVE_MAXWELL_TREE_LEVEL_BLOCKED`.

These reports permit a wave-carried connection/geometric twist at tested layers,
but do not establish effective Maxwell stiffness, independent Gauss/Ampere
closure, reciprocal signed charge, or two transverse radiative modes. They do
not exclude every nonlinear or nonperturbative mechanism.

The current CPU backend `lfm/core/backends/numpy_backend.py` evolves legacy
`Psi_tt = lap(Psi) - chi^2 Psi` and uses a quartic chi restoring term
`-4 lambda_self chi (chi^2-chi0^2)`. The supplied v39.0.2 formulation instead
has a complex-C3 z-dependent Psi nonlinearity and flat-octic chi restoring term.
Changing a field-level flag or preset cannot implement that revision. A current
claim requires a pinned authoritative specification, source mu definition,
rho=0 handling for z, full initial phase-space state, and a verified solver.
This audit deliberately does not fill those missing definitions by assumption.

## Runnable legacy probe

```bash
python -m experiments.quantitative.exp_03_coulomb_law
python -m experiments.quantitative.exp_14_coulomb_dynamics --output coulomb_probe.json
python -m pytest tests/test_coulomb_audit.py tests/validation/test_coulomb_law.py -m 'not slow'
```

Experiment 14 forces CPU evolution, uses four preparations (0,0), (pi,pi),
(0,pi), (pi,0), and fits `x=x0+v*t+a*t^2/2` for both local centroids.
It subtracts isolated-seed acceleration at the same positions. JSON retains
configuration, trajectories, norms, residuals, fitted uncertainties, and all
failed gates. q=+/-1 are **phase preparation labels only**.

The run always leaves signed-source, stability and convergence evidence unset.
A good scaling fit alone cannot elevate it to a validated Coulomb result.
The local norm is not inertial mass; acceleration is reported in lattice units,
not converted to a force or SI units. Equal and opposite accelerations are a
reciprocity screen only for equal independently verified inertial masses.
Fit standard errors describe residual noise, not systematic error or independent
samples. The default short run is a smoke probe, not a production certification.

## Predeclared quantitative gates for a genuine test

1. **Sources:** construct stable isolated objects with independently measured
   conserved signed charge, equal inertial mass, and matched structure. Establish
   that changing charge sign is physically distinct from a global phase shift.
   Global phase 0/pi alone does not supply this evidence. Measure charge magnitude
   and its conservation for all four sectors; include a neutral-source control.
2. **Dynamics:** use initially stationary phase-space states, measure each
   object's trajectory and momentum/force where the action supports it. Subtract
   isolated drift and phase-independent attraction. Reject merging, dissolution,
   large radiation loss, window sensitivity, and boundary-contaminated intervals.
   A potential route requires the full Hamiltonian and `U12-U1-U2+Uvac`, followed
   by its separation derivative, not a norm or minimum chi surrogate.
3. **Signed reciprocity:** like signs must repel and opposite signs attract;
   source interchange and simultaneous sign reversal preserve magnitudes.
   For unequal masses compare `m1*a1+m2*a2`, including field momentum flux.
   The current equal-mass screen tolerates at most 10% mismatch; resolved
   accelerations must exceed five fit standard errors.
4. **Scaling:** use at least four distinct distances covering at least a factor
   two, outside overlapping cores and well inside boundaries. Require slope
   `-2 +/-0.2` and log-fit R-squared >=0.98 in every sign sector. Also compare a
   shared coupling `F*r^2/(q1*q2)` across sectors, source amplitudes and distances;
   use held-out distances and compare Gaussian/exponential and other powers.
   The implementation also requires <=10% spread in unit-charge a*r^2 across
   distances and sectors. These screens are necessary, not sufficient for this full gate.
5. **Convergence:** repeat at dt/2 at identical physical times, larger domains,
   multiple axes/orientations, tracking radii and fit windows. Require acceleration
   and fitted coupling to agree within 10%, including systematic uncertainty;
   reject underresolved or noise-dominated results. The solver fixes dx=1, so
   continuum spatial refinement requires an explicitly supported rescaling or
   another solver. No such refinement is claimed here.
6. **Maxwell:** separately test autonomous Gauss constraint and Ampere closure,
   massless linear dispersion with exactly two transverse stable modes, action
   stiffness, and finite-size scaling. Coulomb-like motion alone does not pass
   these gates.

The adjudicator is an auditable screen for signed response, equal-mass
reciprocity, signal, power law, sign reversal/source swap, and explicit evidence
flags. Flags are supplied only after independent review. It does not implement
or substitute for source construction, field-momentum balance, charge-magnitude controls,
held-out discrimination or Maxwell validation. Even a screen PASS leaves
`coulomb_status=NOT_ESTABLISHED` until these additional checks are implemented. Additional physics must be
identified as an extension and cannot be called emergence from unchanged LFM.

## Validation performed (2026-10-10)

- Historical experiment 03 passes its initial interference control. Norm
  differences at d=4,6,8,10,12 are approximately 13881.33, 7964.46, 3659.08,
  1346.10, 396.53. Regression comparison agrees with the analytic Gaussian
  expression within relative tolerance 1e-5.
- CPU smoke probe: N=48, dt=0.01, 60 steps (time 0.6), stride=10, sigma=2,
  amplitude=6, d=10,14,18,22, all four phase preparations plus isolated controls.
  [Raw trajectories and results](coulomb_probe_smoke.json) retain provenance.
  Signed response, resolved signal, inverse-square and common coupling gates
  fail. Equal-mass acceleration reciprocity and sign-swap symmetry screens pass.
  Charge, object stability and convergence remain unestablished.
- Exploratory log slopes are approximately -3.419 (same phase) and -3.343
  (opposite phase), with R-squared 0.956 and 0.949. Because signal resolution
  fails, these are diagnostic fit outputs, not measured physical force exponents.
- Targeted regression/configuration/simulation suite: 66 passed, one historical
  slow chi-depth test deselected. The slow test was also run separately and passed
  (the combined audit/historical suite before final reporting edits: 15 passed).
  New files pass Ruff; edited examples compile; git diff has no whitespace errors.

**Demonstrated:** initial Gaussian interference. **Force-law evidence:** no
validated Coulomb-like result from this smoke run. **EM closure:** unresolved,
with the cited research layers blocked. Neither the short probe failure nor the
historical overlap pass adjudicates all possible LFM electromagnetic mechanisms.
