# PR #4 compatibility boundary

This change repairs the backend call contract and optional visualization imports.
It does not change numerical evolution, constants, saved data, or adjudicated
research results.

## Backend contract

The NumPy and CuPy implementations and `Evolver._step` agree:

- `step_complex` accepts `dt2`, `epsilon_w`, and the keyword-only current,
  spatial-scale, and chi-floor switches. It does not accept `dt` or color/S_a
  extension arguments.
- `step_color` additionally accepts `kappa_c`, `epsilon_cc`, `kappa_string`,
  `kappa_tube`, S_a buffers, `sa_gamma`, `sa_d`, and `dt`.

The protocol now follows these actual contracts. Regression tests compare
parameter names, kinds, and defaults for all six production/recovery step
methods against both backend classes. CuPy signature inspection does not
instantiate a backend or claim CUDA execution/parity.

`gravity_recovery` follows the existing visualization convention: imports are
local to plotting calls and missing Matplotlib raises an error explaining the
`lfm-physics[viz]` extra. A fresh subprocess blocks all Matplotlib imports,
imports every visualization submodule, runs a CPU step, and checks the three
recovery plotting errors. Three additional tests render PNGs when the extra is
installed. CI also installs the package without visualization extras.

## Historical equations versus v39.0.2

The standalone `examples/LFM_coupled_energy_exchange.py` explicitly preserves
its v38 action-closed core: GOV-01 uses `-chi^2 Psi`, GOV-02 uses the
`N_2 - E0_sq` source and flat-octic self-force. Its Hamiltonian and output
remain historical energy-exchange diagnostics, not v39.0.2 evidence.

The v39.0.2 equations supplied for this review instead specify a complex-C3
wave register and real chi, with

```text
rho = sum_a |Psi_a|^2
sigma = sum_a Psi_a^2
z = sigma / rho
Psi_tt = c^2 Delta19 Psi - chi^2 (2 z conjugate(Psi) - |z|^2 Psi)
chi_tt = c^2 Delta19 chi - (kappa/chi0) chi mu
         - (8 lambda_H/chi0^4) chi (chi^2 - chi0^2)^3
```

These equations are recorded as user-supplied model requirements, not a claim
that this PR implements or validates them. `mu` is intentionally not guessed.
Before a separately versioned v39.0.2 implementation can be accepted, its
canonical source must define `mu` and the `rho = 0` prescription, derive a
consistent discrete Hamiltonian/current ledger, and supply CPU/GPU parity and
model-specific vacuum, polarization, conservation, and convergence tests.

The production complex/color backends retain historical onsite terms; even the
flat-octic recovery path calls those historical wave updates. Selecting a
flat-octic chi potential therefore does not establish v39.0.2 compatibility.
No existing result should be relabeled as a v39.0.2 result.

## PR lineage and unresolved checks

At inspection, PR #3 head `7833a1bf62866931238f03e072eb250aafe69c41` is the
immediate parent of PR #4 head `d1e95c49c72e5d5f9c11495d3a911b5cf55fd227`.
GitHub comparison reports ahead=1, behind=0; the only additional file is the
energy-exchange example. PR #4 supersedes #3; the latter adds no independent
feature set and can be closed without deleting its branch or merging it.

Remaining validation limits are tracked in the PR conversation. In particular,
signature agreement does not prove GPU numerical parity, and historical tests
do not demonstrate the newer model's physical claims. The PR must remain
unmerged until its required CI and compatibility checks are adjudicated.
