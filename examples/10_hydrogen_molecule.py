"""10 – Hydrogen Molecule (H₂)

When two hydrogen atoms get close their χ-wells overlap.  The shared
well is deeper than either alone → the system lowers its energy by
bonding.  This is a covalent bond emerging from GOV-01 + GOV-02.

We use FieldLevel.COMPLEX here because we need both atoms to have
imaginary-component waves.  In a real H₂ molecule the bonding orbital
is a symmetric superposition of atomic orbitals — represented here as
two solitons with the same phase.  An anti-bonding configuration uses
opposite phases (π shift) and should NOT bind (energy goes up).

     Same phase (Δθ=0):     → BONDING    (shared χ-well deepens)
     Opposite phase (Δθ=π): → ANTI-BONDING (shared well shallows)

No molecular orbital theory.  No Schrödinger.  No bond-energy formula.
Just two wave fields and the two governing equations.

Run:
    python examples/10_hydrogen_molecule.py
"""

import numpy as np

import lfm

N = 64
config = lfm.SimulationConfig(grid_size=N, field_level=lfm.FieldLevel.COMPLEX)


def make_h2(bond_half: int, phase_a: float = 0.0, phase_b: float = 0.0):
    """Create two H-atom solitons separated by 2*bond_half cells."""
    s = lfm.Simulation(config)
    cx = N // 2
    s.place_soliton((cx - bond_half, N // 2, N // 2), amplitude=10.0, sigma=2.0, phase=phase_a)
    s.place_soliton((cx + bond_half, N // 2, N // 2), amplitude=10.0, sigma=2.0, phase=phase_b)
    s.place_soliton((cx - bond_half + 2, N // 2, N // 2), amplitude=0.9, sigma=1.5, phase=phase_a)
    s.place_soliton((cx + bond_half - 2, N // 2, N // 2), amplitude=0.9, sigma=1.5, phase=phase_b)
    s.equilibrate()
    return s


print("10 – Hydrogen Molecule (H₂)")
print("=" * 60)
print()

# ─── Experiment A: Bonding (same phase) ────────────────────────────────────
sim_bond = make_h2(bond_half=8, phase_a=0.0, phase_b=0.0)
sim_anti = make_h2(bond_half=8, phase_a=0.0, phase_b=np.pi)

m_bond_0 = sim_bond.metrics()
m_anti_0 = sim_anti.metrics()

print("Initial state (separation = 16 cells):")
print(f"  Bonding    χ_min = {m_bond_0['chi_min']:.3f},  energy = {m_bond_0['energy_total']:.2e}")
print(f"  Anti-bond  χ_min = {m_anti_0['chi_min']:.3f},  energy = {m_anti_0['energy_total']:.2e}")
print()

STEPS = 6000
sim_bond.run(steps=STEPS)
sim_anti.run(steps=STEPS)

m_bond_f = sim_bond.metrics()
m_anti_f = sim_anti.metrics()

psi_sq_b = (
    sim_bond.psi_real**2
    + (sim_bond.psi_imag if sim_bond.psi_imag is not None else np.zeros_like(sim_bond.psi_real))
    ** 2
)
sep_b = lfm.measure_separation(psi_sq_b)

psi_sq_a = (
    sim_anti.psi_real**2
    + (sim_anti.psi_imag if sim_anti.psi_imag is not None else np.zeros_like(sim_anti.psi_real))
    ** 2
)
sep_a = lfm.measure_separation(psi_sq_a)

print(f"After {STEPS} steps:")
header = f"  {'config':>10s}  {'χ_min':>7s}  {'δχ_min':>8s}  {'energy':>12s}  {'sep':>8s}"
print(header)
print(f"  {'-' * 10}  {'-' * 7}  {'-' * 8}  {'-' * 12}  {'-' * 8}")
for label, m0, mf, sep in [
    ("bonding", m_bond_0, m_bond_f, sep_b),
    ("anti-bond", m_anti_0, m_anti_f, sep_a),
]:
    delta_chi = mf["chi_min"] - m0["chi_min"]
    print(
        f"  {label:>10s}  {mf['chi_min']:7.3f}  {delta_chi:+8.3f}  "
        f"{mf['energy_total']:12.2e}  {sep:8.1f}"
    )
print()

bond_chi = m_bond_f["chi_min"] - m_bond_0["chi_min"]
anti_chi = m_anti_f["chi_min"] - m_anti_0["chi_min"]
bond_proxy = bond_chi - anti_chi
print("Bond formation diagnostic:")
if bond_proxy < -0.05:
    print(f"  Δ(χ_min)_bond - Δ(χ_min)_anti = {bond_proxy:.3f}  →  BOND FORMED")
    print("  Bonding deepens the shared well; anti-bonding shallows it.")
    print("  This is a covalent bond from wave interference — no MO theory used.")
else:
    print(f"  Δ(χ_min) difference = {bond_proxy:.3f}  (increase amplitude for clearer signal)")
print()

# ─── Bond-length scan: minimum energy separation ───────────────────────────
print("Bond-length scan (bonding phase only):")
print(f"  {'sep (cells)':>12s}  {'χ_min':>7s}  {'energy':>12s}")
print(f"  {'-' * 12}  {'-' * 7}  {'-' * 12}")

for half_sep in [4, 6, 8, 10, 12, 14]:
    s = make_h2(bond_half=half_sep, phase_a=0.0, phase_b=0.0)
    s.run(steps=2000)
    m = s.metrics()
    print(f"  {2 * half_sep:>12d}  {m['chi_min']:7.3f}  {m['energy_total']:12.2e}")

print()
print("Look for the separation with the deepest χ_min and lowest energy.")
print("That is the equilibrium bond length — it emerged without any formula.")
