"""Measure actual chi gradient and compare to what rotation_curve expects."""
import numpy as np
import lfm
from lfm import CelestialBody, BodyType, place_bodies, BoundaryType, FieldLevel

bodies = [
    CelestialBody("Sun", BodyType.STAR, 0.15, 0),
]

cfg = lfm.SimulationConfig(
    grid_size=128, field_level=FieldLevel.REAL,
    boundary_type=lfm.BoundaryType.FROZEN, dt=0.02
)
sim = lfm.Simulation(cfg)
omegas = place_bodies(sim, bodies, verbose=False)

N = 128
cx = cy = cz = N // 2

chi_arr = np.array(sim.chi, dtype=np.float64)
psi = np.array(sim._evolver.get_psi_real(), dtype=np.float64)
psi2 = psi**2

print("Sun amplitude:", psi.max())
print("Chi min:", chi_arr.min(), "  delta:", chi_arr.min() - 19.0)
print()

# Measure chi profile along +x axis
print("Chi profile along +x axis:")
print("r    chi          delta_chi   expected_delta (Poisson)")
amp = psi.max()
sigma = 2.5
kappa = 1/63

# Analytical Poisson solution for Gaussian source
# delta_chi(r) = -kappa * amp^2 * sigma^2 * erfc(r/(sqrt(2)*sigma))... complicated
# For r >> sigma: delta_chi(r) ≈ -kappa * M / (4*pi*r)
# where M = amp^2 * (2*pi*sigma^2)^(3/2)
M_total = amp**2 * (2 * np.pi * sigma**2)**1.5
print(f"Total |Psi|^2 mass M = {M_total:.1f}")
print()

for r in [1, 2, 3, 5, 8, 10, 12, 15, 18, 20, 25]:
    chi_val = chi_arr[cx + r, cy, cz]
    delta_actual = chi_val - 19.0
    # Poisson prediction for r > sigma (use full 3D integral numerically)
    # For large r: delta_chi ≈ -kappa*M/(4*pi*r)
    delta_poisson_asymptotic = -kappa * M_total / (4 * np.pi * r)
    print(f"r={r:3d}: chi={chi_val:.6f}  delta={delta_actual:+.6f}  theory={delta_poisson_asymptotic:+.6f}")

print()
# Compute dchi/dr numerically at r=12
r12 = chi_arr[cx + 12, cy, cz]
r11 = chi_arr[cx + 11, cy, cz]
r13 = chi_arr[cx + 13, cy, cz]
dchi_dr_num = (r13 - r11) / 2.0
print(f"dchi/dr at r=12 (numerical): {dchi_dr_num:.6f}")

# Expected from Poisson: kappa * M / (4*pi * 12^2)
dchi_dr_theory = kappa * M_total / (4 * np.pi * 12**2)
print(f"dchi/dr at r=12 (theory):    {dchi_dr_theory:.6f}")
print(f"ratio theory/numerical: {dchi_dr_theory/dchi_dr_num:.2f}")

print()
v_circ_num = np.sqrt(12 * (1/19) * abs(dchi_dr_num))
v_circ_theory = np.sqrt(12 * (1/19) * dchi_dr_theory)
print(f"v_circ from numerical chi gradient: {v_circ_num:.5f}c")
print(f"v_circ from Poisson theory: {v_circ_theory:.5f}c")
print(f"v_circ from rotation_curve: 0.05610c (measured)")
