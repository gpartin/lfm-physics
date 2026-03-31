"""Diagnose the tracking failure mode precisely.

The global argmax finds r=6 (near Sun edge) not r=12 (planet).
This means Sun field at r=6 > planet field at r=12.

Solutions:
1. Larger Sun mask radius
2. Track by finding the SECOND peak (ring search at expected radius)
3. Use chi-well position (planet creates a chi depression)
"""
import numpy as np, math, sys
sys.path.insert(0, '.')
import lfm
from lfm.analysis.observables import rotation_curve
from scipy.ndimage import gaussian_filter, label

N = 64
DT = 0.02

cfg = lfm.SimulationConfig(grid_size=N, field_level=lfm.FieldLevel.REAL,
                           boundary_type=lfm.BoundaryType.FROZEN, dt=DT)
sim = lfm.Simulation(cfg)
cx = cy = cz = N // 2

sim.place_soliton((cx, cy, cz), amplitude=7.0, sigma=2.5)
sim.equilibrate()

# Look at Sun field profile to decide mask radius
psi2_sun = np.array(sim.psi_real, dtype=np.float32) ** 2
for r in range(5, 16):
    # Sample values along +x axis at distance r
    if cx+r < N:
        val = psi2_sun[cx+r, cy, cz]
        print(f"  Sun psi2 at r={r}: {val:.6f}")

# Now place planet at r=12
rc = rotation_curve(sim.chi, sim.energy_density, center=(cx,cy,cz), plane_axis=2)
r_arr = np.asarray(rc['r']); v_arr = np.asarray(rc['v_chi'])
r_test = 12
v_raw = float(np.interp(r_test, r_arr, v_arr))
v_nyq = 0.8 * math.pi / 19 * 0.92
v_circ = min(v_raw * 0.85, v_nyq)
sim.place_soliton((cx + r_test, cy, cz), amplitude=0.3, sigma=1.5,
                  velocity=(0.0, v_circ, 0.0))
sim.equilibrate()

psi2_both = np.array(sim.psi_real, dtype=np.float32) ** 2
print(f"\nWith planet placed at r=12:")
# Profile along +x
for r in range(5, 20):
    if cx+r < N:
        val = psi2_both[cx+r, cy, cz]
        print(f"  psi2[r={r}] = {val:.6f}  {'<-- planet' if r==12 else ''}")

# What mask radius is needed?
# Find where Sun psi2 drops below planet's psi2 at r=12
planet_at_12 = psi2_both[cx+12, cy, cz]
print(f"\nPlanet psi2 at r=12: {planet_at_12:.6f}")
print(f"Need mask radius such that Sun psi2 < {planet_at_12:.6f}")
for r in range(5, 14):
    if cx+r < N:
        sun_val = psi2_sun[cx+r, cy, cz]
        planet_val = psi2_both[cx+r, cy, cz]
        need_mask = planet_val < planet_at_12 * 0.5  # Sun still dominates here
        print(f"  r={r}: sun={sun_val:.6f}  combined={planet_val:.6f}  mask_needed={need_mask}")
