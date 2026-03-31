"""Measure chi anisotropy to diagnose orbital instability root cause."""
import numpy as np
import lfm
from lfm import CelestialBody, BodyType, place_bodies, BoundaryType, FieldLevel

bodies = [CelestialBody("Sun", BodyType.STAR, 0.15, 0)]
cfg = lfm.SimulationConfig(grid_size=128, field_level=FieldLevel.REAL,
    boundary_type=BoundaryType.FROZEN, dt=0.02)
sim = lfm.Simulation(cfg)
place_bodies(sim, bodies, verbose=False)

N = 128
cx = cy = cz = N // 2
chi = np.array(sim.chi, dtype=np.float64)

print("=== Chi profile along different directions from center ===")
print("r   chi(+x)      chi(+y)      chi(+xy)     chi(+xyz)")
for r in [4, 6, 8, 10, 12, 14, 16, 18, 20, 25]:
    ix = cx + r if cx+r < N else N-1
    iy = cy + r if cy+r < N else N-1
    r_diag = int(r / np.sqrt(2))
    r_diag3 = int(r / np.sqrt(3))
    chi_x = chi[cx+r, cy, cz] if cx+r < N else 19.0
    chi_y = chi[cx, cy+r, cz] if cy+r < N else 19.0
    chi_xy = chi[cx+r_diag, cy+r_diag, cz] 
    chi_xyz = chi[cx+r_diag3, cy+r_diag3, cz+r_diag3]
    print(f"r={r:3d}: {chi_x:.7f}  {chi_y:.7f}  {chi_xy:.7f}  {chi_xyz:.7f}")

print()
print("=== Chi gradient magnitude at r=12 in different directions ===")
# x-axis: central difference
dcdr_x = (chi[cx+13,cy,cz] - chi[cx+11,cy,cz]) / 2
# y-axis
dcdr_y = (chi[cx,cy+13,cz] - chi[cx,cy+11,cz]) / 2
# diagonal xy
d12x = int(np.round(12/np.sqrt(2)))
dcdr_xy = (chi[cx+d12x+1, cy+d12x+1, cz] - chi[cx+d12x-1, cy+d12x-1, cz]) / (2*np.sqrt(2))
print(f"dchi/dr at r=12 along +x:     {dcdr_x:.7f}")
print(f"dchi/dr at r=12 along +y:     {dcdr_y:.7f}")
print(f"dchi/dr at r=12 along +xy:    {dcdr_xy:.7f}")
print(f"ratio x/xy: {dcdr_x/dcdr_xy:.3f}")
print()
v_x = np.sqrt(12 * (1/19) * abs(dcdr_x))
v_y = np.sqrt(12 * (1/19) * abs(dcdr_y))
v_xy = np.sqrt(12 * (1/19) * abs(dcdr_xy))
print(f"v_circ at r=12 along +x:  {v_x:.5f}c")
print(f"v_circ at r=12 along +y:  {v_y:.5f}c")
print(f"v_circ at r=12 along +xy: {v_xy:.5f}c")
print()
print("rotation_curve v_circ:    0.05610c (from place_bodies)")
print("Mercury launch speed:    ", 0.85*0.05610, "c (v_scale=0.85)")
print()

# Also check if chi is evolving at step 0
print("=== Chi stability: run 100 steps and check chi at r=12 ===")
chi_r12_before = chi[cx+12, cy, cz]
sim.run(100, record_metrics=False)
chi2 = np.array(sim.chi)
chi_r12_after = chi2[cx+12, cy, cz]
print(f"chi at r=12: before={chi_r12_before:.7f}  after={chi_r12_after:.7f}  change={chi_r12_after-chi_r12_before:.7f}")
chi_r0_before = chi[cx, cy, cz]
chi_r0_after = chi2[cx, cy, cz]
print(f"chi at r=0:  before={chi_r0_before:.7f}  after={chi_r0_after:.7f}  change={chi_r0_after-chi_r0_before:.7f}")
