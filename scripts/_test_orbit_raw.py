"""Diagnose: is the planet actually moving, or is the tracker stuck?
Print angle every 10 steps to see real orbital motion."""
import numpy as np, math, sys
sys.path.insert(0, '.')
import lfm
from lfm.analysis.observables import rotation_curve
from scipy.ndimage import gaussian_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = 64   # tiny grid for fast test
DT = 0.02

cfg = lfm.SimulationConfig(grid_size=N, field_level=lfm.FieldLevel.REAL,
                           boundary_type=lfm.BoundaryType.FROZEN, dt=DT)
sim = lfm.Simulation(cfg)
cx = cy = cz = N // 2

# Place Sun
sim.place_soliton((cx, cy, cz), amplitude=7.0, sigma=2.5)
sim.equilibrate()

# Measure velocity at r=12 (Mercury-like)
rc = rotation_curve(sim.chi, sim.energy_density, center=(cx,cy,cz), plane_axis=2)
r_arr = np.asarray(rc['r']); v_arr = np.asarray(rc['v_chi'])
r_test = 12
v_raw = float(np.interp(r_test, r_arr, v_arr))
v_nyq = 0.8 * math.pi / 19 * 0.92
v_circ = min(v_raw * 0.85, v_nyq)
omega = v_circ / r_test
T = int(2 * math.pi / omega)
print(f"r={r_test}: v_raw={v_raw:.4f} v_circ={v_circ:.5f} omega={omega:.5f} T={T} steps")
print(f"Expected: {T} steps/orbit, checking every 10 steps")

# Place planet
sim.place_soliton((cx + r_test, cy, cz), amplitude=0.3, sigma=1.5,
                  velocity=(0.0, v_circ, 0.0))
sim.equilibrate()

# Track by GLOBAL argmax (no local windowing) - raw field positions
STEP = 10
steps_total = T * 2  # 2 full orbits
rex = 5  # exclude Sun

angles = []
radii = []
for s in range(steps_total // STEP):
    sim.run(STEP, record_metrics=False)
    psi2 = np.array(sim.psi_real, dtype=np.float32) ** 2
    # Mask Sun
    psi2[cx-rex:cx+rex+1, cy-rex:cy+rex+1, cz-rex:cz+rex+1] = 0.0
    # Global argmax
    idx = np.unravel_index(np.argmax(psi2), psi2.shape)
    dist = math.sqrt((idx[0]-cx)**2+(idx[1]-cy)**2+(idx[2]-cz)**2)
    ang = math.degrees(math.atan2(idx[1]-cy, idx[0]-cx))
    radii.append(dist)
    angles.append(ang)
    if (s * STEP) % (T // 4) == 0:
        print(f"  step {s*STEP:5d}: pos=({idx[0]-cx:+3d},{idx[1]-cy:+3d},{idx[2]-cz:+3d})"
              f"  r={dist:.1f}  ang={ang:+.0f}°  raw_max={psi2[idx]:.6f}")

# How much did angle change?
d_angles = np.diff(angles)
# Handle wrap
d_angles = (d_angles + 180) % 360 - 180
total_rotation = np.sum(d_angles)
print(f"\nTotal rotation: {total_rotation:.1f}°  (expected ~{360*2:.0f}° for 2 orbits)")
print(f"r range: [{min(radii):.1f}, {max(radii):.1f}]")
print(f"ORBITING: {'YES' if abs(total_rotation) > 300 else 'NO - barely moving'}")

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(np.arange(len(angles)) * STEP, angles)
ax1.set_title('Angle vs time'); ax1.set_xlabel('step'); ax1.set_ylabel('angle (deg)')
ax2.plot(np.arange(len(radii)) * STEP, radii)
ax2.axhline(r_test, color='r', linestyle='--', label=f'r₀={r_test}')
ax2.set_title('Radius vs time'); ax2.set_xlabel('step'); ax2.set_ylabel('r (cells)')
ax2.legend()
plt.tight_layout()
plt.savefig('_orbit_diagnostic.png', dpi=100)
print(f"\nSaved _orbit_diagnostic.png")
