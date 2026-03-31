"""Trace when psi_real_prev becomes NaN in the place_bodies flow."""
import numpy as np, math, sys
sys.path.insert(0, '.')
import lfm
from lfm import solar_system, SimulationConfig, FieldLevel, BoundaryType, Simulation
from lfm.analysis.observables import rotation_curve
from lfm.scenarios.celestial import _VALID_V

N = 64
DT = 0.02
CHI0 = 19

bodies = solar_system()
bodies = [b for b in bodies if b.name in ('Sun', 'Venus')]
sun = next(b for b in bodies if b.orbital_radius <= 0)
venus = next(b for b in bodies if b.name == 'Venus')

cfg = SimulationConfig(grid_size=N, field_level=FieldLevel.REAL,
                       boundary_type=BoundaryType.FROZEN, dt=DT)
sim = Simulation(cfg)
cx = cy = cz = N // 2

def check_prev(label):
    try:
        p = np.array(sim.psi_real_prev)
        print(f"  [{label}] psi_real_prev shape={p.shape} nan={np.isnan(p).any()} range=[{p.min():.4f},{p.max():.4f}]")
    except Exception as e:
        print(f"  [{label}] psi_real_prev error: {e}")

check_prev("initial")

# Step 1: Place Sun
sim.place_soliton((cx, cy, cz), amplitude=sun.amplitude, sigma=sun.sigma)
check_prev("after place Sun (no velocity)")

# Step 2: First equilibrate
sim.equilibrate()
check_prev("after equilibrate Sun")

# Step 3: Place Venus with velocity
rc = rotation_curve(sim.chi, sim.energy_density, center=(cx, cy, cz), plane_axis=2)
r_arr = np.asarray(rc["r"]); v_arr_raw = np.asarray(rc["v_chi"])
valid = v_arr_raw > _VALID_V
r_ref = float(r_arr[valid][-1])
v_ref = float(v_arr_raw[valid][-1])
v_arr = np.where(r_arr > r_ref, v_ref * np.sqrt(r_ref / np.maximum(r_arr, 0.1)), v_arr_raw)
v_nyq = 0.8 * np.pi / CHI0 * 0.92
r = venus.orbital_radius
theta = venus.orbital_phase
v_circ = min(float(np.interp(r, r_arr, v_arr)), v_nyq)
bx = cx + r * np.cos(theta)
by = cy + r * np.sin(theta)
vx_vel = -np.sin(theta) * v_circ
vy_vel = np.cos(theta) * v_circ
print(f"\nVenus: r={r}, theta={theta:.2f}, pos=({bx:.1f},{by:.1f},{cz}), vel=({vx_vel:.5f},{vy_vel:.5f})")

sim.place_soliton((bx, by, cz), amplitude=venus.amplitude, sigma=venus.sigma,
                  velocity=(float(vx_vel), float(vy_vel), 0.0))
check_prev("after place Venus (with velocity)")

# Step 4: Second equilibrate (this is what place_bodies does at the end)
print("\nAbout to call second equilibrate()...")
sim.equilibrate()
check_prev("after second equilibrate (re-equilibrate)")

# Now run 10 steps and check velocity
print("\nRunning 10 steps to check if Venus moves:")
for s in range(10):
    sim.run(1, record_metrics=False)
    p = np.array(sim.psi_real)
    vx_i = int(round(bx))
    vy_i = int(round(by))
    # Sample near Venus initial position
    print(f"  step {s+1}: max_psi near Venus={p[vx_i-2:vx_i+3,vy_i-2:vy_i+3,cz].max():.5f}")
