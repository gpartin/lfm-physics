"""Track Mercury COmeter-of-mass over many timesteps to see if it orbits."""
import numpy as np
import lfm
from lfm import solar_system, place_bodies, BoundaryType, FieldLevel
from scipy.ndimage import gaussian_filter

bodies = solar_system()
cfg = lfm.SimulationConfig(
    grid_size=128, field_level=FieldLevel.REAL,
    boundary_type=BoundaryType.FROZEN, dt=0.02
)
sim = lfm.Simulation(cfg)
omegas = place_bodies(sim, bodies, verbose=True)

N = 128
cx = cy = cz = N // 2
sun = next(b for b in bodies if b.orbital_radius == 0)
merc = next(b for b in bodies if b.name == 'Mercury')

rex = int(sun.sigma * 4) + 1
xi = np.arange(max(0,cx-rex), min(N,cx+rex+1)) - cx
yi = np.arange(max(0,cy-rex), min(N,cy+rex+1)) - cy
zi = np.arange(max(0,cz-rex), min(N,cz+rex+1)) - cz
xs0,xs1 = max(0,cx-rex), min(N,cx+rex+1)
ys0,ys1 = max(0,cy-rex), min(N,cy+rex+1)
zs0,zs1 = max(0,cz-rex), min(N,cz+rex+1)
xx_m,yy_m,zz_m = np.meshgrid(xi,yi,zi,indexing='ij')
sun_sphere = (xx_m**2+yy_m**2+zz_m**2)<=rex**2

xs_arr = np.arange(N,dtype=np.float32)[:,None,None]
ys_arr = np.arange(N,dtype=np.float32)[None,:,None]
zs_arr = np.arange(N,dtype=np.float32)[None,None,:]
r_grid = np.sqrt((xs_arr-cx)**2 + (ys_arr-cy)**2 + (zs_arr-cz)**2)

# Expected shell for Mercury tracking
r_m = merc.orbital_radius
shell = (r_grid >= r_m - 6) & (r_grid <= r_m + 6)

def get_merc_angle(step):
    psi2 = np.array(sim._evolver.get_psi_real(), dtype=np.float32) ** 2
    # Mask Sun
    psi2[xs0:xs1, ys0:ys1, zs0:zs1][sun_sphere] = 0.0
    # Blur to remove carrier oscillation
    psi2_bl = gaussian_filter(psi2 * shell, sigma=1.5)
    total = psi2_bl.sum()
    if total < 1e-10:
        return None, None, None
    cx_p = float((xs_arr * psi2_bl).sum() / total)
    cy_p = float((ys_arr * psi2_bl).sum() / total)
    r_p = np.sqrt((cx_p-cx)**2 + (cy_p-cy)**2)
    theta = np.degrees(np.arctan2(cy_p-cy, cx_p-cx))
    return cx_p-cx, cy_p-cy, theta

print()
print("Step   dX      dY      theta    r")

prev_theta = None
for check_step in [0, 2000, 5000, 10000, 20000, 40000]:
    if check_step > 0:
        steps_to_run = check_step - (sim._step_count if hasattr(sim, '_step_count') else 0)
        # track steps manually
        pass

# Run incrementally
step_count = 0
checks = [0, 2000, 5000, 10000, 20000, 40000]
for i, target in enumerate(checks):
    if i > 0:
        sim.run(target - checks[i-1], record_metrics=False)
        step_count = target
    dx, dy, theta = get_merc_angle(step_count)
    expected_theta = merc.orbital_phase * 180/np.pi + omegas['Mercury'] * step_count * 0.02 * 180/np.pi
    if dx is not None:
        r = np.sqrt(dx**2 + dy**2)
        print(f"{step_count:6d}  {dx:+7.3f}  {dy:+7.3f}  {theta:+8.2f} deg  r={r:.2f}  expected={expected_theta:.1f} deg")
    else:
        print(f"{step_count:6d}  COM not found")

print()
print("DONE")
