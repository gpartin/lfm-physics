"""Clean orbit test: Sun + Mercury only (no other planet psi2 contamination)."""
import numpy as np
import lfm
from lfm import CelestialBody, BodyType, place_bodies, BoundaryType, FieldLevel
from scipy.ndimage import gaussian_filter

# Sun + Mercury only
bodies = [
    CelestialBody("Sun",     BodyType.STAR,         0.15,   0),
    CelestialBody("Mercury", BodyType.ROCKY_PLANET, 1.6e-7, 12, 0.0),
]

cfg = lfm.SimulationConfig(
    grid_size=128, field_level=FieldLevel.REAL,
    boundary_type=lfm.BoundaryType.FROZEN, dt=0.02
)
sim = lfm.Simulation(cfg)
omegas = place_bodies(sim, bodies, verbose=True)

N = 128
cx = cy = cz = N // 2
sun = next(b for b in bodies if b.orbital_radius == 0)
merc = next(b for b in bodies if b.name == 'Mercury')

rex = int(sun.sigma * 4) + 1  # = 11 cells
xs0,xs1 = max(0,cx-rex), min(N,cx+rex+1)
ys0,ys1 = max(0,cy-rex), min(N,cy+rex+1)
zs0,zs1 = max(0,cz-rex), min(N,cz+rex+1)

xi = np.arange(xs0,xs1) - cx
yi = np.arange(ys0,ys1) - cy
zi = np.arange(zs0,zs1) - cz
xx_m,yy_m,zz_m = np.meshgrid(xi,yi,zi,indexing='ij')
sun_sphere = (xx_m**2+yy_m**2+zz_m**2) <= rex**2

xs_arr = np.arange(N,dtype=np.float32)[:,None,None]
ys_arr = np.arange(N,dtype=np.float32)[None,:,None]
r_grid = np.sqrt((xs_arr-cx)**2 + (ys_arr-cy)**2)  # 2D projection

r_m = merc.orbital_radius

def get_merc_angle():
    psi2 = np.array(sim._evolver.get_psi_real(), dtype=np.float32) ** 2
    psi2[xs0:xs1, ys0:ys1, zs0:zs1][sun_sphere] = 0.0
    # No shell filter - just blur and find peak in planet region
    psi2_bl = gaussian_filter(psi2, sigma=1.5)
    # Restrict to shell r in [r_m-8, r_m+8]
    r_3d = np.sqrt((xs_arr - cx)**2 + (ys_arr - cy)**2 +
                   (np.arange(N,dtype=np.float32)[None,None,:] - cz)**2)
    shell = (r_3d >= r_m - 8) & (r_3d <= r_m + 8)
    psi2_m = psi2_bl * shell
    # Use weighted COM in z=cz plane
    psi2_slice = psi2_m[:, :, cz]  # middle z-slice
    total = psi2_slice.sum()
    if total < 1e-12:
        return None, None, None, None
    cx_p = float((np.arange(N,dtype=np.float32)[:,None] * psi2_slice).sum() / total)
    cy_p = float((np.arange(N,dtype=np.float32)[None,:] * psi2_slice).sum() / total)
    dx, dy = cx_p - cx, cy_p - cy
    r_p = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(dy, dx))
    return dx, dy, r_p, theta

print()
print("Mercury orbit tracking (Sun + Mercury only):")
print("Step     dx      dy      r      theta   expected_theta")
print("="*65)

step_count = 0
# T_mercury = 67197 steps (from place_bodies output)
# Run 2 full orbits: 135,000 steps. Check every 10,000 steps
checks = list(range(0, 135001, 5000))
for i, target in enumerate(checks):
    if i > 0:
        sim.run(target - checks[i-1], record_metrics=False)
        step_count = target
    dx, dy, r_p, theta = get_merc_angle()
    expected_theta = merc.orbital_phase * 180/np.pi + omegas['Mercury'] * step_count * 0.02 * 180/np.pi
    if dx is not None:
        print(f"{step_count:6d}  {dx:+7.3f}  {dy:+7.3f}  {r_p:6.2f}  {theta:+8.2f}  {expected_theta:+8.2f}")
    else:
        print(f"{step_count:6d}  COM not found")
