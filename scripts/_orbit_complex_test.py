"""Clean orbit test with complex solitons: does Mercury orbit stably?
Track Mercury's COM position from psi2=|Psi|^2 maximum over 200k steps.
"""
import sys, numpy as np
sys.path.insert(0, r"c:\Papers\lfm-physics")
from lfm import Simulation, SimulationConfig, FieldLevel, BoundaryType
from lfm.scenarios.celestial import solar_system, place_bodies, CelestialBody, BodyType

N = 128
cfg = SimulationConfig(
    grid_size=N, field_level=FieldLevel.COMPLEX,
    boundary_type=BoundaryType.FROZEN, dt=0.02,
)
sim = Simulation(cfg)
cx = cy = cz = N // 2

sun     = CelestialBody("Sun",     BodyType.STAR,         0.15, 0)
mercury = CelestialBody("Mercury", BodyType.ROCKY_PLANET, 1.6e-7, 12, 0.0)
bodies  = [sun, mercury]
body_omegas = place_bodies(sim, bodies, verbose=True)

# Sun mask radius for planet tracking
rex = int(sun.sigma * 4) + 1  # = 11 cells

def find_mercury_com(sim, prev_guess=None):
    """Find Mercury's center via |Psi|^2 peak, masked around Sun."""
    from scipy.ndimage import gaussian_filter
    pr = np.array(sim.psi_real, dtype=np.float32)
    pi = np.array(sim.psi_imag, dtype=np.float32)
    psi2 = pr**2 + pi**2  # energy density = |Psi|^2

    # Mask out Sun region (sphere of radius rex)
    psi2_masked = psi2.copy()
    xs0, xs1 = max(0, cx-rex), min(N, cx+rex+1)
    ys0, ys1 = max(0, cy-rex), min(N, cy+rex+1)
    zs0, zs1 = max(0, cz-rex), min(N, cz+rex+1)
    xi = np.arange(xs0, xs1) - cx
    yi = np.arange(ys0, ys1) - cy
    zi = np.arange(zs0, zs1) - cz
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing='ij')
    sphere = (xx**2 + yy**2 + zz**2) <= rex**2
    psi2_masked[xs0:xs1, ys0:ys1, zs0:zs1][sphere] = 0.0

    # Blur to suppress carrier oscillations
    psi2_smooth = gaussian_filter(psi2_masked, sigma=1.5)

    # Search in window around previous position
    if prev_guess is None:
        search = psi2_smooth
    else:
        px, py, pz = prev_guess
        sr = 12  # generous search window
        x0, x1 = max(0,int(px)-sr), min(N,int(px)+sr+1)
        y0, y1 = max(0,int(py)-sr), min(N,int(py)+sr+1)
        z0, z1 = max(0,int(pz)-sr), min(N,int(pz)+sr+1)
        region = psi2_smooth[x0:x1, y0:y1, z0:z1]
        if region.max() > 0:
            idx = np.unravel_index(np.argmax(region), region.shape)
            return (float(x0+idx[0]), float(y0+idx[1]), float(z0+idx[2]))
        search = psi2_smooth

    idx = np.unravel_index(np.argmax(search), search.shape)
    return (float(idx[0]), float(idx[1]), float(idx[2]))

print(f"\n{'Step':>8}  {'r':>8}  {'theta':>8}  {'|Psi|^2_merc':>15}  {'chi_r':>10}")
prev_pos = (float(cx + 12), float(cy), float(cz))
CHECKPOINTS = list(range(0, 201000, 5000))

prev = 0
for ckpt in CHECKPOINTS:
    if ckpt > 0:
        sim.run(ckpt - prev, record_metrics=False)
        prev = ckpt

    mx, my, mz = find_mercury_com(sim, prev_pos)
    prev_pos = (mx, my, mz)
    dx, dy = mx - cx, my - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(dy, dx))

    # chi and psi2 at Mercury's position
    ix, iy, iz = int(round(mx)), int(round(my)), int(round(mz))
    pr = np.array(sim.psi_real, dtype=np.float32)
    pi = np.array(sim.psi_imag, dtype=np.float32)
    psi2_m = float(pr[ix, iy, iz]**2 + pi[ix, iy, iz]**2)
    chi_m  = float(np.array(sim.chi)[ix, iy, iz])

    print(f"{ckpt:8d}  {r:8.3f}  {theta:8.2f}°  {psi2_m:15.6f}  {chi_m:10.6f}")
