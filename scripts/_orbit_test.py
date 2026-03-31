import numpy as np
import lfm
from lfm import solar_system, place_bodies, BoundaryType, FieldLevel

bodies = solar_system()
cfg = lfm.SimulationConfig(grid_size=128, field_level=FieldLevel.REAL, boundary_type=BoundaryType.FROZEN, dt=0.02)
sim = lfm.Simulation(cfg)
omegas = place_bodies(sim, bodies, verbose=False)

# Check velocity IS encoded
psi_curr = np.array(sim._evolver.get_psi_real())
psi_prev = np.array(sim._evolver.get_psi_real_prev())
diff = psi_curr - psi_prev
print(f"diff nonzero: {np.count_nonzero(np.abs(diff) > 1e-6):,}  max: {np.abs(diff).max():.6f}")
print(f"Omegas: { {k: f'{v:.6f}' for k,v in omegas.items()} }")

# Run 2000 steps and compute COM of each planet
N = 128
cx = cy = cz = N // 2

# Get initial chi for Sun masking
sun = next(b for b in bodies if b.orbital_radius == 0)
# mask out Sun (sphere around center)
def planet_com(psi2, cx, cy, cz, r_planet, sigma_sun):
    """COM of planet region, masking out Sun."""
    psi2_m = psi2.copy()
    rex = int(sigma_sun * 4) + 1
    xi = np.arange(max(0,cx-rex), min(N,cx+rex+1)) - cx
    yi = np.arange(max(0,cy-rex), min(N,cy+rex+1)) - cy
    zi = np.arange(max(0,cz-rex), min(N,cz+rex+1)) - cz
    xx,yy,zz = np.meshgrid(xi,yi,zi,indexing='ij')
    xs0,xs1 = max(0,cx-rex), min(N,cx+rex+1)
    ys0,ys1 = max(0,cy-rex), min(N,cy+rex+1)
    zs0,zs1 = max(0,cz-rex), min(N,cz+rex+1)
    psi2_m[xs0:xs1,ys0:ys1,zs0:zs1][(xx**2+yy**2+zz**2)<=rex**2] = 0.0
    # Focus on a shell around r_planet
    xs = np.arange(N)[:,None,None]
    ys = np.arange(N)[None,:,None]
    zs = np.arange(N)[None,None,:]
    r2 = (xs-cx)**2 + (ys-cy)**2 + (zs-cz)**2
    shell = (r2 >= (r_planet-5)**2) & (r2 <= (r_planet+5)**2)
    psi2_m *= shell
    total = psi2_m.sum()
    if total < 1e-12:
        return None
    cx_p = (xs * psi2_m).sum() / total
    cy_p = (ys * psi2_m).sum() / total
    return cx_p - cx, cy_p - cy

merc = next(b for b in bodies if b.name == 'Mercury')
psi2 = np.array(sim._evolver.get_psi_real())**2
com0 = planet_com(psi2, cx, cy, cz, merc.orbital_radius, sun.sigma)
print(f"\nMercury COM at t=0: {com0}")
theta0 = np.arctan2(com0[1], com0[0]) if com0 else None

# Run 3000 steps
sim.run(3000, record_metrics=False)

psi2 = np.array(sim._evolver.get_psi_real())**2
com3k = planet_com(psi2, cx, cy, cz, merc.orbital_radius, sun.sigma)
print(f"Mercury COM at t=3000 steps: {com3k}")
if com0 and com3k:
    theta3k = np.arctan2(com3k[1], com3k[0])
    dtheta = np.degrees(theta3k - theta0)
    if dtheta < -180: dtheta += 360
    expected = 3000 * omegas['Mercury'] * 0.02 * 180 / np.pi
    print(f"Angle change: {dtheta:.2f} deg  (expected: {expected:.2f} deg)")
