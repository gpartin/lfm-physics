"""Quick verify: does the spherical-mask tracking now find Venus correctly?"""
import numpy as np, math, sys
sys.path.insert(0, '.')
import lfm
from lfm import solar_system, place_bodies, SimulationConfig, FieldLevel, BoundaryType, Simulation
from scipy.ndimage import gaussian_filter

N = 128
DT = 0.02

bodies = solar_system()
# Keep only Sun + Venus for isolation
bodies = [b for b in bodies if b.name in ('Sun', 'Venus')]
print(f"Bodies: {[b.name for b in bodies]}")

cfg = SimulationConfig(grid_size=N, field_level=FieldLevel.REAL,
                       boundary_type=BoundaryType.FROZEN, dt=DT)
sim = Simulation(cfg)
body_omegas = place_bodies(sim, bodies, verbose=True)
cx = cy = cz = N // 2

venus = next(b for b in bodies if b.name == 'Venus')
sun   = next(b for b in bodies if b.orbital_radius <= 0)
print(f"Venus r={venus.orbital_radius}, sigma={venus.sigma}, amp={venus.amplitude:.3f}")
print(f"Sun sigma={sun.sigma}, 4*sigma+1={int(sun.sigma*4)+1}")
print()

# Spherical mask radius
rex = int(sun.sigma * 4) + 1
print(f"Mask sphere radius: {rex} cells (Mercury at r=12, Venus at r=18)")

# Track Venus every 200 steps for 4000 steps (≈4 orbits at T≈1055)
SAMPLE = 200
search_center = (cx + venus.orbital_radius, cy, cz)
search_center = (int(search_center[0]), int(search_center[1]), int(search_center[2]))

blur = venus.sigma
omega = body_omegas[venus.name]

for i in range(20):
    sim.run(SAMPLE, record_metrics=False)
    psi2 = np.array(sim.psi_real, dtype=np.float32) ** 2
    
    # Spherical mask
    psi2_p = psi2.copy()
    xs0, xs1 = max(0, cx - rex), min(N, cx + rex + 1)
    ys0, ys1 = max(0, cy - rex), min(N, cy + rex + 1)
    zs0, zs1 = max(0, cz - rex), min(N, cz + rex + 1)
    xi = np.arange(xs0, xs1) - cx
    yi = np.arange(ys0, ys1) - cy
    zi = np.arange(zs0, zs1) - cz
    xx, yy, zz = np.meshgrid(xi, yi, zi, indexing='ij')
    sphere_mask = (xx**2 + yy**2 + zz**2) <= rex**2
    psi2_p[xs0:xs1, ys0:ys1, zs0:zs1][sphere_mask] = 0.0
    
    # Blur and find
    psi2_ps = gaussian_filter(psi2_p, sigma=blur)
    ex, ey, ez = search_center
    move = abs(omega) * venus.orbital_radius * SAMPLE * DT
    sr = max(int(move * 2 + blur * 4), 8)
    x0,x1 = max(0,ex-sr),min(N,ex+sr+1)
    y0,y1 = max(0,ey-sr),min(N,ey+sr+1)
    z0,z1 = max(0,ez-sr),min(N,ez+sr+1)
    reg = psi2_ps[x0:x1,y0:y1,z0:z1]
    if reg.size>0 and reg.max()>0:
        idx = np.unravel_index(np.argmax(reg), reg.shape)
        fp = (x0+idx[0], y0+idx[1], z0+idx[2])
    else:
        fp = search_center
    r = math.sqrt((fp[0]-cx)**2+(fp[1]-cy)**2+(fp[2]-cz)**2)
    ang = math.degrees(math.atan2(fp[1]-cy, fp[0]-cx))
    print(f"  step {(i+1)*SAMPLE:5d}: r={r:.1f}  ang={ang:+.0f}°  peak={reg.max():.5f}"
          f"  pos=({fp[0]-cx:+d},{fp[1]-cy:+d},{fp[2]-cz:+d})")
    search_center = fp
