"""Diagnose: is the velocity actually applied to the soliton?
Check the momentum of Venus soliton immediately after placement."""
import numpy as np, math, sys
sys.path.insert(0, '.')
import lfm
from lfm import solar_system, place_bodies, SimulationConfig, FieldLevel, BoundaryType, Simulation
from scipy.ndimage import gaussian_filter

N = 64
DT = 0.02

# Sun + Venus only
bodies = solar_system()
bodies = [b for b in bodies if b.name in ('Sun', 'Venus')]

cfg = SimulationConfig(grid_size=N, field_level=FieldLevel.REAL,
                       boundary_type=BoundaryType.FROZEN, dt=DT)
sim = Simulation(cfg)
body_omegas = place_bodies(sim, bodies, verbose=True)
cx = cy = cz = N // 2

venus = next(b for b in bodies if b.name == 'Venus')
print(f"\nVenus omega from place_bodies: {body_omegas[venus.name]:.6f} rad/step")
print(f"Expected v_circ: {body_omegas[venus.name] * venus.orbital_radius:.5f}")
print(f"Expected period: {2*math.pi / abs(body_omegas[venus.name]):.0f} steps")

# Look at psi_real immediately after placement
psi_now = np.array(sim.psi_real, dtype=np.float32)
psi_prev = np.array(sim.psi_real_prev, dtype=np.float32)
dpsi_dt = (psi_now - psi_prev) / DT

# Velocity of a point mass is encoded as dx/dt of the field center-of-mass
# The momentum density is: Re(psi_prev * grad_psi_now) ≈ psi * dpsi/dt
# For a carrier-wave soliton, the field envelope moves at v_group
# Let's find the center of the Venus soliton

# Isolate Venus region (mask Sun)
rex = 11
psi2 = psi_now ** 2
psi2_p = psi2.copy()
psi2_p[cx-rex:cx+rex+1, cy-rex:cy+rex+1, cz-rex:cz+rex+1] = 0.0
xi = np.arange(cx-rex, cx+rex+1) - cx
yi = np.arange(cy-rex, cy+rex+1) - cy
zi = np.arange(cz-rex, cz+rex+1) - cz
xx, yy, zz = np.meshgrid(xi, yi, zi, indexing='ij')
sphere = (xx**2 + yy**2 + zz**2) <= rex**2
psi2_p[cx-rex:cx+rex+1, cy-rex:cy+rex+1, cz-rex:cz+rex+1][sphere] = 0.0

# Venus center via argmax of blurred psi2
psi2_ps = gaussian_filter(psi2_p, sigma=1.5)
idx = np.unravel_index(np.argmax(psi2_ps), psi2_ps.shape)
print(f"\nVenus initial position (argmax): ({idx[0]-cx:+d},{idx[1]-cy:+d},{idx[2]-cz:+d})")
print(f"  distance from center: {math.sqrt((idx[0]-cx)**2+(idx[1]-cy)**2+(idx[2]-cz)**2):.1f}")

# Check the velocity field: compute center-of-mass shift between psi_now and psi_prev
# If venus has velocity, the field moved between the two time steps
mass_now  = np.sum(psi_now[cx+14:cx+22, cy-4:cy+4, cz-1:cz+2] ** 2)
mass_prev = np.sum(psi_prev[cx+14:cx+22, cy-4:cy+4, cz-1:cz+2] ** 2)

def com(arr):
    total = np.sum(arr)
    if total < 1e-10:
        return 0.0, 0.0
    ix = np.arange(arr.shape[0])
    iy = np.arange(arr.shape[1])
    iz = np.arange(arr.shape[2])
    xx_, yy_, zz_ = np.meshgrid(ix, iy, iz, indexing='ij')
    cx_ = np.sum(xx_ * arr) / total
    cy_ = np.sum(yy_ * arr) / total
    cz_ = np.sum(zz_ * arr) / total
    return cx_, cy_

# Extract region around Venus
rx0,rx1 = cx+10, cx+26
ry0,ry1 = cy-8, cy+8
r2 = psi_now[rx0:rx1,ry0:ry1,cz-1:cz+2]**2
r1 = psi_prev[rx0:rx1,ry0:ry1,cz-1:cz+2]**2
com2 = com(r2)
com1 = com(r1)
print(f"\nVenus CoM in psi_now region:  x={com2[0]+rx0:.2f}, y={com2[1]+ry0:.2f}")
print(f"Venus CoM in psi_prev region: x={com1[0]+rx0:.2f}, y={com1[1]+ry0:.2f}")
print(f"CoM shift: dx={com2[0]-com1[0]:.4f}, dy={com2[1]-com1[1]:.4f}  (per dt={DT})")
dx = com2[0]-com1[0]
dy = com2[1]-com1[1]
print(f"Velocity: vx={dx/DT:.4f},  vy={dy/DT:.5f}  (should be ~0 and ~{body_omegas[venus.name]*venus.orbital_radius:.5f})")

# Now run 10 steps and track again
print("\n--- Running 10 steps ---")
for step in range(1, 6):
    sim.run(2, record_metrics=False)
    psi2 = np.array(sim.psi_real, dtype=np.float32) ** 2
    psi2_p = psi2.copy()
    psi2_p[cx-rex:cx+rex+1, cy-rex:cy+rex+1, cz-rex:cz+rex+1][sphere] = 0.0
    psi2_ps = gaussian_filter(psi2_p, sigma=1.5)
    idx = np.unravel_index(np.argmax(psi2_ps), psi2_ps.shape)
    r = math.sqrt((idx[0]-cx)**2+(idx[1]-cy)**2+(idx[2]-cz)**2)
    ang = math.degrees(math.atan2(idx[1]-cy, idx[0]-cx))
    print(f"  step {step*2:3d}: pos=({idx[0]-cx:+d},{idx[1]-cy:+d})  r={r:.1f}  ang={ang:+.1f}°  max={psi2_ps[idx]:.5f}")
