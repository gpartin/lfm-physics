"""Check sim attributes and how to access psi_prev."""
import numpy as np, math, sys
sys.path.insert(0, '.')
import lfm
from lfm import solar_system, place_bodies, SimulationConfig, FieldLevel, BoundaryType, Simulation

N = 64
DT = 0.02

bodies = solar_system()
bodies = [b for b in bodies if b.name in ('Sun', 'Venus')]

cfg = SimulationConfig(grid_size=N, field_level=FieldLevel.REAL,
                       boundary_type=BoundaryType.FROZEN, dt=DT)
sim = Simulation(cfg)
body_omegas = place_bodies(sim, bodies, verbose=False)

print("dir(sim) relevant:", [a for a in dir(sim) if 'psi' in a.lower() or 'prev' in a.lower()])
print()

# Check the psi_real field for velocity encoding
# In place_soliton with velocity, the previous time-step buffer is set to
# psi_real - dt*v*grad(psi_real) to encode the velocity

# Just run 1 step and track position change
psi_before = np.array(sim.psi_real, dtype=np.float32).copy()
sim.run(1, record_metrics=False)
psi_after = np.array(sim.psi_real, dtype=np.float32)

dpsi = psi_after - psi_before
cx = cy = cz = N // 2
# Show change near Venus position (r=18)
v = next(b for b in bodies if b.name == 'Venus')
vx_pos = cx + int(v.orbital_radius)
print(f"Venus expected at: ({vx_pos},{cy},{cz})")
print(f"psi change near Venus [18:22, cy-2:cy+2]:")
print(psi_after[vx_pos-2:vx_pos+3, cy-2:cy+3, cz].round(4))
print()
print("psi_real at Venus:")
print(psi_after[vx_pos-2:vx_pos+3, cy-2:cy+3, cz].round(4))
print()

# Run a few more steps and record center-of-mass
from scipy.ndimage import gaussian_filter

def get_venus_pos(psi_real, cx, cy, cz, N, blur=1.5):
    psi2 = psi_real.astype(np.float32) ** 2
    rex = 11
    psi2_p = psi2.copy()
    xs0,xs1 = max(0,cx-rex),min(N,cx+rex+1)
    ys0,ys1 = max(0,cy-rex),min(N,cy+rex+1)
    zs0,zs1 = max(0,cz-rex),min(N,cz+rex+1)
    xi=np.arange(xs0,xs1)-cx; yi=np.arange(ys0,ys1)-cy; zi=np.arange(zs0,zs1)-cz
    xx,yy,zz=np.meshgrid(xi,yi,zi,indexing='ij')
    sphere=(xx**2+yy**2+zz**2)<=rex**2
    psi2_p[xs0:xs1,ys0:ys1,zs0:zs1][sphere]=0.0
    psi2_ps=gaussian_filter(psi2_p,sigma=blur)
    idx=np.unravel_index(np.argmax(psi2_ps),psi2_ps.shape)
    r=math.sqrt((idx[0]-cx)**2+(idx[1]-cy)**2+(idx[2]-cz)**2)
    ang=math.degrees(math.atan2(idx[1]-cy,idx[0]-cx))
    return idx,r,ang,psi2_ps[idx]

print("Tracking Venus every 1 step for 20 steps:")
for s in range(20):
    sim.run(1, record_metrics=False)
    psi=np.array(sim.psi_real)
    idx,r,ang,pk=get_venus_pos(psi,cx,cy,cz,N)
    print(f"  step {s+2:3d}: ({idx[0]-cx:+d},{idx[1]-cy:+d},{idx[2]-cz:+d}) r={r:.1f} ang={ang:+.1f}° pk={pk:.5f}")
