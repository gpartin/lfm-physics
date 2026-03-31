"""Deep diagnostic: check if psi_real != psi_real_prev after equilibration.
If these are equal, there is no velocity → soliton stays frozen."""
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
body_omegas = place_bodies(sim, bodies, verbose=True)
cx = cy = cz = N // 2

# Get psi_real and psi_real_prev
psi_now  = np.array(sim.psi_real,      dtype=np.float64)
psi_prev = np.array(sim.psi_real_prev, dtype=np.float64)

diff = psi_now - psi_prev
print(f"\npsi_real range: [{psi_now.min():.4f}, {psi_now.max():.4f}]")
print(f"psi_real_prev range: [{psi_prev.min():.4f}, {psi_prev.max():.4f}]")
print(f"diff range:          [{diff.min():.6f}, {diff.max():.6f}]")
print(f"diff max abs:        {np.abs(diff).max():.6f}")
print(f"diff nonzero frac:   {(np.abs(diff) > 1e-6).mean()*100:.1f}%")

# If diff ≈ 0 everywhere, the velocity encoding was lost
if np.abs(diff).max() < 1e-5:
    print("\n❌ VELOCITY ENCODING LOST: psi_real_prev == psi_real => soliton has no velocity")
else:
    print("\n✅ VELOCITY ENCODED: psi_real != psi_real_prev")

# Now check the chi buffers
chi_now  = np.array(sim.chi, dtype=np.float64)
try:
    chi_prev = np.array(sim.chi_prev, dtype=np.float64)
    chi_diff = chi_now - chi_prev
    print(f"\nchi range: [{chi_now.min():.4f}, {chi_now.max():.4f}]")
    print(f"chi_prev range: [{chi_prev.min():.4f}, {chi_prev.max():.4f}]")
    print(f"chi diff max abs: {np.abs(chi_diff).max():.6f}")
    if np.abs(chi_diff).max() > 1e-5:
        print("✅ chi has time derivative (chi_prev != chi now)")
    else:
        print("❌ chi has NO time derivative (chi_prev == chi)")
except Exception as e:
    print(f"chi_prev not accessible: {e}")

# Looking at specific Venus region
vr = 18
px_v = cx + vr  # initial x of Venus
print(f"\nVenus region (r≈18, near ({px_v},{cy},{cz})):")
for dr in range(-3, 4):
    x = px_v + dr
    if 0 <= x < N:
        print(f"  x={x}: psi_now={psi_now[x,cy,cz]:.5f}  psi_prev={psi_prev[x,cy,cz]:.5f}  "
              f"diff={psi_now[x,cy,cz]-psi_prev[x,cy,cz]:.6f}")

# Also look near the start of Venus (initial position in place_bodies)
venus = next(b for b in bodies if b.name == 'Venus')
print(f"\nVenus orbital phase: {venus.orbital_phase:.3f} rad")
vx_pos = cx + venus.orbital_radius * math.cos(venus.orbital_phase)
vy_pos = cy + venus.orbital_radius * math.sin(venus.orbital_phase)
print(f"Venus center: ({vx_pos:.1f}, {vy_pos:.1f}, {cz})")
ivx = int(round(vx_pos))
ivy = int(round(vy_pos))
print(f"Nearest int: ({ivx},{ivy},{cz})")
for di in range(-2,3):
    for dj in range(-2,3):
        xi,yi = ivx+di, ivy+dj
        if 0<=xi<N and 0<=yi<N:
            d = psi_now[xi,yi,cz] - psi_prev[xi,yi,cz]
            if abs(psi_now[xi,yi,cz]) > 0.001:
                print(f"  ({xi-cx:+d},{yi-cy:+d}): now={psi_now[xi,yi,cz]:.5f}  prev={psi_prev[xi,yi,cz]:.5f}  diff={d:.6f}")
