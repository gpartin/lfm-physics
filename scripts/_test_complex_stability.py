"""Test complex soliton chi stability: does |Psi|^2 stay constant and chi stop oscillating?"""
import sys, numpy as np
sys.path.insert(0, r"c:\Papers\lfm-physics")
from lfm import Simulation, SimulationConfig, FieldLevel, BoundaryType
from lfm.scenarios.celestial import solar_system, place_bodies

N = 128
cfg = SimulationConfig(
    grid_size=N, field_level=FieldLevel.COMPLEX,
    boundary_type=BoundaryType.FROZEN, dt=0.02,
)
sim = Simulation(cfg)
cx = cy = cz = N // 2

bodies = solar_system()  # Sun + 4 planets
body_omegas = place_bodies(sim, bodies, verbose=True)

print("\nTracking chi stability at center and planet radii for 5000 steps:")
print(f"{'Step':>8}  {'chi_ctr':>10}  {'Dchi_ctr':>10}  {'chi_r12':>10}  {'|Psi|^2_r12':>12}  {'v_circ_r12':>12}")

def measure(step, sim):
    chi   = np.array(sim.chi, dtype=np.float64)
    pr    = np.array(sim.psi_real, dtype=np.float32)
    pi    = np.array(sim.psi_imag, dtype=np.float32)
    psi2  = pr**2 + pi**2  # energy density = |Psi|^2
    chi_c = float(chi[cx, cy, cz])
    chi_r = float(chi[cx+12, cy, cz])
    p2_r  = float(psi2[cx+12, cy, cz])
    p2_c  = float(psi2[cx, cy, cz])
    # direct gradient at r=12 along x
    dchi_dr = (float(chi[cx+13, cy, cz]) - float(chi[cx+11, cy, cz])) / 2.0
    v_circ  = float(np.sqrt(max(12.0 * (1.0/19.0) * abs(dchi_dr), 0.0)))
    print(f"{step:8d}  {chi_c:10.6f}  {19-chi_c:10.6f}  {chi_r:10.6f}  {p2_r:12.6f}  {v_circ:12.6f}")
    return chi_c, p2_c

# Initial snapshot
chi0_ctr, p2_0 = measure(0, sim)

for ckpt in [100, 500, 1000, 2000, 3000, 5000]:
    prev = int(sim._step_count) if hasattr(sim, '_step_count') else 0
    sim.run(ckpt - getattr(sim, '_last_ckpt', 0), record_metrics=False)
    sim._last_ckpt = ckpt
    measure(ckpt, sim)
