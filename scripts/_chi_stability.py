"""Chi stability test: track chi well depth after equilibrate over 6000 steps.

Goal: find when chi settles, and measure rotation_curve v_circ at each planet radius.
"""
import sys, numpy as np
sys.path.insert(0, r"c:\Papers\lfm-physics")
from lfm import Simulation, SimulationConfig, FieldLevel, BoundaryType
from lfm.analysis.observables import rotation_curve

N = 128
cfg = SimulationConfig(
    grid_size=N,
    field_level=FieldLevel.REAL,
    boundary_type=BoundaryType.FROZEN,
    dt=0.02,
)
sim = Simulation(cfg)
cx = cy = cz = N // 2

# Place Sun (mass_solar=0.15 → amplitude≈3.0)
MASS_SOLAR = 0.15
SUN_AMPLITUDE = 7.0 * MASS_SOLAR  # rough: 1.05 → the library uses 1.85*Mass^0.33 ...
# Actually just use the library constant
from lfm.scenarios.celestial import solar_system, place_bodies, CelestialBody, BodyType
sun = CelestialBody("Sun", BodyType.STAR, MASS_SOLAR, 0)
sim.place_soliton((cx, cy, cz), amplitude=sun.amplitude, sigma=sun.sigma)
sim.equilibrate()

chi = np.array(sim.chi)
print(f"{'Step':>8} {'chi_ctr':>10} {'delta_chi':>12} {'chi_r12':>10} {'chi_r18':>10} {'v_chi_r12':>12} {'v_chi_r18':>12}")
PLANET_RADII = [12, 18, 25, 33]

def measure_direct_vcir(chi_arr, r):
    """Direct v_circ from chi gradient at integer r (along x-axis)."""
    cx2 = cy2 = cz2 = N // 2
    chi_before = chi_arr[cx2 + r - 1, cy2, cz2]
    chi_after  = chi_arr[cx2 + r + 1, cy2, cz2]
    dchi_dr = (chi_after - chi_before) / 2.0  # cells^-1
    v_sq = r * (1.0 / 19.0) * abs(dchi_dr)
    return float(np.sqrt(max(v_sq, 0.0))), float(dchi_dr)

def do_report(step, sim):
    chi_arr = np.array(sim.chi, dtype=np.float64)
    chi_ctr = chi_arr[cx, cy, cz]
    delta   = 19.0 - float(chi_ctr)
    chi_r12 = chi_arr[cx + 12, cy, cz]
    chi_r18 = chi_arr[cx + 18, cy, cz]
    v12, g12 = measure_direct_vcir(chi_arr, 12)
    v18, g18 = measure_direct_vcir(chi_arr, 18)

    # rotation_curve at all planet radii
    rc = rotation_curve(chi_arr, np.array(sim.energy_density, dtype=np.float64),
                        center=(cx, cy, cz))
    rc_interp = {r: float(np.interp(r, rc["r"], rc["v_chi"])) for r in PLANET_RADII}

    print(f"{step:8d} {chi_ctr:10.6f} {delta:12.6f} {chi_r12:10.6f} {chi_r18:10.6f}  v_chi={v12:.5f}  rc_v_chi={rc_interp[12]:.5f}")
    return {
        "step": step, "chi_ctr": chi_ctr, "delta": delta,
        "chi_r12": chi_r12, "chi_r18": chi_r18,
        "v12_direct": v12, "v18_direct": v18,
        "rc": rc_interp,
    }

records = []
records.append(do_report(0, sim))

CHECKPOINTS = [100, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000]
prev = 0
for ckpt in CHECKPOINTS:
    sim.run(ckpt - prev, record_metrics=False)
    prev = ckpt
    records.append(do_report(ckpt, sim))

print()
print("Summary: v_chi from rotation_curve at each planet radius:")
print(f"{'Step':>8}  {'r=12':>8}  {'r=18':>8}  {'r=25':>8}  {'r=33':>8}  {'chi_min':>10}")
for rec in records:
    delta = rec.get("delta", 0)
    print(f"{rec['step']:8d}  {rec['rc'][12]:.5f}  {rec['rc'][18]:.5f}  "
          f"{rec['rc'][25]:.5f}  {rec['rc'][33]:.5f}  Δχ={delta:.5f}")
