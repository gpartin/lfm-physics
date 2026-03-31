"""Test: what amplitude keeps a planet soliton stable while orbiting?"""
import numpy as np, math, sys
sys.path.insert(0, '.')
import lfm
from lfm.analysis.observables import rotation_curve
from scipy.ndimage import gaussian_filter

N = 128
DT = 0.02
SAMPLE = 200

results = []
for amp_planet in [0.3, 1.0, 2.0, 4.0]:
    cfg = lfm.SimulationConfig(grid_size=N, field_level=lfm.FieldLevel.REAL,
                               boundary_type=lfm.BoundaryType.FROZEN, dt=DT)
    sim = lfm.Simulation(cfg)
    cx = cy = cz = N // 2

    sim.place_soliton((cx, cy, cz), amplitude=7.0, sigma=2.5)
    sim.equilibrate()

    rc = rotation_curve(sim.chi, sim.energy_density, center=(cx, cy, cz), plane_axis=2)
    r_arr = np.asarray(rc['r']); v_arr = np.asarray(rc['v_chi'])
    v_nyq = 0.8 * math.pi / 19 * 0.92
    r_test = 25
    v_raw = float(np.interp(r_test, r_arr, v_arr))
    v_circ = min(v_raw * 0.85, v_nyq)
    omega = v_circ / r_test
    T = int(2 * math.pi / omega)
    print(f"  v_raw={v_raw:.5f}  v_circ={v_circ:.5f}  T={T}  (~2 orbits={2*T} steps)", flush=True)

    vx = 0.0
    vy = v_circ
    sim.place_soliton((cx + r_test, cy, cz), amplitude=amp_planet, sigma=1.5,
                      velocity=(vx, vy, 0.0))
    sim.equilibrate()

    ecenter = (cx + r_test, cy, cz)
    rex = 9
    blur = 1.5
    distances = [float(r_test)]
    peaks = []
    n_samples = 2 * T // SAMPLE

    for i in range(n_samples):
        sim.run(SAMPLE, record_metrics=False)
        psi2 = np.array(sim.psi_real, dtype=np.float32) ** 2
        psi2_p = psi2.copy()
        psi2_p[cx-rex:cx+rex+1, cy-rex:cy+rex+1, cz-rex:cz+rex+1] = 0.0
        psi2_ps = gaussian_filter(psi2_p, sigma=blur)
        ex, ey, ez = ecenter
        move = omega * r_test * SAMPLE * DT
        sr = max(int(move * 1.5 + blur * 3), 5)
        x0, x1 = max(0, ex-sr), min(N, ex+sr+1)
        y0, y1 = max(0, ey-sr), min(N, ey+sr+1)
        z0, z1 = max(0, ez-sr), min(N, ez+sr+1)
        reg = psi2_ps[x0:x1, y0:y1, z0:z1]
        if reg.size > 0 and reg.max() > 0:
            idx = np.unravel_index(np.argmax(reg), reg.shape)
            fp = (x0+idx[0], y0+idx[1], z0+idx[2])
            peaks.append(float(reg.max()))
        else:
            fp = ecenter
            peaks.append(0.0)
        dist = math.sqrt((fp[0]-cx)**2 + (fp[1]-cy)**2 + (fp[2]-cz)**2)
        distances.append(dist)
        ecenter = fp
        if (i+1) % 5 == 0:
            ang = math.degrees(math.atan2(fp[1]-cy, fp[0]-cx))
            print(f"    sample {i+1}/{n_samples}: r={dist:.1f} ang={ang:.0f}  peak={peaks[-1]:.6f}", flush=True)

    r_min = min(distances); r_max = max(distances)
    peak_ratio = (peaks[-1] / peaks[0] * 100) if peaks else 0
    orbiting = r_min > 12
    print(f"amp={amp_planet:.1f}: r=[{r_min:.1f},{r_max:.1f}]  peak_0={peaks[0]:.5f}  "
          f"peak_f={peaks[-1]:.5f} ({peak_ratio:.0f}%)  "
          f"ORBITING={'YES' if orbiting else 'NO (collapsed)'}")
    results.append((amp_planet, r_min, r_max, peak_ratio, orbiting))

print("\n=== SUMMARY ===")
for amp, r_min, r_max, pr, orb in results:
    print(f"  amp={amp:.1f}: r=[{r_min:.1f},{r_max:.1f}]  peak_retained={pr:.0f}%  {'STABLE' if orb else 'COLLAPSED'}")
