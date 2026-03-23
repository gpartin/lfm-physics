"""Debug the two failing user tests."""
import lfm
import numpy as np

# Dave 4a: try amplitude=8.0
print("Dave 4a with amplitude=8.0:")


def scatter_chi_min(amp, b_offset):
    s = lfm.Simulation(
        lfm.SimulationConfig(grid_size=32, field_level=lfm.FieldLevel.REAL)
    )
    s.place_soliton((8, 16, 16), amplitude=amp, velocity=(0.04, 0.0, 0.0))
    s.place_soliton((24, 16 + b_offset, 16), amplitude=amp, velocity=(-0.04, 0.0, 0.0))
    s.equilibrate()
    s.run(steps=400)
    stats = lfm.chi_statistics(s.chi)
    print(f"  amp={amp}, b={b_offset}: chi_min={stats['chi_min']:.3f}")
    return stats["chi_min"]


scatter_chi_min(8.0, 0)
scatter_chi_min(8.0, 4)

# Eve 5b: single central soliton to test radial profile
print("\nEve 5b: single central soliton amp=8, sigma=3")
sim8 = lfm.Simulation(lfm.SimulationConfig(grid_size=48, field_level=lfm.FieldLevel.REAL))
N8 = 48
sim8.place_soliton((N8 // 2, N8 // 2, N8 // 2), amplitude=8.0, sigma=3.0)
sim8.equilibrate()
sim8.run(steps=400)
prof = lfm.radial_profile(sim8.chi, center=(N8 // 2, N8 // 2, N8 // 2))
chi_arr = np.asarray(prof["mean"])
print(f"  chi_mean[:8] = {np.round(chi_arr[:8], 3)}")
print(f"  chi_min (r=0) = {chi_arr[0]:.3f}")
print(f"  global chi_min = {np.min(chi_arr):.3f}")
