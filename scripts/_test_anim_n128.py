"""Quick N=128 animation test — 40 frames × 750 steps.

Covers ~0.45 Mercury orbits.  Focus: tracking quality and visual.
"""
from pathlib import Path
import lfm
from lfm import solar_system, place_bodies, animate_celestial_3d

N               = 128
FRAMES          = 40
STEPS_PER_FRAME = 750
DT              = 0.02

print(f"LFM Solar System Animation Test  N={N}³  {FRAMES}f × {STEPS_PER_FRAME} steps")
print("=" * 60)

bodies = solar_system()
print(f"\n  {'Name':12s} {'amp':>6s}  {'sigma':>5s}  {'r':>4s}")
print("  " + "─" * 35)
for b in bodies:
    print(f"  {b.name:12s} {b.amplitude:>6.3f}  {b.sigma:>5.1f}  {b.orbital_radius:>4.0f}")

cfg = lfm.SimulationConfig(
    grid_size=N,
    field_level=lfm.FieldLevel.COMPLEX,
    boundary_type=lfm.BoundaryType.FROZEN,
    dt=DT,
    report_interval=999_999,
)
sim = lfm.Simulation(cfg)

body_omegas = place_bodies(sim, bodies, verbose=True)

print("\nOmega (rad/time) for each body:")
for name, omega in body_omegas.items():
    T = (2 * 3.14159 / omega / DT) if omega > 0 else 0
    print(f"  {name:12s}: omega={omega:.5f}  T≈{T:.0f} steps")

out = str(Path(__file__).parent / "_test_anim_n128.mp4")
saved = animate_celestial_3d(
    sim, bodies, body_omegas,
    n_frames=FRAMES,
    steps_per_frame=STEPS_PER_FRAME,
    fps=15,
    title=f"LFM Solar System  N={N}³  (test)",
    save_path=out,
    verbose=True,
    camera_rotate_speed=1.5,   # 1.5°/frame → 60° rotation over 40 frames
)

print(f"\nDONE.  Saved: {saved or 'FAILED'}")
