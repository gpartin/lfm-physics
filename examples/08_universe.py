"""08 — Simulate a Universe

This is the capstone.  We start from a handful of Poisson-equilibrated
solitons on a 64³ grid and let GOV-01 + GOV-02 run.  Structure forms
by itself: χ-wells (galaxies) separated by χ-voids (dark energy).

We use CosmicScale to translate lattice ticks into billions of
years and cell counts into megaparsecs.

All you need is two equations and some initial energy.
"""

import numpy as np

import lfm
from lfm import CosmicScale
from _common import make_out_dir, parse_no_anim, run_and_save_3d_movie

_args = parse_no_anim()
_OUT  = make_out_dir("08_universe")

# --- Configuration ---
N = 64
config = lfm.SimulationConfig(grid_size=N)
sim = lfm.Simulation(config)

# Map lattice units to physical units
scale = CosmicScale(box_mpc=100.0, grid_size=N)

print("08 — Simulate a Universe")
print("=" * 55)
print()
print(f"Grid:   {N}³ = {N**3:,} cells")
print(f"Scale:  {scale.cell_to_mpc():.2f} Mpc/cell")
print("Box:    100 Mpc across")
print()

# --- Seed with Poisson-equilibrated solitons (like primordial matter) ---
rng = np.random.default_rng(2026)
amplitude = 6.0
sigma = 3.0
min_sep = 10
n_particles = 9

# Random non-overlapping positions
positions = []
for _ in range(200):  # try up to 200 placements
    pos = (
        int(rng.integers(sigma * 2, N - sigma * 2)),
        int(rng.integers(sigma * 2, N - sigma * 2)),
        int(rng.integers(sigma * 2, N - sigma * 2)),
    )
    if all(
        abs(pos[0] - p[0]) + abs(pos[1] - p[1]) + abs(pos[2] - p[2]) >= min_sep for p in positions
    ):
        positions.append(pos)
    if len(positions) == n_particles:
        break

print(f"Placing {len(positions)} solitons, amplitude={amplitude}, σ={sigma}")
for pos in positions:
    sim.place_soliton(pos, amplitude=amplitude, sigma=sigma)

# Poisson-equilibrate χ to match the initial energy distribution
psi_sq = sim.psi_real**2
kappa = lfm.KAPPA
chi0 = lfm.CHI0
rho_hat = np.fft.fftn(kappa * psi_sq)
N3 = N
kx = np.fft.fftfreq(N3) * 2 * np.pi
ky, kz = kx.copy(), kx.copy()
KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
K2 = KX**2 + KY**2 + KZ**2
K2[0, 0, 0] = 1.0  # avoid divide-by-zero at DC
chi_hat = -rho_hat / K2
chi_hat[0, 0, 0] = 0.0  # DC mode = chi0 background
chi_init = chi0 + np.real(np.fft.ifftn(chi_hat)).astype(np.float32)
sim.chi = chi_init.astype(np.float32)
print(f"Poisson equilibration: χ range {chi_init.min():.2f} → {chi_init.max():.2f}")
print()

# --- Milestones to track ---
total_steps = 50_000
snapshot_interval = 10_000
well_threshold = 18.0  # χ below this = gravitational well
void_threshold = 18.8  # χ above this = expanding void

print(f"Running {total_steps:,} steps...")
print()
header = (
    f"{'Time':>12s}  {'χ_min':>6s}  {'Wells%':>7s}"
    f"  {'Voids%':>7s}  {'Clusters':>8s}  {'Energy':>10s}"
)
print(header)
print("-" * len(header))


def report(label: str) -> None:
    chi = sim.chi
    N3 = chi.size
    chi_min = float(chi.min())
    wells_pct = 100.0 * np.sum(chi < well_threshold) / N3
    voids_pct = 100.0 * np.sum(chi > void_threshold) / N3

    # Simple cluster count: connected components of wells
    psi_sq = sim.energy_density
    peaks = lfm.find_peaks(psi_sq, n=20, min_separation=3)
    n_clusters = len(peaks)

    e = float(np.sum(psi_sq))

    print(
        f"{label:>12s}  {chi_min:6.2f}  {wells_pct:6.1f}%"
        f"  {voids_pct:6.1f}%  {n_clusters:8d}  {e:10.2f}"
    )


report(scale.format_cosmic_time(0))

for milestone in range(snapshot_interval, total_steps + 1, snapshot_interval):
    sim.run(steps=snapshot_interval)
    report(scale.format_cosmic_time(milestone))

print()

# --- Final summary ---
chi = sim.chi
wells_pct = 100.0 * np.sum(chi < well_threshold) / chi.size
voids_pct = 100.0 * np.sum(chi > void_threshold) / chi.size

print("Final state:")
print(f"  χ range: {chi.min():.2f} → {chi.max():.2f}")
print(f"  Wells (χ < {well_threshold}): {wells_pct:.1f}%")
print(f"  Voids (χ > {void_threshold}): {voids_pct:.1f}%")
print()

if wells_pct > 0.1 and voids_pct > 30.0:
    print("Structure formed!  Dense regions (galaxies) and voids (dark energy)")
    print("coexist — just like the real cosmos.")
elif wells_pct > 0.5:
    print("Some structure forming.  Run longer for more pronounced wells.")
else:
    print("Mostly uniform.  Try higher amplitude or more steps.")

print()
print("You just simulated a universe with two equations and some noise.")
print("Everything else — gravity, structure, expansion — emerged.")

# 3-D movie of the final cosmic structure
snaps, _movie = run_and_save_3d_movie(
    sim, steps=5000, out_dir=_OUT, stem="universe",
    field="chi_deficit", snapshot_every=100, no_anim=_args.no_anim,
)
