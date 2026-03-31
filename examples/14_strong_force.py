"""14 - Strong Force (Color Confinement Proxy)

Demonstrate a color-field confinement observable without injecting
external potentials.

We run a Level-2 (color) simulation and measure the chi-depression
line integral between color sources:

    I = integral (max(chi) - chi) ds

If color flux forms tube-like structures, I grows approximately
with source separation.
"""

from __future__ import annotations

import numpy as np

import lfm
from _common import make_out_dir, parse_no_anim, run_and_save_3d_movie


def run_pair(separation: int) -> tuple[dict[str, float], lfm.Simulation]:
    n = 56
    cx = n // 2
    p0 = (cx - separation // 2, n // 2, n // 2)
    p1 = (cx + separation // 2, n // 2, n // 2)

    cfg = lfm.SimulationConfig(
        grid_size=n,
        field_level=lfm.FieldLevel.COLOR,
    )
    sim = lfm.Simulation(cfg)

    # Two color excitations in different components.
    sim.place_solitons(
        positions=[p0, p1],
        amplitude=5.5,
        sigma=3.0,
        phases=[0.0, np.pi],
        colors=[0, 1],
    )
    sim.equilibrate()
    sim.run(steps=2500)

    conf = lfm.confinement_proxy(sim.chi, p0, p1, samples=96)
    return conf, sim


def main() -> None:
    _args = parse_no_anim()
    _OUT  = make_out_dir("14_strong_force")

    print("14 - Strong Force (Color Confinement Proxy)")
    print("=" * 64)
    print()
    print(f"{'sep':>5}  {'distance':>9}  {'line_integral':>14}  {'mean_depression':>16}")
    print(f"{'-----':>5}  {'---------':>9}  {'--------------':>14}  {'----------------':>16}")

    rows: list[tuple[int, dict[str, float]]] = []
    last_sim: lfm.Simulation | None = None
    for sep in (10, 14, 18):
        conf, sim = run_pair(sep)
        rows.append((sep, conf))
        last_sim = sim
        print(
            f"{sep:>5d}  "
            f"{conf['distance']:>9.3f}  "
            f"{conf['line_integral']:>14.5f}  "
            f"{conf['mean_depression']:>16.5f}"
        )

    distances = np.array([r[1]["distance"] for r in rows], dtype=np.float64)
    integrals = np.array([r[1]["line_integral"] for r in rows], dtype=np.float64)

    A = np.column_stack([distances, np.ones_like(distances)])
    coeffs, _, _, _ = np.linalg.lstsq(A, integrals, rcond=None)
    pred = A @ coeffs
    ss_res = np.sum((integrals - pred) ** 2)
    ss_tot = np.sum((integrals - np.mean(integrals)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    print()
    print(f"Linear fit: I ~ a*r + b,  a={coeffs[0]:.5f}, b={coeffs[1]:.5f}, R^2={r2:.4f}")
    if r2 > 0.8:
        print("Confinement proxy is strongly distance-dependent (approximately linear).")
    else:
        print("Weak linear trend at this resolution. Try larger grid/longer integration.")

    if last_sim is not None:
        run_and_save_3d_movie(last_sim, steps=500, out_dir=_OUT, stem="strong_force",
            field="chi_deficit", snapshot_every=20, no_anim=_args.no_anim)


if __name__ == "__main__":
    main()
