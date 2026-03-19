"""13 - Weak Force (Parity Asymmetry)

This example isolates the epsilon_w * j term in GOV-02.

We run the same complex-field setup twice:
  1) with epsilon_w = 0.1 (default from chi0)
  2) with epsilon_w = 0.0 (control)

Then we compare the left/right chi-depression asymmetry using
lfm.weak_parity_asymmetry().

No external force law is injected. We measure only what emerges
from GOV-01 + GOV-02.
"""

from __future__ import annotations

import numpy as np

import lfm


def build_run(epsilon_w: float) -> dict[str, float]:
    cfg = lfm.SimulationConfig(
        grid_size=48,
        field_level=lfm.FieldLevel.COMPLEX,
        epsilon_w=epsilon_w,
    )
    sim = lfm.Simulation(cfg)

    # Two opposite-phase packets with an imposed phase gradient in +x
    # to generate nonzero j = Im(Psi*grad(Psi)).
    sim.place_soliton((18, 24, 24), amplitude=5.0, sigma=3.2, phase=0.0)
    sim.place_soliton((30, 24, 24), amplitude=5.0, sigma=3.2, phase=np.pi)
    sim.equilibrate()

    pr = sim.psi_real
    pi = sim.psi_imag
    n = pr.shape[0]

    # Impose a smooth helical phase twist along x (still purely lattice-defined).
    x = np.arange(n, dtype=np.float32)
    phase = 0.35 * np.sin(2.0 * np.pi * x / n).astype(np.float32)
    phase3 = phase[:, None, None]
    c = np.cos(phase3)
    s = np.sin(phase3)

    pr_new = pr * c - pi * s
    pi_new = pr * s + pi * c
    sim.psi_real = pr_new.astype(np.float32)
    sim.psi_imag = pi_new.astype(np.float32)

    sim.run(steps=2500)

    j = lfm.momentum_density(sim.psi_real, sim.psi_imag)
    asym = lfm.weak_parity_asymmetry(sim.chi, axis=0)

    return {
        "j_rms": float(np.sqrt(np.mean(j["j_total"] ** 2))),
        "asymmetry": float(asym["asymmetry"]),
        "plus_weight": float(asym["plus_weight"]),
        "minus_weight": float(asym["minus_weight"]),
    }


def main() -> None:
    print("13 - Weak Force (Parity Asymmetry)")
    print("=" * 60)
    print()

    w_on = build_run(epsilon_w=lfm.EPSILON_W)
    w_off = build_run(epsilon_w=0.0)

    print(f"epsilon_w ON  ({lfm.EPSILON_W:.3f})")
    print(f"  j_rms      = {w_on['j_rms']:.6f}")
    print(f"  asymmetry  = {w_on['asymmetry']:+.6f}")
    print(f"  plus/minus = {w_on['plus_weight']:.3f} / {w_on['minus_weight']:.3f}")
    print()

    print("epsilon_w OFF (0.000 control)")
    print(f"  j_rms      = {w_off['j_rms']:.6f}")
    print(f"  asymmetry  = {w_off['asymmetry']:+.6f}")
    print(f"  plus/minus = {w_off['plus_weight']:.3f} / {w_off['minus_weight']:.3f}")
    print()

    delta = abs(w_on["asymmetry"]) - abs(w_off["asymmetry"])
    print(f"delta |asym| (on - off) = {delta:+.6f}")
    if delta > 0:
        print("Parity asymmetry strengthens when epsilon_w is enabled.")
    else:
        print("No clear parity separation at this scale. Try larger grid or longer run.")


if __name__ == "__main__":
    main()
