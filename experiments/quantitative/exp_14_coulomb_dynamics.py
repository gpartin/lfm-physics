#!/usr/bin/env python
"""Legacy acceleration probe, deliberately NOT a signed-charge certification.

Run: python -m experiments.quantitative.exp_14_coulomb_dynamics --steps 100
Positions are local |Psi|^2 centroids, so deformation can mimic acceleration.
The four preparation labels are phases, not measured Noether charges.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import numpy as np

from lfm import Simulation
from lfm.config_presets import gravity_em


def fit_acceleration(times, positions):
    """Fit x=x0+v*t+a*t^2/2 without assuming zero initial velocity."""
    t = np.asarray(times, dtype=float)
    x = np.asarray(positions, dtype=float)
    if t.ndim != 1 or x.shape != t.shape or len(t) < 6:
        raise ValueError("Need at least six matching time/position samples")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or np.any(np.diff(t) <= 0):
        raise ValueError("Nonfinite data or non-increasing times")
    design = np.column_stack((np.ones_like(t), t - t[0], (t - t[0]) ** 2 / 2))
    coeff, _, rank, _ = np.linalg.lstsq(design, x, rcond=None)
    if rank != 3:
        raise ValueError("Degenerate time window")
    residual = x - design @ coeff
    covariance = np.linalg.inv(design.T @ design) * np.sum(residual**2) / (len(t) - 3)
    return {
        "acceleration": float(coeff[2]),
        "fit_se": float(np.sqrt(covariance[2, 2])),
        "rms_residual": float(np.sqrt(np.mean(residual**2))),
    }


def scaling_gate(distances, magnitudes):
    """Predeclared screen: >=4 distances over >=2x range, p=-2 +/-0.2, R2>=.98."""
    d = np.asarray(distances, dtype=float)
    a = np.asarray(magnitudes, dtype=float)
    valid = (
        d.ndim == 1
        and d.shape == a.shape
        and len(d) >= 4
        and np.all(np.isfinite(d))
        and np.all(np.isfinite(a))
        and np.all(d > 0)
        and np.all(a > 0)
        and len(np.unique(d)) == len(d)
    )
    if not valid or d.max() / d.min() < 2:
        return {"passed": False, "reason": "insufficient finite positive distance range"}
    slope, intercept = np.polyfit(np.log(d), np.log(a), 1)
    residual = np.log(a) - (slope * np.log(d) + intercept)
    variance = np.sum((np.log(a) - np.mean(np.log(a))) ** 2)
    r2 = 1 - np.sum(residual**2) / variance if variance > 0 else 0.0
    return {
        "passed": bool(abs(slope + 2) <= 0.2 and r2 >= 0.98),
        "exponent": float(slope),
        "r_squared": float(r2),
    }


def adjudicate(rows, *, signed_sources=False, converged=False, stable_objects=False):
    """Equal-mass axial acceleration screen: left positive points toward right.

    Each row carries independently isolated-baseline-subtracted accelerations.
    Phase labels alone must NEVER set signed_sources=True. Certification inputs
    require independently reviewed charge, stability, and convergence evidence.
    """
    gates = {
        "signed_sources": signed_sources,
        "numerical_convergence": converged,
        "stable_objects": stable_objects,
    }
    signs, reciprocity, signal = [], [], []
    scaling = {}
    for labels in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
        group = sorted(
            [r for r in rows if (r["q_left"], r["q_right"]) == labels],
            key=lambda r: r["separation"],
        )
        for r in group:
            left, right = r["a_left"], r["a_right"]
            se = r["uncertainty"]
            finite = bool(np.all(np.isfinite([left, right, se])) and se >= 0)
            scale = max(abs(left), abs(right), 1e-30) if finite else 1e-30
            signs.append(
                finite and left * labels[0] * labels[1] < 0 and right * labels[0] * labels[1] > 0
            )
            reciprocity.append(finite and abs(left + right) / scale <= 0.1)
            signal.append(finite and min(abs(left), abs(right)) > 5 * se)
        scaling[str(labels)] = scaling_gate(
            [r["separation"] for r in group],
            [(abs(r["a_left"]) + abs(r["a_right"])) / 2 for r in group],
        )
    gates.update(
        signed_response=bool(signs) and all(signs),
        equal_mass_reciprocity=bool(reciprocity) and all(reciprocity),
        resolved_signal=bool(signal) and all(signal),
        inverse_square=all(g["passed"] for g in scaling.values()),
    )
    # Simultaneous sign reversal and source swap must preserve magnitudes.
    lookup = {(r["q_left"], r["q_right"], r["separation"]): r for r in rows}
    symmetric = len(lookup) == len(rows) and bool(rows)
    for key, r in lookup.items():
        for labels in ((-key[0], -key[1]), (key[1], key[0])):
            partner = lookup.get((*labels, key[2]))
            if partner is None:
                symmetric = False
            else:
                for component in ("a_left", "a_right"):
                    a, b = abs(r[component]), abs(partner[component])
                    symmetric &= bool(np.isfinite(a + b) and abs(a - b) <= 0.1 * max(a, b, 1e-30))
    couplings = np.asarray(
        [(abs(r["a_left"]) + abs(r["a_right"])) * r["separation"] ** 2 / 2 for r in rows]
    )
    gates["common_unit_charge_coupling"] = bool(
        len(couplings)
        and np.all(np.isfinite(couplings))
        and np.all(couplings > 0)
        and (couplings.max() - couplings.min()) / np.mean(couplings) <= 0.1
    )
    gates["source_swap_and_sign_reversal"] = bool(symmetric)
    return {
        "gates": gates,
        "scaling": scaling,
        "screen_status": "PASS" if all(gates.values()) else "FAIL",
        "coulomb_status": "NOT_ESTABLISHED",
        "remaining_validation": [
            "neutral/source-magnitude controls",
            "held-out force-model discrimination",
            "independent mass and field-momentum balance",
        ],
        "maxwell_status": "NOT_TESTED",
    }


def local_centroid(sim, center, radius):
    """Fixed local window; report retention separately to reject dissolving seeds."""
    density = np.asarray(sim.psi_real, dtype=float) ** 2
    if sim.psi_imag is not None:
        density += np.asarray(sim.psi_imag, dtype=float) ** 2
    xyz = np.indices(density.shape)
    mask = sum((xyz[i] - center[i]) ** 2 for i in range(3)) <= radius**2
    w = density[mask]
    mass = float(w.sum())
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Localized seed disappeared or nonfinite density")
    return float((xyz[0][mask] * w).sum() / mass), mass


def trajectory(config, centers, phases, steps, stride, sigma):
    sim = Simulation(config, backend="cpu")
    for center, phase in zip(centers, phases, strict=True):
        sim.place_soliton(center, amplitude=6.0, sigma=sigma, phase=phase)
    sim.equilibrate()
    times, positions, norms = [], [[] for _ in centers], [[] for _ in centers]
    for step in range(0, steps + 1, stride):
        if step:
            sim.run(stride, record_metrics=False)
        times.append(step * config.dt)
        for i, center in enumerate(centers):
            x, norm = local_centroid(sim, center, 2 * sigma)
            positions[i].append(x)
            norms[i].append(norm)
    fits = [fit_acceleration(times, x) for x in positions]
    return {"times": times, "positions": positions, "local_norms": norms, "fits": fits}


def run_probe(n=48, separations=(10, 14, 18, 22), steps=100, stride=10, dt=0.01, sigma=2):
    if steps < 5 * stride or stride < 1 or steps % stride:
        raise ValueError("Need >=6 evenly sampled snapshots and steps divisible by stride")
    if min(separations) <= 4 * sigma or max(separations) / min(separations) < 2:
        raise ValueError("Need nonoverlapping tracking windows and >=2x distance range")
    if n / 2 - max(separations) / 2 - 2 * sigma <= n * 0.15:
        raise ValueError("Tracking windows too near boundary; increase grid")
    cfg = gravity_em(grid_size=n, dt=dt)
    rows, traces = [], []
    for sep in separations:
        centers = [(n / 2 - sep / 2, n / 2, n / 2), (n / 2 + sep / 2, n / 2, n / 2)]
        isolated = [trajectory(cfg, [c], [0.0], steps, stride, sigma) for c in centers]
        for labels in ((1, 1), (-1, -1), (1, -1), (-1, 1)):
            pair = trajectory(
                cfg, centers, [0 if q == 1 else np.pi for q in labels], steps, stride, sigma
            )
            fits = pair["fits"]
            baseline = [s["fits"][0] for s in isolated]
            rows.append(
                {
                    "separation": sep,
                    "q_left": labels[0],
                    "q_right": labels[1],
                    "a_left": fits[0]["acceleration"] - baseline[0]["acceleration"],
                    "a_right": fits[1]["acceleration"] - baseline[1]["acceleration"],
                    "uncertainty": max(
                        np.hypot(f["fit_se"], b["fit_se"])
                        for f, b in zip(fits, baseline, strict=True)
                    ),
                }
            )
            traces.append(
                {"separation": sep, "phase_labels": labels, "pair": pair, "isolated": isolated}
            )
    return {
        "model": "legacy_public_complex_solver_not_v39.0.2",
        "label_warning": "q labels are phases only, NOT measured signed charges",
        "config": asdict(cfg),
        "steps": steps,
        "stride": stride,
        "sigma": sigma,
        "rows": rows,
        "trajectories": traces,
        "adjudication": adjudicate(rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--grid", type=int, default=48)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--output", default="coulomb_probe.json")
    args = parser.parse_args()
    result = run_probe(n=args.grid, steps=args.steps, stride=args.stride, dt=args.dt)
    from pathlib import Path

    Path(args.output).write_text(json.dumps(result, indent=2, default=str, allow_nan=False) + "\n")
    print(json.dumps(result["adjudication"], indent=2))
