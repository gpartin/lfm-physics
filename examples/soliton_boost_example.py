"""Example: create a gaussian "soliton" and boost it two ways:
1) fractional FFT-based shift applied to `E_prev` (real-field approach)
2) complex phase boost and corresponding `E_prev` (phase approach)

Compare center-of-mass drift over short evolution using leapfrog with a simple 1D laplacian.
"""
import numpy as np
import importlib.util
from pathlib import Path

# Import particles module by file path (handles hyphen in folder name)
spec = importlib.util.spec_from_file_location(
    "lfm_particles", str(Path(__file__).resolve().parent.parent / "particles.py")
)
particles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(particles)


def laplacian(u, dx=1.0):
    return (np.roll(u, -1) - 2*u + np.roll(u, 1)) / (dx*dx)


def com(field):
    x = np.arange(field.size)
    m = (field*field).sum()
    if m == 0:
        return 0.0
    return (x * (field*field)).sum() / m


def run_evolution(E_init, E_prev_init, steps=800, dt=0.02):
    E = E_init.copy()
    E_prev = E_prev_init.copy()
    coms = []
    for n in range(steps):
        lap = laplacian(E)
        E_next = 2*E - E_prev + (dt**2) * (lap - 0.0 * E)
        E_prev = E
        E = E_next
        if n % 20 == 0:
            coms.append(com(E))
    return coms


if __name__ == '__main__':
    N = 256
    x0 = N//4
    sigma = 6.0
    E = particles.create_gaussian_1d(N, x0, sigma, A=1.0)

    # Method A: fractional shift for E_prev corresponding to desired velocity
    v = 0.05  # desired velocity (samples per unit time)
    dt = 0.02
    shift_samples = v * dt  # fraction of cell per timestep
    E_prev_A = particles.fractional_shift_1d(E, -shift_samples)  # approximate backward step
    coms_A = run_evolution(E, E_prev_A, steps=800, dt=dt)

    # Method B: complex phase boost
    k = 2*np.pi * 0.02
    E_boosted, E_prev_B = particles.boost_complex_from_real(E, k=k, dt=dt)
    coms_B = run_evolution(E_boosted, E_prev_B, steps=800, dt=dt)

    print('Method A (fractional shift):', coms_A[0], '->', coms_A[-1], 'Δ=', coms_A[-1]-coms_A[0])
    print('Method B (phase boost):    ', coms_B[0], '->', coms_B[-1], 'Δ=', coms_B[-1]-coms_B[0])

    np.savez(Path(__file__).parent / 'soliton_boost_results.npz', coms_A=coms_A, coms_B=coms_B)
