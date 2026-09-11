"""Exact plane-invariant C3/C19 flat-octic flow in canonical scaled units.

X=chi0*x/c, s=chi0*t, F=Psi/(chi0*sqrt(B)), u=chi/chi0.
No metric, force law, imposed background, or ray transport is present.
This optimization is restricted to fields constant on transverse planes.
"""
from __future__ import annotations

import numpy as np
from numba import njit
from lfm.constants import LAMBDA_H


def acceleration(f, u, h):
    """Scaled canonical RHS, for independent full-3D equivalence tests."""
    lap_f = (np.roll(f, 1, axis=0) + np.roll(f, -1, axis=0) - 2*f)/h**2
    lap_u = (np.roll(u, 1) + np.roll(u, -1) - 2*u)/h**2
    return (lap_f-u[:, None]**2*f,
            lap_u-u*np.sum(abs(f)**2, axis=1)-8*LAMBDA_H*u*(u*u-1)**3)


@njit(cache=True)
def advance(f, p_half, u, v_half, h, ds, steps, sample_stride, receivers,
            lambda_h=LAMBDA_H):
    """Drift/kick on the two equations; no field modification after a step."""
    n = len(u)
    records = steps//sample_stride
    samples = np.empty((records, len(receivers), 3), dtype=np.complex128)
    chi_samples = np.empty((records, len(receivers)), dtype=np.float64)
    chi_bounds = np.empty((records, 2), dtype=np.float64)
    r = 0
    for tick in range(1, steps+1):
        for i in range(n):
            u[i] += ds*v_half[i]
            for a in range(3):
                f[i, a] += ds*p_half[i, a]
        for i in range(n):
            left, right = (i-1) % n, (i+1) % n
            rho = 0.0
            for a in range(3):
                z = f[i, a]
                rho += z.real*z.real+z.imag*z.imag
                p_half[i, a] += ds*((f[left,a]+f[right,a]-2*z)/h**2-u[i]**2*z)
            v_half[i] += ds*((u[left]+u[right]-2*u[i])/h**2-u[i]*rho
                              -8*lambda_h*u[i]*(u[i]*u[i]-1)**3)
        if tick % sample_stride == 0:
            for j in range(len(receivers)):
                for a in range(3):
                    samples[r,j,a] = f[receivers[j],a]
                chi_samples[r,j] = u[receivers[j]]
            chi_bounds[r,0], chi_bounds[r,1] = np.min(u), np.max(u)
            r += 1
    return samples, chi_samples, chi_bounds


def invariants(f, p_half, u, v_half, h, ds):
    """Unmodified Hamiltonian and leapfrog Noether charge per scaled area."""
    af, au = acceleration(f, u, h)
    p, v = p_half-ds*af/2, v_half-ds*au/2
    energy = .5*np.sum(abs(p)**2)+.5*np.sum(v*v)
    energy += .5*np.sum(abs(np.roll(f,1,axis=0)-f)**2)/h**2
    energy += .5*np.sum((np.roll(u,1)-u)**2)/h**2
    energy += .5*np.sum(u[:,None]**2*abs(f)**2)
    energy += LAMBDA_H*np.sum((u*u-1)**4)
    return float(h*energy), float(h*np.sum(np.imag(np.conj(f)*p_half)))
