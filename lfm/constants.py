"""
LFM Fundamental Constants
=========================

ALL parameters derived from χ₀ = 19 (3D lattice geometry: 1+6+12 = 19).
This is the SINGLE SOURCE OF TRUTH for LFM constants.

Derivation chain:
    Axiom 1: Simple cubic lattice, D=3 (observation)
    Axiom 2: Discrete wave update (leapfrog, 19-point stencil)
    Axiom 3: Unit coupling (Δx = c = ℏ = 1)

    χ₀ = 3^D − 2^D = 19        (non-propagating lattice modes)
    κ  = 1/(4^D − 1) = 1/63    (gravity coupling)
    λ_H = D_st/(2·D_st² − 1)   (Higgs self-coupling)
    ε_W = 2/(χ₀ + 1) = 0.1     (weak/helicity coupling)
"""

from __future__ import annotations

import math

# ============================================================
# AXIOMS — From lattice geometry
# ============================================================

D: int = 3
"""Spatial dimensions (observation selects D=3)."""

D_ST: int = D + 1
"""Spacetime dimensions."""

CHI0: float = float(3**D - 2**D)
"""Background χ value = 19.0. From 3D discrete Laplacian: 1 center + 6 face + 12 edge modes."""

N_COLORS: int = 3
"""Number of color components (Ψₐ, a = 1,2,3)."""

# ============================================================
# DERIVED — From χ₀ and lattice geometry
# ============================================================

KAPPA: float = 1.0 / (4**D - 1)
"""Gravity coupling = 1/63. From unit-coupling axiom on N=4 unit cell."""

LAMBDA_H: float = D_ST / (2 * D_ST**2 - 1)
"""Higgs self-coupling = 4/31 ≈ 0.12903. From z₂ = 2·D_st² geometry."""

EPSILON_W: float = 2.0 / (CHI0 + 1)
"""Weak/helicity coupling = 0.1. Factorizes as sin²θ_W × 4/(χ₀−4)."""

ALPHA_S: float = 2.0 / (CHI0 - 2)
"""Strong coupling at M_Z = 2/17 ≈ 0.1176. Numerator = rank(SU(3))."""

# ============================================================
# v14+ — Color variance and cross-color coupling
# ============================================================

KAPPA_C: float = KAPPA / N_COLORS
"""Color variance coupling = κ/N_c = 1/189 ≈ 0.005291 (v14).
Extra χ deepening for non-singlet color configurations."""

EPSILON_CC: float = ALPHA_S
"""Cross-color coupling = α_s = 2/17 ≈ 0.1176 (v15).
Drives f_c dynamic: −ε_cc·χ²·(Ψₐ − Ψ̄) in GOV-01."""

# ============================================================
# v16 — S_a auxiliary fields (flux tube / confinement)
# ============================================================

BETA_0: int = int(CHI0) - 12
"""QCD β₀ coefficient = 7 = χ₀ − 12. Also the S_a diffusion range L."""

SA_GAMMA: float = EPSILON_W
"""S_a decay rate γ = ε_W = 0.1.
Same physics as weak helicity coupling: both derived from 2/(χ₀+1)."""

SA_L: int = BETA_0
"""S_a coherence / diffusion range L = β₀ = 7 lattice units.
Sets the spatial scale of flux tubes."""

SA_D: float = SA_GAMMA * SA_L**2
"""S_a diffusion coefficient D = γ·L² = 0.1 × 49 = 4.9.
CFL for S_a Euler step: dt < 1/(6D) ≈ 0.034. Our dt=0.02 is safe."""

Z2_COORD: int = 2 * D_ST**2
"""Second coordination shell z₂ = 2·D_st² = 32 (face+edge+NNN on 4D hypercubic)."""

RANK_SU3: int = N_COLORS - 1
"""Rank of SU(3) gauge group = 2 = N_c − 1. Appears in κ_tube derivation."""

KAPPA_TUBE: float = (Z2_COORD - RANK_SU3) * KAPPA
"""Smoothed color variance (SCV) coupling κ_tube = (z₂ − rank_G)·κ = 30/63 ≈ 0.4762.
All parameters derived from χ₀=19 (Session 143).
At this full value the simulation requires λ_self=LAMBDA_H for stability.
For standalone experiments without Mexican hat use 10*KAPPA for safety."""

KAPPA_STRING: float = KAPPA_C
"""Color current variance (CCV) coupling κ_string = κ_c = 1/189.
Activates v15-style CCV term in GOV-02."""

# ============================================================
# NUMERICAL — Simulation defaults
# ============================================================

DT_DEFAULT: float = 0.02
"""Default timestep. Safe for both CFL and Mexican hat resolution."""

DT_MOTION: float = 0.005
"""Recommended timestep for moving solitons (4× finer than default).

Leapfrog integration of boosted solitons requires smaller dt to
maintain > 80% velocity retention across all particle types."""

C_DEFAULT: float = 1.0
"""Wave speed in natural lattice units."""

E0_SQ_DEFAULT: float = 0.0
"""Background energy density (0 = vacuum, no cosmological constant source)."""

BOUNDARY_FRACTION_DEFAULT: float = 0.3
"""Fraction of grid radius used for frozen boundary shell."""

# 19-point stencil CFL limit: dt < dx / (c * sqrt(16/3))
CFL_19PT: float = 1.0 / math.sqrt(16.0 / 3.0)
"""CFL stability limit for 19-point stencil ≈ 0.434 (massless, Δx=c=1)."""

CFL_19PT_MASSIVE: float = 1.0 / math.sqrt(16.0 / 3.0 + CHI0**2)
"""CFL limit with mass term ≈ 0.051 (for Δx=c=1, χ₀=19). Our dt=0.02 is safe."""

# ============================================================
# STENCIL WEIGHTS — 19-point isotropic Laplacian
# ============================================================

STENCIL_FACE_WEIGHT: float = 1.0 / 3.0
"""Weight for 6 face neighbors (distance 1). O(h^4) isotropy (19-pt)."""

STENCIL_EDGE_WEIGHT: float = 1.0 / 6.0
"""Weight for 12 edge neighbors (distance sqrt(2)). O(h^4) isotropy (19-pt)."""

STENCIL_CENTER_WEIGHT: float = -4.0
"""Center coefficient = -(6x(1/3) + 12x(1/6)) = -4 (19-pt)."""

# ============================================================
# STENCIL WEIGHTS -- 7-point Laplacian (face only)
# ============================================================

STENCIL_7PT_FACE_WEIGHT: float = 1.0
"""Weight for 6 face neighbors (distance 1). O(h^2) accuracy, 12.3% anisotropy."""

STENCIL_7PT_CENTER_WEIGHT: float = -6.0
"""Center coefficient = -(6x1) = -6 (7-pt)."""

# ============================================================
# STENCIL WEIGHTS -- 27-point full-cube Laplacian (face+edge+corner)
# ============================================================

STENCIL_27PT_FACE_WEIGHT: float = 1.0 / 2.0
"""Weight for 6 face neighbors (distance 1). O(h^4) isotropic 27-pt stencil."""

STENCIL_27PT_EDGE_WEIGHT: float = 1.0 / 12.0
"""Weight for 12 edge neighbors (distance sqrt(2)). O(h^4) isotropic 27-pt stencil."""

STENCIL_27PT_CORNER_WEIGHT: float = 1.0 / 24.0
"""Weight for 8 corner neighbors (distance sqrt(3)). O(h^4) isotropic 27-pt stencil."""

STENCIL_27PT_CENTER_WEIGHT: float = -13.0 / 3.0
"""Center coefficient = -(6x(1/2) + 12x(1/12) + 8x(1/24)) = -13/3 (27-pt)."""

# ============================================================
# PER-GRID AMPLITUDE SCALING
# ============================================================

E_AMPLITUDE_BY_GRID: dict[int, float] = {
    32: 12.0,
    64: 6.0,
    128: 3.6,
    256: 1.8,
    512: 0.9,
}
"""Empirically tuned E amplitudes that prevent NaN for each grid size."""

# ============================================================
# MODE STRUCTURE — 3D BZ cube decomposition
# ============================================================

N_FACE_MODES: int = 2 * D
"""6 face modes (±1,0,0), (0,±1,0), (0,0,±1)."""

N_EDGE_MODES: int = 2 * D * (D - 1)
"""12 edge modes (±1,±1,0) and permutations."""

N_CORNER_MODES: int = 2**D
"""8 corner modes (±1,±1,±1). Maps to N_gluons."""

N_DC_MODE: int = 1
"""1 DC mode (0,0,0). Background χ₀."""

N_NON_PROPAGATING: int = N_DC_MODE + N_FACE_MODES + N_EDGE_MODES
"""1 + 6 + 12 = 19 = χ₀. Non-propagating modes define vacuum stiffness."""

# ============================================================
# STRUCTURE THRESHOLDS
# ============================================================

WELL_THRESHOLD: float = 17.0
"""χ < WELL_THRESHOLD marks gravitational well regions."""

VOID_THRESHOLD: float = 18.0
"""χ > VOID_THRESHOLD marks void regions."""

# ============================================================
# PREDICTIONS — All derived from χ₀ = 19
# ============================================================

ALPHA_EM: float = (CHI0 - 8) / (480 * math.pi)
"""Fine structure constant ≈ 1/137.088 (measured: 1/137.036, error 0.04%)."""

OMEGA_LAMBDA: float = (CHI0 - 2 * D) / CHI0
"""Dark energy fraction = 13/19 ≈ 0.6842 (measured: 0.685, error 0.12%)."""

OMEGA_MATTER: float = (2 * D) / CHI0
"""Matter fraction = 6/19 ≈ 0.3158 (measured: 0.315, error 0.25%)."""

SIN2_THETA_W: float = 3.0 / (CHI0 - 11)
"""Weak mixing angle sin²θ_W = 3/8 = 0.375 (GUT scale, exact)."""

N_GENERATIONS: int = int((CHI0 - 1) // 6)
"""Number of fermion generations = 3 (exact)."""

N_GLUONS: int = int(CHI0 - 11)
"""Number of gluons = 8 (exact). Equals corner modes 2^D."""

N_EFOLDINGS: int = int(D * (CHI0 + 1))
"""Inflation e-folds = 60 (exact). D × (χ₀ + 1)."""

CABIBBO_ANGLE_SIN: float = math.sqrt(1.0 / (CHI0 + 1))
"""sin(θ_C) = 1/√20 ≈ 0.2236 (measured: 0.2257, error 0.9%)."""

# ============================================================
# MEXICAN HAT POTENTIAL
# ============================================================

MEXICAN_HAT_CURVATURE: float = 8 * LAMBDA_H * CHI0**2
"""V''(χ₀) ≈ 373. Stiffness of vacuum around χ₀ = 19."""

HIGGS_OSCILLATION_FREQ: float = math.sqrt(8 * LAMBDA_H) * CHI0
"""ω_H = √(8λ_H)·χ₀ ≈ 19.30. Natural oscillation of χ around χ₀."""

MEXICAN_HAT_PERIOD: float = 2 * math.pi / HIGGS_OSCILLATION_FREQ
"""T_MH ≈ 0.326 lattice time units. Need Δt < T_MH/10 for accuracy."""

# ============================================================
# PARAMETRIC RESONANCE
# ============================================================

RESONANCE_FREQUENCY: float = 2 * CHI0
"""Ω = 2χ₀ = 38. Parametric resonance frequency for matter creation."""

# ============================================================
# COSMOLOGICAL SCALE
# ============================================================

PLANCK_TIME_SEC: float = 5.391e-44
"""Planck time in seconds (SI measurement)."""

PLANCK_LENGTH_M: float = 1.616e-35
"""Planck length in meters (SI measurement)."""

AGE_UNIVERSE_GYR: float = 13.8
"""Age of the observable universe in Gyr.

OBSERVED INPUT — not derived by LFM. LFM derives universe SIZE given this
age, not the age itself. Calibrated from canonical 256³ simulation:
541,000 steps ≅ 13.8 Gyr.
"""

_SEC_PER_GYR: float = 1e9 * 365.25 * 24 * 3600

OBSERVABLE_RADIUS_PLANCK: float = (AGE_UNIVERSE_GYR * _SEC_PER_GYR) / PLANCK_TIME_SEC
"""Observable universe radius in Planck cells ≈ 8.07×10⁶⁰.

Since c = 1 cell/tick in LFM natural units, radius = age in Planck ticks.
Uses AGE_UNIVERSE_GYR as input.
"""

TOTAL_RADIUS_LOWER_BOUND_PLANCK: float = math.exp(N_EFOLDINGS) * OBSERVABLE_RADIUS_PLANCK
"""Lower bound on total universe radius in Planck cells ≈ 9×10⁸⁶.

DERIVED: e^N_EFOLDINGS × OBSERVABLE_RADIUS_PLANCK.
The ratio e^60 ≈ 10²⁶ is fully LFM-derived from χ₀ = 19.
"""
