"""
LFM Composite Particle Systems: Atoms and Molecules.

Implements Approach A from the Phase 4 project plan:
  - Nuclear chi-well is set analytically (Gaussian depression in chi).
  - Electron eigenmode is solved in that FROZEN nuclear potential by running
    GOV-01 for many steps (radiation damping removes unbound components).
  - This avoids the scale-separation problem (mass ratio 1836:1) of
    simulating a full proton soliton alongside an electron on the same grid.

Physics:
  chi_nuclear(r) = chi0 - depth * exp(-r^2 / (2*sigma^2))

  This represents the proton's gravitational + strong-force chi-well.  The
  electron eigenmode solves GOV-01 with chi FROZEN at chi_nuclear.

  Reference: Paper 51 (lfm_pure_atom.py), Phase 4 project plan Section 9.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lfm.config import BoundaryType, FieldLevel
from lfm.constants import CHI0
from lfm.particles.catalog import (
    amplitude_for_particle,
    get_particle,
    sigma_for_particle,
)
from lfm.particles.solver import _energy_in_sphere
from lfm.simulation import Simulation, SimulationConfig

if TYPE_CHECKING:
    from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Per-element nuclear potential defaults
# ---------------------------------------------------------------------------
_NUCLEAR_DEPTH: dict[str, float] = {
    "H": 14.0,  # chi_min = 19 - 14 = 5
    "He": 16.0,  # deeper for Z=2
}
_NUCLEAR_SIGMA: dict[str, float] = {
    "H": 3.0,
    "He": 2.5,
}
_ELECTRON_STEPS: int = 10_000  # GOV-01-only steps for eigenmode relaxation
_SUPPORTED_ELEMENTS = frozenset({"H", "He"})
_SUPPORTED_MOLECULES = frozenset({"H2"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AtomState:
    """A solved LFM atom (nuclear chi-well + electron eigenmode).

    Attributes
    ----------
    element : str
        Chemical symbol (e.g. "H").
    sim : Simulation
        Finalised simulation containing the electron psi eigenmode.
        chi is frozen at chi_nuclear.
    chi_nuclear : NDArray
        The analytic nuclear chi-well that was used.
    electron_energy : float
        Sum of |psi|^2 inside a sphere of radius N/6 near the nucleus.
    electron_energy_free : float
        Same metric measured with flat chi = chi0 (reference; unbound).
    binding_energy : float
        fraction_near_nucleus_bound - fraction_near_nucleus_free.  Positive
        means the nuclear well increases |psi|^2 concentration near center
        relative to a flat-chi reference.  Range: [-1, 1].  A well-bound
        atom has binding_energy close to +1.
        NOTE: this uses |psi|^2 fraction as a proxy; a full stress-energy
        integral would give the proper binding energy in physical units.
    bound : bool
        True if fraction_near_nucleus >= 0.40.
    fraction_near_nucleus : float
        Fraction of total |psi|^2 inside radius N/8 of the nuclear centre.
    """

    element: str
    sim: Simulation
    chi_nuclear: NDArray
    electron_energy: float
    electron_energy_free: float
    binding_energy: float
    bound: bool
    fraction_near_nucleus: float


@dataclass
class MoleculeState:
    """A solved LFM molecule (two nuclear chi-wells + shared electron).

    Attributes
    ----------
    formula : str
        Chemical formula (e.g. "H2").
    sim : Simulation
        Finalised simulation with electron psi in the molecular potential.
    chi_nuclear : NDArray
        Combined nuclear chi-well (superposition of both proton wells).
    electron_energy : float
        Sum of |psi|^2 inside a sphere of radius bond_length around midpoint.
    bond_stable : bool
        True if electron density is present near both nuclear centres.
    proton_separation : float
        Distance between the two proton chi-wells in grid cells.
    """

    formula: str
    sim: Simulation
    chi_nuclear: NDArray
    electron_energy: float
    bond_stable: bool
    proton_separation: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def nuclear_chi_well(
    N: int,
    position: tuple[float, float, float],
    depth: float = 14.0,
    sigma: float = 3.0,
    chi0: float = CHI0,
) -> NDArray:
    """Return a chi array with a Gaussian depression at `position`.

    chi(r) = chi0 - depth * exp(-r^2 / (2 * sigma^2))

    The result is clamped to >= 1.0 to prevent Z2 vacuum flip (chi < 0).

    Parameters
    ----------
    N : grid size
    position : (cx, cy, cz) centre of the nuclear well
    depth : chi depression depth; chi_min = chi0 - depth
    sigma : well width in cells
    chi0 : background chi value
    """
    xs = np.arange(N, dtype=np.float64)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    cx, cy, cz = position
    r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    chi_arr = chi0 - depth * np.exp(-r2 / (2.0 * sigma**2))
    chi_arr = np.maximum(chi_arr, 1.0)
    return chi_arr.astype(np.float32)


def _combine_chi_wells(*wells: NDArray, chi0: float = CHI0) -> NDArray:
    """Superpose multiple chi-wells: chi_mol = chi0 - sum(chi0 - chi_i).

    The combined depression is the sum of individual depressions, clamped
    to avoid negative chi.
    """
    combined_delta = np.zeros_like(wells[0], dtype=np.float64)
    for w in wells:
        combined_delta += chi0 - w.astype(np.float64)
    result = chi0 - combined_delta
    result = np.maximum(result, 1.0)
    return result.astype(np.float32)


def _fraction_near_center(
    psi_r: NDArray,
    psi_i: NDArray | None,
    center: tuple[float, float, float],
    radius: float,
) -> float:
    """Fraction of total |psi|^2 within `radius` cells of `center`."""
    N = psi_r.shape[0]
    cx, cy, cz = center
    xs = np.arange(N, dtype=np.float64)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
    psi2 = psi_r.astype(np.float64) ** 2
    if psi_i is not None:
        psi2 = psi2 + psi_i.astype(np.float64) ** 2
    total = float(psi2.sum())
    if total < 1e-30:
        return 0.0
    near = float(psi2[r2 <= radius**2].sum())
    return near / total


def _evolve_electron_in_potential(
    N: int,
    center: tuple[float, float, float],
    chi_potential: NDArray,
    steps: int = _ELECTRON_STEPS,
) -> tuple[Simulation, float, float]:
    """
    Solve electron eigenmode in a fixed external chi potential.

    Procedure:
      1. Create Simulation with chi = chi_potential (frozen).
      2. Place electron Gaussian at `center`.
      3. Run GOV-01 only (`evolve_chi=False`) for `steps` steps.
         Radiation escapes to boundaries; bound modes survive.
      4. Return (sim, energy_in_sphere, fraction_near_center).

    Parameters
    ----------
    N : grid size
    center : nuclear centre in grid coordinates
    chi_potential : pre-built chi array (shape N^3)
    steps : GOV-01-only relaxation steps

    Returns
    -------
    sim : Simulation with electron psi eigenmode
    energy : sum of |psi|^2 in sphere of radius N/6 around center
    fraction : fraction of total |psi|^2 within radius N/8
    """
    electron = get_particle("electron")
    amp = amplitude_for_particle(electron, N)
    sig = sigma_for_particle(electron, N)

    config = SimulationConfig(
        grid_size=N,
        field_level=FieldLevel.COMPLEX,
        boundary_type=BoundaryType.FROZEN,
        chi0=CHI0,
    )
    sim = Simulation(config)

    # Set nuclear chi BEFORE placing electron psi (no equilibrate)
    sim.set_chi(chi_potential)

    # Seed electron Gaussian at nuclear center
    sim.place_soliton(
        position=center,
        amplitude=amp,
        sigma=sig,
        phase=0.0,
    )

    # Relax electron in frozen nuclear potential
    sim.run(steps, evolve_chi=False, record_metrics=False)

    # Measure energy retention in sphere
    radius_e = float(N) / 6.0
    center_i = (int(center[0]), int(center[1]), int(center[2]))
    energy = _energy_in_sphere(sim.psi_real, sim.psi_imag, sim.chi, center_i, radius_e)

    # Fraction near center (binding indicator)
    frac = _fraction_near_center(sim.psi_real, sim.psi_imag, center, float(N) / 8.0)

    return sim, float(energy), frac


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_atom(
    element: str = "H",
    N: int = 64,
    nuclear_depth: float | None = None,
    nuclear_sigma: float | None = None,
    steps: int = _ELECTRON_STEPS,
) -> AtomState:
    """Create an LFM atom using Approach A (effective nuclear potential).

    The proton is replaced by an analytic Gaussian chi-well:
        chi_nuc(r) = chi0 - depth * exp(-r^2 / (2*sigma^2))

    The electron eigenmode is solved by running GOV-01 for `steps` steps
    with chi frozen at chi_nuc.  Unbound radiation escapes; the residual
    psi is the bound eigenmode.

    Binding is verified by comparing ``fraction_near_nucleus`` against
    a reference run with flat chi (no nuclear potential).

    Parameters
    ----------
    element : "H" or "He"
    N : grid size (64 recommended)
    nuclear_depth : depth of chi depression (default per element)
    nuclear_sigma : width of chi-well in cells (default per element)
    steps : GOV-01-only steps for electron relaxation

    Returns
    -------
    AtomState

    Raises
    ------
    ValueError
        If `element` is not in the supported set.
    """
    if element not in _SUPPORTED_ELEMENTS:
        raise ValueError(f"element must be one of {sorted(_SUPPORTED_ELEMENTS)}; got {element!r}")

    depth = nuclear_depth if nuclear_depth is not None else _NUCLEAR_DEPTH[element]
    sigma = nuclear_sigma if nuclear_sigma is not None else _NUCLEAR_SIGMA[element]

    half = float(N // 2)
    center = (half, half, half)

    # Build nuclear chi-well
    chi_nuc = nuclear_chi_well(N, center, depth=depth, sigma=sigma)

    # Solve electron in nuclear well (bound state)
    sim_bound, E_bound, frac_bound = _evolve_electron_in_potential(N, center, chi_nuc, steps=steps)

    # Reference: electron in FLAT chi=chi0 (free/unbound baseline)
    flat_chi = np.full((N, N, N), float(CHI0), dtype=np.float32)
    _, E_free, frac_free = _evolve_electron_in_potential(N, center, flat_chi, steps=steps)

    # Binding criterion: fraction_near_nucleus >= 0.40 in nuclear case
    bound = frac_bound >= 0.40
    # Positive when nuclear well concentrates more psi^2 near center than free case
    binding_energy = float(frac_bound - frac_free)

    return AtomState(
        element=element,
        sim=sim_bound,
        chi_nuclear=chi_nuc,
        electron_energy=float(E_bound),
        electron_energy_free=float(E_free),
        binding_energy=binding_energy,
        bound=bound,
        fraction_near_nucleus=frac_bound,
    )


def create_molecule(
    formula: str = "H2",
    N: int = 128,
    bond_length: float | None = None,
    nuclear_depth: float = 14.0,
    nuclear_sigma: float = 3.0,
    steps: int = _ELECTRON_STEPS,
) -> MoleculeState:
    """Create an LFM molecule with two nuclear chi-wells (H₂).

    Two proton chi-wells are placed symmetrically along the x-axis separated
    by `bond_length` grid cells.  The electron eigenmode is solved in the
    combined frozen potential.

    Parameters
    ----------
    formula : "H2"
    N : grid size (128 recommended to fit two nuclei + margin)
    bond_length : proton-proton separation in cells (default = N/8)
    nuclear_depth : depth of each proton chi-well
    nuclear_sigma : width of each proton chi-well in cells
    steps : GOV-01-only relaxation steps

    Returns
    -------
    MoleculeState

    Raises
    ------
    ValueError
        If `formula` is not "H2".
    """
    if formula not in _SUPPORTED_MOLECULES:
        raise ValueError(f"formula must be one of {sorted(_SUPPORTED_MOLECULES)}; got {formula!r}")

    bl = bond_length if bond_length is not None else max(10.0, float(N) / 8.0)
    half = float(N // 2)

    # Nuclear positions: symmetric about grid centre along x-axis
    x1 = half - bl / 2.0
    x2 = half + bl / 2.0
    pos1 = (x1, half, half)
    pos2 = (x2, half, half)
    midpoint = (half, half, half)

    # Build individual and combined chi-wells
    chi1 = nuclear_chi_well(N, pos1, depth=nuclear_depth, sigma=nuclear_sigma)
    chi2 = nuclear_chi_well(N, pos2, depth=nuclear_depth, sigma=nuclear_sigma)
    chi_mol = _combine_chi_wells(chi1, chi2)

    # Solve electron in molecular potential
    sim_mol, E_mol, frac_mid = _evolve_electron_in_potential(N, midpoint, chi_mol, steps=steps)

    # Check if electron density reaches BOTH nuclear sites
    pr = sim_mol.psi_real
    pi_ = sim_mol.psi_imag if sim_mol.psi_imag is not None else np.zeros_like(pr)
    psi2 = pr.astype(np.float64) ** 2 + pi_.astype(np.float64) ** 2
    float(psi2.sum())

    r_check = nuclear_sigma * 3.0  # count psi^2 within 3*sigma of each nucleus
    frac1 = _fraction_near_center(pr, pi_, pos1, r_check)
    frac2 = _fraction_near_center(pr, pi_, pos2, r_check)
    bond_stable = (frac1 >= 0.02) and (frac2 >= 0.02)

    return MoleculeState(
        formula=formula,
        sim=sim_mol,
        chi_nuclear=chi_mol,
        electron_energy=float(E_mol),
        bond_stable=bond_stable,
        proton_separation=bl,
    )
