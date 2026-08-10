"""Native one-link gradient-operator spectroscopy for full-R2 LFM.

The operator basis is complete for Hermitian one-link bilinears of the three
complex GOV-01 components. It is an internal readout of the existing site
fields, not an independent gauge register or an added equation of motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lfm.analysis.energy_current import stencil_links


def u3_hermitian_generators() -> tuple[tuple[str, ...], np.ndarray]:
    """Return T0 and the eight Gell-Mann matrices with Tr(TA TB)=2 deltaAB."""

    zero = 0.0j
    one = 1.0 + 0.0j
    generators = [math.sqrt(2.0 / 3.0) * np.eye(3, dtype=np.complex128)]
    generators.extend(
        [
            np.array([[zero, one, zero], [one, zero, zero], [zero, zero, zero]]),
            np.array([[zero, -1j, zero], [1j, zero, zero], [zero, zero, zero]]),
            np.array([[one, zero, zero], [zero, -one, zero], [zero, zero, zero]]),
            np.array([[zero, zero, one], [zero, zero, zero], [one, zero, zero]]),
            np.array([[zero, zero, -1j], [zero, zero, zero], [1j, zero, zero]]),
            np.array([[zero, zero, zero], [zero, zero, one], [zero, one, zero]]),
            np.array([[zero, zero, zero], [zero, zero, -1j], [zero, 1j, zero]]),
            np.diag([1.0, 1.0, -2.0]).astype(np.complex128) / math.sqrt(3.0),
        ]
    )
    names = ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
    return names, np.stack(generators, axis=0)


def operator_completeness_audit(seed: int = 20260727) -> dict[str, float | bool]:
    """Numerically audit orthonormality and Hermitian-matrix reconstruction."""

    _, generators = u3_hermitian_generators()
    gram = np.einsum("aij,bji->ab", generators, generators).real
    orthonormal_error = float(np.max(np.abs(gram - 2.0 * np.eye(9))))
    rng = np.random.default_rng(seed)
    trial = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    hermitian = 0.5 * (trial + trial.conj().T)
    coefficients = 0.5 * np.einsum("aij,ji->a", generators, hermitian)
    reconstructed = np.einsum("a,aij->ij", coefficients, generators)
    reconstruction_error = float(np.max(np.abs(reconstructed - hermitian)))
    return {
        "generator_count": 9.0,
        "trace_orthonormality_max_error": orthonormal_error,
        "hermitian_reconstruction_max_error": reconstruction_error,
        "pass": orthonormal_error < 1.0e-12 and reconstruction_error < 1.0e-12,
    }


def _link_bilinears(
    psi: np.ndarray,
    shift: tuple[int, int, int, int],
) -> np.ndarray:
    _, generators = u3_hermitian_generators()
    neighbor = np.roll(psi, shift=shift, axis=(0, 1, 2, 3))
    return np.imag(
        np.einsum(
            "...a,gab,...b->...g",
            np.conj(psi),
            generators,
            neighbor,
            optimize=True,
        )
    )


def _link_berry(
    psi: np.ndarray,
    chi: np.ndarray,
    shift: tuple[int, int, int, int],
) -> np.ndarray:
    neighbor_psi = np.roll(psi, shift=shift, axis=(0, 1, 2, 3))
    neighbor_chi = np.roll(chi, shift=shift, axis=(0, 1, 2, 3))
    overlap = chi * neighbor_chi + np.sum(np.conj(psi) * neighbor_psi, axis=-1)
    return np.angle(overlap)


def native_gradient_operators(psi: np.ndarray, chi: np.ndarray) -> np.ndarray:
    """Return the registered 19-channel site-centered four-vector basis.

    Channels 0:9 are the raw U(3) currents, channels 9:18 are their regular
    full-state-normalized counterparts, and channel 18 is the regular Berry
    link. The final axis is (time, x, y, z).
    """

    values = np.asarray(psi, dtype=np.complex128)
    chi_values = np.asarray(chi, dtype=float)
    if values.ndim != 5 or values.shape[-1] != 3:
        raise ValueError("psi must have shape (Lt,Lx,Ly,Lz,3)")
    if chi_values.shape != values.shape[:-1]:
        raise ValueError("chi must match the four-dimensional psi lattice")

    shape = values.shape[:-1] + (9, 4)
    raw = np.zeros(shape, dtype=np.float64)
    regular = np.zeros(shape, dtype=np.float64)
    berry = np.zeros(values.shape[:-1] + (1, 4), dtype=np.float64)
    norms = np.sqrt(chi_values**2 + np.sum(np.abs(values) ** 2, axis=-1))

    spacetime_links: list[tuple[tuple[int, int, int, int], float]] = [
        ((-1, 0, 0, 0), 1.0),
        ((1, 0, 0, 0), 1.0),
    ]
    for offset, weight in stencil_links("19"):
        spacetime_links.append(((0, offset[0], offset[1], offset[2]), weight))

    for shift, weight in spacetime_links:
        displacement = np.asarray(shift, dtype=float)
        link = _link_bilinears(values, shift)
        neighbor_norm = np.roll(norms, shift=shift, axis=(0, 1, 2, 3))
        denominator = np.maximum(norms * neighbor_norm, 1.0e-300)
        regular_link = link / denominator[..., None]
        berry_link = _link_berry(values, chi_values, shift)
        for component in range(4):
            coefficient = weight * displacement[component]
            if coefficient == 0.0:
                continue
            raw[..., component] += coefficient * link
            regular[..., component] += coefficient * regular_link
            berry[..., 0, component] += coefficient * berry_link
    return np.concatenate((raw, regular, berry), axis=-2)


def operator_family_slices() -> dict[str, slice]:
    """Return fixed channel slices for preregistered operator families."""

    return {
        "raw_u3": slice(0, 9),
        "regular_u3": slice(9, 18),
        "regular_berry": slice(18, 19),
        "combined": slice(0, 19),
    }


def extract_low_momentum_modes(operators: np.ndarray) -> dict[str, np.ndarray]:
    """Extract registered transverse, longitudinal, temporal, and cone modes."""

    vector = np.asarray(operators, dtype=float)
    if vector.ndim != 6 or vector.shape[-1] != 4:
        raise ValueError("operators must have shape (Lt,Lx,Ly,Lz,C,4)")
    if len(set(vector.shape[:4])) != 1:
        raise ValueError("registered spectroscopy uses equal four-dimensional extents")
    size = vector.shape[0]
    transformed = np.fft.fftn(vector, axes=(0, 1, 2, 3), norm="ortho")

    def spatial_modes(harmonic: int, temporal: int) -> tuple[np.ndarray, ...]:
        transverse = []
        longitudinal = []
        pol0 = []
        pol1 = []
        temporal_values = []
        for direction in range(3):
            index = [temporal, 0, 0, 0]
            index[direction + 1] = harmonic % size
            sample = transformed[tuple(index)]
            spatial_components = [axis for axis in range(3) if axis != direction]
            pol0.append(sample[:, spatial_components[0] + 1])
            pol1.append(sample[:, spatial_components[1] + 1])
            transverse.extend(
                [
                    sample[:, spatial_components[0] + 1],
                    sample[:, spatial_components[1] + 1],
                ]
            )
            longitudinal.append(sample[:, direction + 1])
            temporal_values.append(sample[:, 0])
        return (
            np.stack(transverse),
            np.stack(longitudinal),
            np.stack(pol0),
            np.stack(pol1),
            np.stack(temporal_values),
        )

    t1, l1, p01, p11, a01 = spatial_modes(1, 0)
    t2, l2, _p02, _p12, a02 = spatial_modes(2, 0)
    cone, _cone_l, _cone_p0, _cone_p1, _cone_a0 = spatial_modes(1, 1)
    return {
        "transverse_k1": t1,
        "transverse_k2": t2,
        "longitudinal_k1": l1,
        "polarization_0_k1": p01,
        "polarization_1_k1": p11,
        "cone_p1_k1": cone,
        "temporal_k1": a01,
        "temporal_k2": a02,
    }


def _covariance(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples, dtype=np.complex128)
    flattened = values.reshape(-1, values.shape[-1])
    flattened = flattened - np.mean(flattened, axis=0, keepdims=True)
    covariance = np.real(flattened.conj().T @ flattened) / max(len(flattened), 1)
    return 0.5 * (covariance + covariance.T)


def _leading_generalized_mode(
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c_low = _covariance(low)
    c_high = _covariance(high)
    scale = float(np.trace(c_high) / max(c_high.shape[0], 1))
    ridge = max(scale * 1.0e-8, 1.0e-300)
    eigenvalues, eigenvectors = np.linalg.eigh(c_high + ridge * np.eye(c_high.shape[0]))
    keep = eigenvalues > max(eigenvalues[-1] * 1.0e-10, ridge * 0.1)
    whitening = eigenvectors[:, keep] / np.sqrt(eigenvalues[keep])[None, :]
    reduced = whitening.T @ c_low @ whitening
    ratios, modes = np.linalg.eigh(0.5 * (reduced + reduced.T))
    order = np.argsort(ratios)[::-1]
    vector = whitening @ modes[:, order[0]]
    vector /= max(float(np.linalg.norm(vector)), 1.0e-300)
    dominant = int(np.argmax(np.abs(vector)))
    if vector[dominant] < 0.0:
        vector = -vector
    return vector, ratios[order]


def _power(samples: np.ndarray, vector: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.complex128)
    projection = values @ vector
    return float(np.mean(np.abs(projection) ** 2))


def expected_massless_ir_ratio(size: int) -> float:
    """Return the canonical lattice 1/k^2 ratio between harmonics one and two."""

    k = 2.0 * math.pi / size
    stiffness_1 = 4.0 * math.sin(0.5 * k) ** 2
    stiffness_2 = 4.0 * math.sin(k) ** 2
    return stiffness_2 / stiffness_1


@dataclass(frozen=True)
class CrossValidatedMode:
    """Held-out statistics for one registered operator family and volume."""

    family: str
    size: int
    expected_ir_ratio: float
    heldout_ir_ratios: tuple[float, float]
    heldout_k1_powers: tuple[float, float]
    heldout_cone_ratios: tuple[float, float]
    heldout_longitudinal_ratios: tuple[float, float]
    heldout_polarization_splits: tuple[float, float]
    train_eigenvalue_counts: tuple[int, int]
    vectors: tuple[np.ndarray, np.ndarray]

    @property
    def mean_ir_ratio(self) -> float:
        return float(np.mean(self.heldout_ir_ratios))

    @property
    def mean_k1_power(self) -> float:
        return float(np.mean(self.heldout_k1_powers))


def cross_validated_mode(
    sample_modes: dict[str, np.ndarray],
    family: str,
    size: int,
) -> CrossValidatedMode:
    """Select on one sample half and evaluate on the held-out half both ways."""

    family_slice = operator_family_slices()[family]
    selected = {key: np.asarray(value)[..., family_slice] for key, value in sample_modes.items()}
    sample_count = selected["transverse_k1"].shape[0]
    halves = (np.arange(sample_count) % 2 == 0, np.arange(sample_count) % 2 == 1)
    expected = expected_massless_ir_ratio(size)
    ir_ratios = []
    k1_powers = []
    cone_ratios = []
    longitudinal_ratios = []
    polarization_splits = []
    eigenvalue_counts = []
    vectors = []
    for train_mask, test_mask in (halves, halves[::-1]):
        vector, eigenvalues = _leading_generalized_mode(
            selected["transverse_k1"][train_mask],
            selected["transverse_k2"][train_mask],
        )
        k1_power = _power(selected["transverse_k1"][test_mask], vector)
        k2_power = _power(selected["transverse_k2"][test_mask], vector)
        cone_power = _power(selected["cone_p1_k1"][test_mask], vector)
        longitudinal = _power(selected["longitudinal_k1"][test_mask], vector)
        pol0 = _power(selected["polarization_0_k1"][test_mask], vector)
        pol1 = _power(selected["polarization_1_k1"][test_mask], vector)
        ir_ratios.append(k1_power / max(k2_power, 1.0e-300))
        k1_powers.append(k1_power)
        cone_ratios.append(cone_power / max(k1_power, 1.0e-300))
        longitudinal_ratios.append(longitudinal / max(k1_power, 1.0e-300))
        polarization_splits.append(abs(pol0 - pol1) / max(0.5 * (pol0 + pol1), 1.0e-300))
        eigenvalue_counts.append(int(np.sum(np.abs(eigenvalues / expected - 1.0) <= 0.25)))
        vectors.append(vector)
    return CrossValidatedMode(
        family=family,
        size=size,
        expected_ir_ratio=expected,
        heldout_ir_ratios=(float(ir_ratios[0]), float(ir_ratios[1])),
        heldout_k1_powers=(float(k1_powers[0]), float(k1_powers[1])),
        heldout_cone_ratios=(float(cone_ratios[0]), float(cone_ratios[1])),
        heldout_longitudinal_ratios=(
            float(longitudinal_ratios[0]),
            float(longitudinal_ratios[1]),
        ),
        heldout_polarization_splits=(
            float(polarization_splits[0]),
            float(polarization_splits[1]),
        ),
        train_eigenvalue_counts=(int(eigenvalue_counts[0]), int(eigenvalue_counts[1])),
        vectors=(vectors[0], vectors[1]),
    )


def gauss_charge_regression(
    sample_modes: dict[str, np.ndarray],
    family: str,
    size: int,
    vector: np.ndarray,
    sample_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """Test one coupling across k1/k2 against the fixed raw U(1) charge mode."""

    family_slice = operator_family_slices()[family]
    if sample_mask is None:
        sample_mask = np.ones(len(sample_modes["temporal_k1"]), dtype=bool)
    mask = np.asarray(sample_mask, dtype=bool)
    temporal_1 = np.asarray(sample_modes["temporal_k1"])[mask][..., family_slice] @ vector
    temporal_2 = np.asarray(sample_modes["temporal_k2"])[mask][..., family_slice] @ vector
    raw_charge_1 = np.asarray(sample_modes["temporal_k1"])[mask][..., 0]
    raw_charge_2 = np.asarray(sample_modes["temporal_k2"])[mask][..., 0]
    k = 2.0 * math.pi / size
    stiffness = (
        4.0 * math.sin(0.5 * k) ** 2,
        4.0 * math.sin(k) ** 2,
    )
    gauss_1 = -stiffness[0] * temporal_1
    gauss_2 = -stiffness[1] * temporal_2
    gauss = np.concatenate((gauss_1.ravel(), gauss_2.ravel()))
    charge = np.concatenate((raw_charge_1.ravel(), raw_charge_2.ravel()))
    gauss -= np.mean(gauss)
    charge -= np.mean(charge)
    covariance = np.mean(np.conj(charge) * gauss)
    variance_gauss = float(np.mean(np.abs(gauss) ** 2))
    variance_charge = float(np.mean(np.abs(charge) ** 2))
    r_squared = float(abs(covariance) ** 2 / max(variance_gauss * variance_charge, 1.0e-300))
    coupling_1 = abs(np.mean(np.conj(raw_charge_1) * gauss_1)) / max(
        float(np.mean(np.abs(raw_charge_1) ** 2)),
        1.0e-300,
    )
    coupling_2 = abs(np.mean(np.conj(raw_charge_2) * gauss_2)) / max(
        float(np.mean(np.abs(raw_charge_2) ** 2)),
        1.0e-300,
    )
    coupling_cv = abs(coupling_1 - coupling_2) / max(0.5 * (coupling_1 + coupling_2), 1.0e-300)
    return {
        "r_squared": r_squared,
        "coupling_k1": float(coupling_1),
        "coupling_k2": float(coupling_2),
        "coupling_cv": float(coupling_cv),
    }


__all__ = [
    "CrossValidatedMode",
    "cross_validated_mode",
    "expected_massless_ir_ratio",
    "extract_low_momentum_modes",
    "gauss_charge_regression",
    "native_gradient_operators",
    "operator_completeness_audit",
    "operator_family_slices",
    "u3_hermitian_generators",
]
