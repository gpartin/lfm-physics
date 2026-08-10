"""Focused tests for configurable persistent simulation precision."""

from __future__ import annotations

import re

import numpy as np
import pytest

import lfm
from lfm.core.backends import get_backend, gpu_available
from lfm.core.backends.kernel_source import (
    EVOLUTION_COMPLEX_KERNEL_SRC,
    EVOLUTION_KERNEL_SRC,
    EVOLUTION_REAL_KERNEL_SRC,
    PHASE1_KERNEL_SRC,
    SA_DIFFUSION_KERNEL_SRC,
    kernel_source_for_precision,
)
from lfm.fields.equilibrium import equilibrate_from_fields, equilibrate_from_fields_19pt

CUDA_SOURCES = (
    EVOLUTION_REAL_KERNEL_SRC,
    EVOLUTION_COMPLEX_KERNEL_SRC,
    EVOLUTION_KERNEL_SRC,
    PHASE1_KERNEL_SRC,
    SA_DIFFUSION_KERNEL_SRC,
)


def _config(precision: lfm.Precision, field_level: lfm.FieldLevel) -> lfm.SimulationConfig:
    return lfm.SimulationConfig(
        grid_size=8,
        precision=precision,
        field_level=field_level,
        boundary_type=lfm.BoundaryType.PERIODIC,
        lambda_self=lfm.LAMBDA_H,
        epsilon_w=0.1,
        e0_sq=1.0,
        report_interval=10**9,
    )


def _seed_phase_space(sim: lfm.Simulation, dtype: np.dtype) -> None:
    rng = np.random.default_rng(20260715)
    n = sim.config.grid_size
    prefix = (3,) if sim.config.field_level == lfm.FieldLevel.COLOR else ()
    shape = prefix + (n, n, n)
    real = rng.normal(0.0, 0.01, shape).astype(dtype)
    real_prev = (real + rng.normal(0.0, 1e-4, shape)).astype(dtype)
    sim.set_psi_real(real)
    sim.set_psi_real_prev(real_prev)
    if sim.config.field_level != lfm.FieldLevel.REAL:
        imag = rng.normal(0.0, 0.01, shape).astype(dtype)
        imag_prev = (imag + rng.normal(0.0, 1e-4, shape)).astype(dtype)
        sim.set_psi_imag(imag)
        sim.set_psi_imag_prev(imag_prev)
    chi = (19.0 + rng.normal(0.0, 1e-3, (n, n, n))).astype(dtype)
    chi_prev = (chi + rng.normal(0.0, 1e-5, (n, n, n))).astype(dtype)
    sim.set_chi(chi)
    sim.set_chi_prev(chi_prev)


class TestPrecisionConfig:
    def test_default_is_float32(self):
        assert lfm.SimulationConfig().precision == lfm.Precision.FLOAT32

    def test_string_is_normalized(self):
        cfg = lfm.SimulationConfig(precision="float64")
        assert cfg.precision == lfm.Precision.FLOAT64

    def test_invalid_precision_rejected(self):
        with pytest.raises(ValueError, match="precision"):
            lfm.SimulationConfig(precision="float16")


class TestKernelPrecisionSource:
    @pytest.mark.parametrize("source", CUDA_SOURCES)
    def test_float32_source_is_identical_object(self, source):
        assert kernel_source_for_precision(source, "float32") is source

    @pytest.mark.parametrize("source", CUDA_SOURCES)
    def test_float64_source_promotes_types_and_literals(self, source):
        promoted = kernel_source_for_precision(source, "float64")
        assert not re.search(r"\bfloat\b", promoted)
        assert not re.search(r"(?<=[0-9.])f\b", promoted)
        assert "double" in promoted


class TestCpuPrecision:
    @pytest.mark.parametrize(
        "field_level",
        (lfm.FieldLevel.REAL, lfm.FieldLevel.COMPLEX, lfm.FieldLevel.COLOR),
    )
    @pytest.mark.parametrize(
        ("precision", "dtype"),
        (
            (lfm.Precision.FLOAT32, np.dtype(np.float32)),
            (lfm.Precision.FLOAT64, np.dtype(np.float64)),
        ),
    )
    def test_complete_phase_space_uses_configured_dtype(self, field_level, precision, dtype):
        sim = lfm.Simulation(_config(precision, field_level), backend="cpu")
        _seed_phase_space(sim, np.dtype(np.float64))
        sim.run(2, record_metrics=False)
        snapshot = sim.phase_space_snapshot()
        for key in ("psi_real", "psi_real_prev", "chi", "chi_prev"):
            assert snapshot[key].dtype == dtype
        if field_level == lfm.FieldLevel.REAL:
            assert snapshot["psi_imag"] is None
            assert snapshot["psi_imag_prev"] is None
        else:
            assert snapshot["psi_imag"].dtype == dtype
            assert snapshot["psi_imag_prev"].dtype == dtype
        assert sim.get_boundary_mask().dtype == dtype

    def test_float64_setter_retains_sub_float32_increment(self):
        sim = lfm.Simulation(_config(lfm.Precision.FLOAT64, lfm.FieldLevel.REAL), backend="cpu")
        value = 1.0 + 2.0**-40
        field = np.full((8, 8, 8), value, dtype=np.float64)
        sim.set_psi_real(field)
        assert sim.get_psi_real()[0, 0, 0] == value
        assert sim.get_psi_real()[0, 0, 0] != np.float64(np.float32(value))

    def test_backend_conversion_uses_requested_dtype(self):
        source = np.array([1.0, 2.0], dtype=np.float32)
        backend = get_backend("cpu", lfm.Precision.FLOAT64)
        converted = backend.from_numpy(source)
        assert backend.dtype == np.dtype(np.float64)
        assert converted.dtype == np.dtype(np.float64)

    def test_float64_color_sa_path_preserves_dtype(self):
        cfg = _config(lfm.Precision.FLOAT64, lfm.FieldLevel.COLOR)
        cfg.kappa_tube = 0.1
        sim = lfm.Simulation(cfg, backend="cpu")
        _seed_phase_space(sim, np.dtype(np.float64))
        sim.run(1, record_metrics=False)
        assert sim.sa_fields is not None
        assert sim.sa_fields.dtype == np.dtype(np.float64)
        assert np.isfinite(sim.sa_fields).all()

    @pytest.mark.parametrize(
        "equilibrate",
        (equilibrate_from_fields, equilibrate_from_fields_19pt),
    )
    @pytest.mark.parametrize("color", (False, True))
    def test_equilibrium_helpers_preserve_float64(self, equilibrate, color):
        rng = np.random.default_rng(41)
        shape = (3, 8, 8, 8) if color else (8, 8, 8)
        real = rng.normal(0.0, 0.2, shape).astype(np.float64)
        imag = rng.normal(0.0, 0.2, shape).astype(np.float64)

        chi = equilibrate(real, imag)

        assert chi.dtype == np.dtype(np.float64)
        assert np.any(chi != chi.astype(np.float32).astype(np.float64))

    def test_simulation_equilibrate_preserves_float64(self):
        sim = lfm.Simulation(
            _config(lfm.Precision.FLOAT64, lfm.FieldLevel.COMPLEX),
            backend="cpu",
        )
        _seed_phase_space(sim, np.dtype(np.float64))
        with pytest.warns(UserWarning, match="before any solitons"):
            sim.equilibrate()
        assert sim.get_chi().dtype == np.dtype(np.float64)

    def test_remote_float32_direct_job_backend_is_available(self):
        backend = get_backend("remote", lfm.Precision.FLOAT32)
        assert backend.__class__.__name__ == "RemoteBackend"

    def test_remote_float64_is_rejected_before_submission(self):
        with pytest.raises(NotImplementedError, match="float32"):
            get_backend("remote", lfm.Precision.FLOAT64)

    def test_remote_simulation_contract_is_rejected_intentionally(self):
        with pytest.raises(NotImplementedError, match="direct float32 remote jobs"):
            lfm.Simulation(
                _config(lfm.Precision.FLOAT32, lfm.FieldLevel.REAL),
                backend="remote",
            )

    def test_float64_checkpoint_round_trip(self, tmp_path):
        cfg = _config(lfm.Precision.FLOAT64, lfm.FieldLevel.COMPLEX)
        sim = lfm.Simulation(cfg, backend="cpu")
        _seed_phase_space(sim, np.dtype(np.float64))
        sim.run(3, record_metrics=False)
        expected = sim.phase_space_snapshot()
        path = tmp_path / "float64_checkpoint.npz"
        sim.save_checkpoint(path)

        restored = lfm.Simulation.load_checkpoint(path, backend="cpu")
        assert restored.config.precision == lfm.Precision.FLOAT64
        assert restored.get_psi_real().dtype == np.dtype(np.float64)
        assert restored.get_chi().dtype == np.dtype(np.float64)
        np.testing.assert_array_equal(restored.get_psi_real(), expected["psi_real"])
        np.testing.assert_array_equal(restored.get_psi_imag(), expected["psi_imag"])
        np.testing.assert_array_equal(restored.get_chi(), expected["chi"])
        restored_phase = restored.phase_space_snapshot()
        for key in (
            "psi_real_prev",
            "psi_imag_prev",
            "chi_prev",
        ):
            np.testing.assert_array_equal(restored_phase[key], expected[key])

        sim.run(1, record_metrics=False)
        restored.run(1, record_metrics=False)
        continued = sim.phase_space_snapshot()
        restored_continued = restored.phase_space_snapshot()
        for key in (
            "psi_real",
            "psi_real_prev",
            "psi_imag",
            "psi_imag_prev",
            "chi",
            "chi_prev",
        ):
            np.testing.assert_array_equal(restored_continued[key], continued[key])


@pytest.mark.gpu
@pytest.mark.skipif(not gpu_available(), reason="CuPy GPU backend unavailable")
class TestGpuPrecision:
    def test_all_float64_kernels_compile(self):
        backend = get_backend("gpu", lfm.Precision.FLOAT64)
        for name in (
            "_kernel_real",
            "_kernel_complex",
            "_kernel_color",
            "_kernel_phase1",
            "_kernel_sa_diffusion",
        ):
            getattr(backend, name).compile()

    @pytest.mark.parametrize(
        "field_level",
        (lfm.FieldLevel.REAL, lfm.FieldLevel.COMPLEX, lfm.FieldLevel.COLOR),
    )
    def test_float64_gpu_state_and_evolution(self, field_level):
        sim = lfm.Simulation(
            _config(lfm.Precision.FLOAT64, field_level), backend="gpu"
        )
        _seed_phase_space(sim, np.dtype(np.float64))
        sim.run(2, record_metrics=False)
        snapshot = sim.phase_space_snapshot()
        for value in snapshot.values():
            if isinstance(value, np.ndarray):
                assert value.dtype == np.dtype(np.float64)
                assert np.isfinite(value).all()

    def test_float64_color_gpu_matches_cpu(self):
        cpu = lfm.Simulation(
            _config(lfm.Precision.FLOAT64, lfm.FieldLevel.COLOR), backend="cpu"
        )
        gpu = lfm.Simulation(
            _config(lfm.Precision.FLOAT64, lfm.FieldLevel.COLOR), backend="gpu"
        )
        _seed_phase_space(cpu, np.dtype(np.float64))
        _seed_phase_space(gpu, np.dtype(np.float64))
        cpu.run(3, record_metrics=False)
        gpu.run(3, record_metrics=False)
        cpu_state = cpu.phase_space_snapshot()
        gpu_state = gpu.phase_space_snapshot()
        for key in (
            "psi_real",
            "psi_real_prev",
            "psi_imag",
            "psi_imag_prev",
            "chi",
            "chi_prev",
        ):
            np.testing.assert_allclose(
                gpu_state[key], cpu_state[key], rtol=5e-12, atol=5e-12
            )

    def test_float64_color_sa_gpu_path_preserves_dtype(self):
        cfg = _config(lfm.Precision.FLOAT64, lfm.FieldLevel.COLOR)
        cfg.kappa_tube = 0.1
        sim = lfm.Simulation(cfg, backend="gpu")
        _seed_phase_space(sim, np.dtype(np.float64))
        sim.run(1, record_metrics=False)
        assert sim.sa_fields is not None
        assert sim.sa_fields.dtype == np.dtype(np.float64)
        assert np.isfinite(sim.sa_fields).all()
