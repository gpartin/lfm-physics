"""
Tests for the remote backend (POST /v1/simulate_job).

All tests that make real HTTP calls are skipped when LFM_SIMULATE_API_KEY
is not set in the environment, so this suite is safe to run in CI without
credentials.

To run against the live service:
    $env:LFM_SIMULATE_API_KEY = "sk-lfmsim-..."
    $env:LFM_REMOTE_ENDPOINT  = "https://gpartin--waveguard-api-fastapi-app.modal.run"
    python -m pytest tests/test_remote_backend.py -v

Or locally (uses fallback _local_simulate, no network):
    python -m pytest tests/test_remote_backend.py -v -k "not live"
"""

from __future__ import annotations

import os
import json
import time
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lfm.core.backends.job_schema import (
    HookSpec,
    JobResult,
    RunPlanStep,
    SimulationJob,
    Snapshot,
    SnapshotSpec,
)
from lfm.core.backends.remote_backend import RemoteBackend, configure_remote


# ─── Fixtures ─────────────────────────────────────────────────────────────────

_LIVE = bool(os.environ.get("LFM_SIMULATE_API_KEY"))
_SKIP_LIVE = pytest.mark.skipif(not _LIVE, reason="LFM_SIMULATE_API_KEY not set")


@pytest.fixture
def tiny_job() -> SimulationJob:
    """A minimal 8³ grid job for fast unit tests."""
    return SimulationJob(
        grid_size=8,
        chi0=19.0,
        kappa=1 / 63,
        dt=0.02,
        run_plan=[
            RunPlanStep(
                steps=100,
                hooks=[
                    HookSpec(
                        type="continuous_source",
                        axis=2,
                        position=2,
                        omega=1.0,
                        amplitude=2.0,
                    )
                ],
                snapshots=SnapshotSpec(every_n_steps=50),
            )
        ],
    )


@pytest.fixture
def mock_backend() -> RemoteBackend:
    """RemoteBackend with a mocked HTTP layer."""
    return RemoteBackend(
        api_key="sk-lfmsim-test",
        endpoint="http://localhost:8080",
    )


# ─── Unit tests (no network) ──────────────────────────────────────────────────


class TestSimulationJob:
    def test_to_request_dict_basic(self, tiny_job):
        d = tiny_job.to_request_dict()
        assert d["grid_size"] == 8
        assert d["chi0"] == 19.0
        assert len(d["run_plan"]) == 1
        assert d["run_plan"][0]["steps"] == 100
        assert d["run_plan"][0]["hooks"][0]["type"] == "continuous_source"
        assert d["run_plan"][0]["snapshots"]["every_n_steps"] == 50

    def test_to_request_dict_with_initial_fields(self, tiny_job):
        N = 8
        tiny_job.initial_psi = np.zeros((N, N, N), dtype=np.float32)
        tiny_job.initial_psi[4, 4, 4] = 5.0
        d = tiny_job.to_request_dict()
        assert "initial_psi" in d
        assert len(d["initial_psi"]) == N**3
        assert d["initial_psi"][4 * N * N + 4 * N + 4] == pytest.approx(5.0)

    def test_no_fusion_depth_by_default(self, tiny_job):
        d = tiny_job.to_request_dict()
        assert "fusion_depth" not in d

    def test_fusion_depth_included_when_set(self, tiny_job):
        tiny_job.fusion_depth = 20
        d = tiny_job.to_request_dict()
        assert d["fusion_depth"] == 20


class TestJobResult:
    def _make_raw(self, N: int, n_snaps: int = 2) -> Dict[str, Any]:
        import base64

        snaps = []
        for i in range(n_snaps):
            arr = np.ones((N, N, N), dtype=np.float32) * i
            snaps.append(
                {
                    "step": (i + 1) * 50,
                    "psi_b64": base64.b64encode(arr.tobytes()).decode(),
                    "chi_b64": base64.b64encode(arr.tobytes()).decode(),
                }
            )
        return {
            "job_id": "sim-abc123",
            "steps_completed": 100,
            "elapsed_ms": 250,
            "kernel_info": {
                "fusion_depth": 10,
                "active_fraction": 0.82,
                "pruning_efficiency": "18.0%",
                "backend": "cupy",
            },
            "snapshots": snaps,
            "detector_patterns": {},
            "metrics": {
                "energy_initial": 5.0,
                "energy_final": 4.97,
                "energy_drift_pct": 0.6,
            },
        }

    def test_from_response_dict(self):
        N = 8
        raw = self._make_raw(N, n_snaps=2)
        result = JobResult.from_response_dict(raw, N)
        assert result.job_id == "sim-abc123"
        assert result.steps_completed == 100
        assert result.fusion_depth == 10
        assert result.pruning_efficiency == "18.0%"
        assert abs(result.energy_drift_pct - 0.6) < 1e-6

    def test_snapshots_decoded(self):
        N = 8
        raw = self._make_raw(N, n_snaps=2)
        result = JobResult.from_response_dict(raw, N)
        assert len(result.snapshots) == 2
        s = result.snapshots[0]
        assert s.psi is not None
        assert s.psi.shape == (N, N, N)
        assert s.psi.dtype == np.float32

    def test_psi_final_chi_final(self):
        N = 8
        raw = self._make_raw(N, n_snaps=3)
        result = JobResult.from_response_dict(raw, N)
        # Last snapshot has values = 2.0
        assert result.psi_final is not None
        assert float(result.psi_final[0, 0, 0]) == pytest.approx(2.0)

    def test_no_snapshots(self):
        N = 8
        raw = self._make_raw(N, n_snaps=0)
        result = JobResult.from_response_dict(raw, N)
        assert result.psi_final is None
        assert result.chi_final is None


class TestConfigureRemote:
    def test_configure_sets_globals(self):
        """configure_remote() updates module-level globals."""
        import lfm.core.backends.remote_backend as rb

        configure_remote(
            api_key="sk-lfmsim-test-key",
            endpoint="http://test-endpoint",
        )
        assert rb._REMOTE_API_KEY == "sk-lfmsim-test-key"
        assert rb._REMOTE_ENDPOINT == "http://test-endpoint"

    def test_configure_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("LFM_SIMULATE_API_KEY", raising=False)
        import lfm.core.backends.remote_backend as rb

        rb._REMOTE_API_KEY = None
        with pytest.raises(ValueError, match="No API key"):
            configure_remote()


class TestRemoteBackendMocked:
    def _make_mock_response(self, status_code: int = 200, body: Any = None) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        if body is None:
            import base64

            N = 8
            arr = np.ones((N, N, N), dtype=np.float32)
            body = {
                "job_id": "sim-mocked",
                "steps_completed": 100,
                "elapsed_ms": 150,
                "kernel_info": {
                    "fusion_depth": 10,
                    "active_fraction": 1.0,
                    "pruning_efficiency": "0%",
                    "backend": "numpy",
                },
                "snapshots": [],
                "detector_patterns": {},
                "metrics": {"energy_initial": 1.0, "energy_final": 0.99, "energy_drift_pct": 1.0},
            }
        resp.json.return_value = body
        return resp

    def test_run_job_success(self, tiny_job, mock_backend):
        with patch("requests.post", return_value=self._make_mock_response()) as mock_post:
            result = mock_backend.run_job(tiny_job)
        assert result.job_id == "sim-mocked"
        assert result.steps_completed == 100
        assert mock_post.called

    def test_run_job_http_error(self, tiny_job, mock_backend):
        error_resp = self._make_mock_response(status_code=400, body={"detail": "bad request"})
        with patch("requests.post", return_value=error_resp):
            with pytest.raises(RuntimeError, match="HTTP 400"):
                mock_backend.run_job(tiny_job)

    def test_run_job_403_blocked(self, tiny_job, mock_backend):
        forbidden = self._make_mock_response(
            status_code=403, body={"detail": "Simulation jobs require an API key."}
        )
        with patch("requests.post", return_value=forbidden):
            with pytest.raises(RuntimeError, match="HTTP 403"):
                mock_backend.run_job(tiny_job)

    def test_run_steps_wrapper(self, mock_backend):
        N = 8
        psi = np.zeros((N, N, N), dtype=np.float32)
        chi = np.full((N, N, N), 19.0, dtype=np.float32)
        with patch("requests.post", return_value=self._make_mock_response()) as mock_post:
            result = mock_backend.run_steps(psi, chi, n_steps=100)
        assert result.job_id == "sim-mocked"


# ─── Live integration tests (skipped when no key) ─────────────────────────────


@_SKIP_LIVE
class TestRemoteBackendLive:
    """These tests call the real WaveGuard API.  Require LFM_SIMULATE_API_KEY."""

    @pytest.fixture(autouse=True)
    def setup(self):
        configure_remote()  # reads env vars

    def test_health_check(self):
        backend = RemoteBackend()
        result = backend.health_check()
        assert result.get("status") == "ok", f"Unexpected health: {result}"

    def test_tiny_job_roundtrip(self):
        N = 16
        job = SimulationJob(
            grid_size=N,
            run_plan=[
                RunPlanStep(
                    steps=200,
                    hooks=[
                        HookSpec(
                            type="continuous_source", axis=2, position=4, omega=1.0, amplitude=3.0
                        )
                    ],
                    snapshots=SnapshotSpec(every_n_steps=100),
                )
            ],
        )
        backend = RemoteBackend()
        t0 = time.time()
        result = backend.run_job(job)
        elapsed = time.time() - t0

        assert result.steps_completed == 200
        assert result.job_id.startswith("sim-")
        assert len(result.snapshots) == 2
        assert result.psi_final is not None
        assert result.psi_final.shape == (N, N, N)
        print(
            f"\nLive test passed in {elapsed:.1f}s. "
            f"Backend: {result.backend}, fusion_depth={result.fusion_depth}, "
            f"energy_drift={result.energy_drift_pct:.3f}%"
            if result.energy_drift_pct is not None
            else f"\nLive test passed in {elapsed:.1f}s. "
            f"Backend: {result.backend}, fusion_depth={result.fusion_depth}, "
            f"energy_drift=N/A"
        )

    def test_rejected_by_invalid_key(self):
        backend = RemoteBackend(api_key="sk-lfmsim-invalid-key")
        job = SimulationJob(
            grid_size=8,
            run_plan=[RunPlanStep(steps=10)],
        )
        with pytest.raises(RuntimeError):
            backend.run_job(job)
