"""
LFM Remote Backend
===================

Implements the ``Backend`` protocol by dispatching simulation calls to the
``POST /v1/simulate_job`` endpoint on the WaveGuard API instead of running
locally.

Configuration is read from environment variables (or ``configure_remote()``):

    LFM_SIMULATE_API_KEY  — the internal API key (sk-lfmsim-...)
    LFM_REMOTE_ENDPOINT   — base URL of the WaveGuard API
                            (default: https://gpartin--waveguard-api-fastapi-app.modal.run)

Usage::

    import lfm
    lfm.configure_remote()                          # reads env vars
    # or
    lfm.configure_remote(
        api_key="sk-lfmsim-...",
        endpoint="https://...",
    )

    backend = lfm.get_backend("remote")
    # Then use lfm.Simulation normally — it will call the remote API
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .job_schema import HookSpec, JobResult, RunPlanStep, SimulationJob, SnapshotSpec

# ─── Global remote config ─────────────────────────────────────────────────────

_REMOTE_API_KEY: str | None = None
_REMOTE_ENDPOINT: str = "https://gpartin--waveguard-api-fastapi-app.modal.run"


def configure_remote(
    api_key: str | None = None,
    endpoint: str | None = None,
) -> None:
    """Configure the remote backend credentials.

    If ``api_key`` / ``endpoint`` are not provided, falls back to the
    ``LFM_SIMULATE_API_KEY`` / ``LFM_REMOTE_ENDPOINT`` environment variables.

    Raises ``ValueError`` if no API key is available from any source.
    """
    global _REMOTE_API_KEY, _REMOTE_ENDPOINT

    resolved_key = api_key or os.environ.get("LFM_SIMULATE_API_KEY")
    if not resolved_key:
        raise ValueError(
            "No API key for remote backend. "
            "Set LFM_SIMULATE_API_KEY or call configure_remote(api_key=...)."
        )
    _REMOTE_API_KEY = resolved_key

    resolved_ep = endpoint or os.environ.get("LFM_REMOTE_ENDPOINT")
    if resolved_ep:
        _REMOTE_ENDPOINT = resolved_ep.rstrip("/")


# ─── RemoteBackend ────────────────────────────────────────────────────────────


class RemoteBackend:
    """Backend implementation that executes GOV-01/02 on the WaveGuard cloud GPU.

    Implements the minimal interface expected by ``lfm.Simulation``
    (``allocate``, ``step_real``, ``run_steps``).

    Most use-cases go through the higher-level ``run_job()`` method directly.
    """

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        # Per-instance key/endpoint override (falls back to globals)
        self._api_key = api_key
        self._endpoint = endpoint

    # ── Credential helpers ─────────────────────────────────────────────────

    def _resolved_key(self) -> str:
        key = self._api_key or _REMOTE_API_KEY or os.environ.get("LFM_SIMULATE_API_KEY")
        if not key:
            raise RuntimeError(
                "Remote backend: no API key. "
                "Call lfm.configure_remote() or set LFM_SIMULATE_API_KEY."
            )
        return key

    def _resolved_endpoint(self) -> str:
        ep = self._endpoint or _REMOTE_ENDPOINT
        return ep.rstrip("/")

    # ── Core RPC ───────────────────────────────────────────────────────────

    def run_job(self, job: SimulationJob, timeout: float = 3600.0) -> JobResult:
        """Submit a ``SimulationJob`` and return the ``JobResult``.

        This is the primary API for the remote backend.

        Parameters
        ----------
        job :
            A fully specified simulation job.
        timeout :
            HTTP timeout in seconds.  Defaults to 30 minutes.

        Returns
        -------
        JobResult
            Decoded result including snapshots and energy metrics.

        Raises
        ------
        RuntimeError
            On HTTP error or network failure.
        """

        try:
            import requests  # type: ignore[import]
        except ImportError:
            return self._run_job_urllib(job, timeout)

        url = f"{self._resolved_endpoint()}/v1/simulate_job"
        headers = {
            "X-API-Key": self._resolved_key(),
            "Content-Type": "application/json",
        }
        payload = job.to_request_dict()

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Remote simulation request failed: {exc}") from exc

        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text[:200])
            except Exception:
                detail = resp.text[:200]
            raise RuntimeError(f"Remote simulation returned HTTP {resp.status_code}: {detail}")

        return JobResult.from_response_dict(resp.json(), job.grid_size)

    def _run_job_urllib(self, job: SimulationJob, timeout: float) -> JobResult:
        """Fallback using stdlib urllib when ``requests`` is not installed."""
        import json
        import urllib.error
        import urllib.request

        url = f"{self._resolved_endpoint()}/v1/simulate_job"
        payload_bytes = json.dumps(job.to_request_dict()).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload_bytes,
            headers={
                "X-API-Key": self._resolved_key(),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:200]
            raise RuntimeError(f"Remote simulation returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Remote simulation request failed: {exc.reason}") from exc

        return JobResult.from_response_dict(data, job.grid_size)

    # ── Convenience wrappers ───────────────────────────────────────────────

    def run_steps(
        self,
        psi: np.ndarray,
        chi: np.ndarray,
        n_steps: int,
        *,
        dt: float = 0.02,
        kappa: float = 1.0 / 63.0,
        chi0: float = 19.0,
        snapshot_every: int = 0,
        hooks: list[HookSpec] | None = None,
    ) -> JobResult:
        """Run ``n_steps`` leapfrog steps starting from ``psi``/``chi`` state.

        Parameters
        ----------
        psi, chi :
            Current field state arrays, shape (N,N,N).
        n_steps :
            Number of timesteps to execute.
        snapshot_every :
            Record fields every this many steps (0 = no snapshots).
        hooks :
            Optional list of ``HookSpec`` (continuous sources, detectors).

        Returns
        -------
        JobResult
        """
        assert psi.shape == chi.shape and psi.ndim == 3, "psi and chi must be (N,N,N)"
        N = psi.shape[0]

        plan_step = RunPlanStep(
            steps=n_steps,
            hooks=hooks or [],
            snapshots=SnapshotSpec(every_n_steps=snapshot_every),
        )
        job = SimulationJob(
            grid_size=N,
            run_plan=[plan_step],
            chi0=chi0,
            kappa=kappa,
            dt=dt,
            initial_psi=psi.astype(np.float32),
            initial_chi=chi.astype(np.float32),
        )
        return self.run_job(job)

    def health_check(self) -> dict[str, Any]:
        """Call GET /v1/health and return the JSON response dict."""
        try:
            import requests

            resp = requests.get(
                f"{self._resolved_endpoint()}/v1/health",
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            return {"status": "unreachable", "error": str(exc)}
