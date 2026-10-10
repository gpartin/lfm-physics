"""Call contracts must match even on hosts without a CUDA runtime."""

import inspect

import pytest

from lfm.core.backends.cupy_backend import CupyBackend
from lfm.core.backends.numpy_backend import NumpyBackend
from lfm.core.backends.protocol import Backend


@pytest.mark.parametrize("backend", [NumpyBackend, CupyBackend])
@pytest.mark.parametrize(
    "method",
    [
        "step_real",
        "step_complex",
        "step_color",
        "step_real_gravity_recovery",
        "step_complex_gravity_recovery",
        "step_color_gravity_recovery",
    ],
)
def test_step_call_contract(backend, method):
    # Runtime Protocol checks only method presence, not argument compatibility.
    def contract(cls):
        return [
            (p.name, p.kind, p.default)
            for p in inspect.signature(getattr(cls, method)).parameters.values()
        ]

    assert contract(backend) == contract(Backend)
