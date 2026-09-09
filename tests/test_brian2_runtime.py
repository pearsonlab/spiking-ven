"""Tests for the optional Brian2 runtime glue.

Skipped entirely when the `brian2` extra is not installed, so the core test run on a
numpy+scipy-only environment stays green.
"""

import numpy as np
import pytest

brian2 = pytest.importorskip("brian2", reason="requires the 'brian2' extra")

from spiking_ven.brian2_runtime import (  # noqa: E402  (after importorskip by design)
    build_ven_groups,
    build_ven_synapses,
)


def test_build_ven_groups_shapes():
    brian2.start_scope()
    groups = build_ven_groups(N_e=20, N_i=5)
    assert groups["e_group"].N == 20
    assert groups["i_group"].N == 5
    # tau_s is echoed back so callers can match the numpy model's trace constant.
    assert groups["tau_s"] is not None


def test_ven_synapses_wire_from_numpy_weights():
    """The Brian2 side must accept weight matrices straight from the numpy model."""
    brian2.start_scope()
    n_e, n_i, n_aud, n_hvc = 12, 4, 6, 5
    groups = build_ven_groups(N_e=n_e, N_i=n_i)

    aud = brian2.SpikeGeneratorGroup(n_aud, [0], [1] * brian2.ms)
    hvc = brian2.SpikeGeneratorGroup(n_hvc, [0], [1] * brian2.ms)

    rng = np.random.default_rng(0)
    B = rng.random((n_e, n_aud)) * 0.1          # aud -> E
    B_hvc = rng.random((n_i, n_hvc)) * 0.1      # HVC -> I

    syns = build_ven_synapses(groups, aud, hvc, B, B_hvc)
    assert len(syns) == 4, "expected B, B_hvc, JEI and JIE synapse objects"
