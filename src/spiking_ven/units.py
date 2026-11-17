"""Physiological units for the network: potentials in mV, currents in pA, conductances in nS.

The membrane equation, with every synapse a conductance acting through its reversal
potential, is

    C dV/dt = -g_L (V - E_L) - sum_k g_k(t) (V - E_k) + I

with ``g_L = C / tau_m`` (nS = pF / ms, so pA / nS = mV) and ``I`` the tonic drive,
adaptation and noise currents (pA). It is stepped at dt = 1 ms by exponential Euler
(``vocal_error_net._step``): with the conductances held over the step, V relaxes exactly
toward ``V_inf = (g_L E_L + sum g_k E_k + I) / g_tot`` with time constant ``C / g_tot``.

Each synaptic conductance is a trace that jumps by the weight's peak conductance (nS) at a
presynaptic spike and decays with the synapse's time constant.
"""

from __future__ import annotations

import numpy as np

__all__ = ["leak_conductance", "tonic_drive_pa"]


def leak_conductance(c_pf: float, tau_m_ms: float) -> float:
    """``g_L`` (nS) of a membrane with capacitance ``c_pf`` (pF) and time constant (ms)."""
    return float(c_pf) / float(tau_m_ms)


def tonic_drive_pa(rate_hz: float, tau_m_ms: float, g_l_ns: float,
                   e_l_mv: float, v_th_mv: float, v_reset_mv: float) -> float:
    """Tonic current (pA) that alone fires a cell at ``rate_hz`` from reset to threshold.

    Relative to rest the membrane relaxes from ``v_reset`` toward ``I / g_L``; after
    ``n = 1000 / rate_hz`` steps it reaches threshold, so
    ``I / g_L = (v_th - v_reset d^n) / (1 - d^n)`` (potentials relative to ``e_l``).
    """
    d_n = float(np.exp(-1.0 / tau_m_ms)) ** (1000.0 / rate_hz)
    v_th, v_r = v_th_mv - e_l_mv, v_reset_mv - e_l_mv
    return g_l_ns * (v_th - v_r * d_n) / (1.0 - d_n)
