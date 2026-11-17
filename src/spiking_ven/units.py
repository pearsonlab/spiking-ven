"""Physiological units for the network: membrane potentials in mV, currents in pA.

The membrane equation, stepped exactly at dt = 1 ms, is

    C dV/dt = -g_L (V - E_L) + I        ->   V <- E_L + d (V - E_L) + (1 - d) I / g_L

with ``d = exp(-dt / tau_m)`` and ``g_L = C / tau_m`` (nS = pF / ms, so pA / nS = mV).
``I`` sums the tonic drive, the synaptic currents, adaptation and noise, all in pA.

Synaptic weights are stated as the **peak PSP in mV** one presynaptic spike produces at
rest. Every synaptic current is a trace that jumps by the weight's current (pA) at a spike
and decays with the synapse's time constant, so the current per spike -- its peak EPSC
or IPSC -- is ``psp_mv * g_L / psp_peak(tau_syn, tau_m)``.

The model was once dimensionless: reset at 0, a threshold ``theta`` in input units, and
weights in those units. :func:`from_dimensionless` converts such a parameter set, so
older model files and :class:`~spiking_ven.vocal_error_net.VocalErrorNetV2` keep their
dynamics exactly (up to floating point).
"""

from __future__ import annotations

import numpy as np

__all__ = ["psp_peak", "leak_conductance", "current_per_psp", "tonic_drive_pa",
           "from_dimensionless"]


def psp_peak(tau_syn_ms: float, tau_m_ms: float, dt: float = 1.0) -> float:
    """Peak membrane deflection (mV) from one synaptic trace of input-referred size 1 mV.

    The trace jumps to 1 on the step it arrives and decays with ``tau_syn_ms`` (0: that
    step only); the membrane integrates it with ``tau_m_ms``, as the simulation does. The
    PSP of a weight ``w`` mV of input-referred drive peaks at ``w * psp_peak``.
    """
    dm = float(np.exp(-dt / tau_m_ms))
    ds = float(np.exp(-dt / tau_syn_ms)) if tau_syn_ms > 0.0 else 0.0
    h, v, best = 0.0, 0.0, 0.0
    for t in range(int(20 * max(tau_m_ms, tau_syn_ms, 1.0) / dt)):
        h = ds * h + (1.0 if t == 0 else 0.0)
        v = dm * v + (1.0 - dm) * h
        best = max(best, v)
    return best


def leak_conductance(c_pf: float, tau_m_ms: float) -> float:
    """``g_L`` (nS) of a membrane with capacitance ``c_pf`` (pF) and time constant (ms)."""
    return float(c_pf) / float(tau_m_ms)


def current_per_psp(tau_syn_ms: float, tau_m_ms: float, g_l_ns: float) -> float:
    """pA of synaptic current per spike (the trace's peak) per mV of peak PSP."""
    return g_l_ns / psp_peak(tau_syn_ms, tau_m_ms)


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


def from_dimensionless(p: dict, c_e_pf: float, c_i_pf: float) -> dict:
    """Convert a dimensionless parameter set to mV / pA.

    ``p`` holds the old names: ``theta_e``, ``theta_i`` (thresholds, reset at 0),
    ``tau_e``, ``tau_i``, ``tau_s``, ``tau_inh``, ``tau_exc``, ``drive_e``, ``drive_i``,
    ``noise_e``, ``noise_i``, ``a_sfa``, ``A_jie``, ``J_max_ie``, ``g_nmda``, ``e_l_mv``,
    ``v_th_mv``, ``e_nmda_mv``. The dimensionless membrane maps reset to ``e_l_mv`` and
    each threshold to ``v_th_mv``, so one input unit is ``k = (v_th - e_l) / theta`` mV.

    Returns the new scalars and ``w_factor``: the multipliers that turn the old
    ``B``, ``B_hvc``, ``JEI`` and ``JIE`` matrices into peak PSPs in mV.
    """
    span = p["v_th_mv"] - p["e_l_mv"]
    k_e, k_i = span / p["theta_e"], span / p["theta_i"]
    g_e = leak_conductance(c_e_pf, p["tau_e"])
    g_i = leak_conductance(c_i_pf, p["tau_i"])
    psp_ie = k_e * psp_peak(p.get("tau_inh", 0.0), p["tau_e"])
    return {
        "e_l_mv": p["e_l_mv"], "v_th_mv": p["v_th_mv"], "v_reset_mv": p["e_l_mv"],
        "c_e_pf": c_e_pf, "c_i_pf": c_i_pf,
        "i_drive_e_pa": p["drive_e"] * k_e * g_e,
        "i_drive_i_pa": p["drive_i"] * k_i * g_i,
        "noise_e_pa": p["noise_e"] * k_e * g_e,
        "noise_i_pa": p["noise_i"] * k_i * g_i,
        "a_sfa_pa": p["a_sfa"] * k_e * g_e,
        "a_jie_mv": p["A_jie"] * psp_ie,
        "j_max_ie_mv": p["J_max_ie"] * psp_ie,
        # NMDA current = g (B @ s) mg(V) (E_nmda - V) / (E_nmda - E_L), B in pA.
        "g_nmda": p["g_nmda"] * (p["e_nmda_mv"] - p["e_l_mv"]) / k_e,
        "w_factor": {
            "B": k_e * psp_peak(p["tau_s"], p["tau_e"]),
            "B_hvc": k_i * psp_peak(p["tau_s"], p["tau_i"]),
            "JEI": k_i * psp_peak(p.get("tau_exc", 0.0), p["tau_i"]),
            "JIE": psp_ie,
        },
    }
