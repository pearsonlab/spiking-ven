"""The validated operating point: the parameters the shipped model was trained with.

One source for values that were previously duplicated between CLI defaults and the
scratch scripts that drove sweeps. That duplication was not hypothetical -- a grid search
constructed the network from library defaults rather than these values and returned four
identical, broken points before anyone noticed, because the real parameters lived only in
an untracked script. Anything that drives or analyses the network should import from here
rather than restate a number.

Importable rather than a data file because some of these are DERIVED, and a formula has to
live with its inputs or it goes stale: ``a_jie`` is tied to the STDP window width (see
:func:`a_jie_for`), so freezing its computed value somewhere else would silently
desynchronise the learning rule the next time ``TAU_STDP`` changed.

Values that are physical or experimental rather than tuned -- sample rate, the DAF window,
conduction delays, the auditory rate -- live in :mod:`spiking_ven.constants` instead. The
split is whether the number describes the preparation or the fit.
"""

from __future__ import annotations

__all__ = [
    "N_E", "N_I", "N_HVC", "N_RENDITIONS",
    "B_SCALE", "C_B", "B_HVC_SCALE", "C_HVC",
    "DRIVE_E", "DRIVE_I", "R_E_TARGET",
    "TAU_S", "TAU_E", "TAU_STDP",
    "C_JIE", "R_I_TH", "J_MAX_IE",
    "ALPHA_THETA", "A_SFA", "TAU_SFA",
    "T_POST", "T_BURN", "N_ISTA",
    "a_jie_for",
    "CAUSAL_EPOCHS", "CAUSAL_LAMBDA", "CAUSAL_TARGET_USAGE", "CAUSAL_LR_BIAS",
    "CAUSAL_LR", "CAUSAL_L1", "CAUSAL_SMOOTH", "CAUSAL_N_ISTA", "CAUSAL_MP_STRENGTH",
]

# --- network ---------------------------------------------------------------------
N_E = 600
N_I = 150
N_HVC = 60
N_RENDITIONS = 600

#: Auditory -> E. ``C_B`` is sparse enough that each E cell samples only a few encoder
#: channels, so ``B_SCALE`` is the mean weight per surviving connection rather than a
#: population gain.
B_SCALE = 16.0
C_B = 0.05

#: HVC -> I, the motor prediction limb. The only thing that moves the interneuron rate:
#: E -> I alone supplies 0.28 against a threshold gap of 0.37, so I is effectively an
#: HVC-driven population. Compensating one limb without the other starves it.
B_HVC_SCALE = 0.333
C_HVC = 0.1

#: Tonic drives. ``theta_e`` is derived from ``DRIVE_E`` inside the network, so raising
#: the drive raises the threshold with it rather than the firing rate.
DRIVE_E = 0.06
DRIVE_I = 0.058
R_E_TARGET = 16.0

TAU_S = 20.0
TAU_E = 30.0

#: STDP coincidence window, pinned independently of the synaptic ``TAU_S``.
TAU_STDP = 20.0

#: I -> E, the only plastic weights in the model. Zero-initialised, so only song-driven
#: E cells ever acquire inhibition.
C_JIE = 0.1
R_I_TH = 6.0
J_MAX_IE = 1.0

#: Threshold homeostasis, off at the shipped point.
ALPHA_THETA = 0.0

#: Firing-rate adaptation on E. ``TAU_SFA`` must stay SHORTER than the 50 ms DAF burst it
#: has to truncate: at 100 ms the current integrates over the whole motif and acts as a
#: tonic gain change, cutting the baseline 59% while barely moving the peak. At 10 ms it
#: builds and recovers within the burst. Swept at that timescale, ``A_SFA`` 1.0 is the
#: strongest setting that keeps latency and duration inside the bird's ranges.
A_SFA = 1.0
TAU_SFA = 10.0

T_POST = 200
T_BURN = 500
N_ISTA = 50


def a_jie_for(tau_stdp: float = TAU_STDP, tau_s: float = TAU_S) -> float:
    """STDP amplitude for a given kernel width.

    Derived, not a constant: the per-spike STDP integral goes as ``A_jie * tau``, so the
    amplitude scales inversely with the window to keep that integral fixed. Storing the
    computed number instead would desynchronise the learning rule the first time the
    window changed -- which is the reason this module is Python and not a data file.

    ``tau_stdp <= 0`` falls back to the synaptic constant, matching the network, where an
    unset STDP window means the synaptic one is reused.
    """
    tau_eff = tau_stdp if tau_stdp > 0.0 else tau_s
    return 3e-3 * (10.0 / tau_eff)


# --- causal minimum-phase encoder -------------------------------------------------

#: Code sparsity, and with it how many filters survive: 0.3 gives ~10.7% activation and
#: 56 of 64 filters, 0.6 gives 3.2% and 23, 1.2 collapses to one. Chosen on song
#: cancellation, which is what the network exists to do.
CAUSAL_LAMBDA = 0.3

#: A soft pull toward minimum phase, not a projection. The hard projection has a
#: degenerate fixed point -- minimum phase maps a white magnitude response to an impulse,
#: so each application concentrates energy at lag 0 and whitens the spectrum further.
#: Not tuned: 0.10 / 0.25 / 0.50 all converge to the same bank.
CAUSAL_MP_STRENGTH = 0.25

#: L1 on kernel entries and a roughness penalty over channel and lag. ``CAUSAL_L1`` has to
#: be scaled against the gradient step, which is small: at 0.03 the soft-threshold sits
#: 1300x above it and annihilates the dictionary.
CAUSAL_L1 = 7e-5
CAUSAL_SMOOTH = 1e-3

#: Usage equalisation, OFF. It did not reduce redundancy -- effective rank 29/64 with it
#: and 32/64 without -- and it pinned the code so sparse that the dictionary could not
#: train: kernels stayed 0.67 correlated with their random initialisation.
CAUSAL_LR_BIAS = 0.0
CAUSAL_TARGET_USAGE = 0.03

CAUSAL_LR = 10.0
CAUSAL_EPOCHS = 80
CAUSAL_N_ISTA = 20
