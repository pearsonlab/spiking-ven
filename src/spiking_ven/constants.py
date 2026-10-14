"""Every shared constant: what describes the preparation, and what describes the fit.

Three sections, kept apart because they change for different reasons and carry different
authority:

1. **The preparation.** Sample rate, the DAF window, conduction delays, the auditory
   firing rate. These describe the experiment being modelled; several are measured and
   carry citations, and changing one means claiming something different about the bird.
2. **The network operating point.** The tuned parameters the shipped model was trained
   with. Empirical, not measured: they are what the fit landed on.
3. **The causal encoder operating point.** The same, for the minimum-phase filter bank.

All three are here rather than restated at each use site, which is a drift hazard rather
than a style problem: the stimulus the figure draws has to be the stimulus the network was
trained and scored on, or the figure describes a different experiment. A sweep once built
the network from library defaults rather than section 2, because those values lived only
in a scratch script, and four grid points came back identical and broken.

Some of section 2 is DERIVED -- see :func:`a_jie_for`, which ties the STDP amplitude to
the window width. That is why this is a module and not a data file: a formula has to live
with its inputs or it goes stale.

The CLIs read these as argparse defaults, so the tests in ``test_cli.py`` assert WIRING
(the flag reads the constant) rather than values. Asserting a value against the constant
it is defined by is tautological; what catches a wrong value is
``test_reproduction_causal.py``, which runs the model.
"""

from __future__ import annotations

__all__ = [
    # 1. the preparation
    "SR",
    "KERNEL_WIDTH_MS",
    "PEAK_RATE_HZ",
    "DAF_WN_AMPLITUDE",
    "DAF_WINDOW_S",
    "AUD_DELAY_MS",
    "AUD_MEAN_RATE_HZ",
    "HVC_DELAY_MS",
    # 2. the network operating point
    "N_E", "N_I", "N_HVC", "N_RENDITIONS",
    "B_SCALE", "C_B", "B_HVC_SCALE", "C_HVC",
    "DRIVE_E", "DRIVE_I",
    "TAU_S", "TAU_E", "TAU_STDP",
    "R_I_TH", "J_MAX_IE",
    "A_SFA", "TAU_SFA",
    "T_POST", "T_BURN", "N_ISTA",
    "a_jie_for",
    # 3. the causal encoder operating point
    "CAUSAL_EPOCHS", "CAUSAL_LAMBDA",
    "CAUSAL_LR", "CAUSAL_L1", "CAUSAL_SMOOTH", "CAUSAL_N_ISTA", "CAUSAL_MP_STRENGTH",
]

# =============================================================================
# 1. The preparation -- what is being modelled
# =============================================================================
#: Corpus sample rate (Hz). The corpus is resampled to this during preprocessing.
SR = 16000

#: Width of the Gaussian HVC burst envelope (ms).
KERNEL_WIDTH_MS = 10.0

#: Peak HVC burst rate (spk/s). Scales inversely with the kernel width so the integrated
#: drive per burst is constant -- derived, never written as a literal.
PEAK_RATE_HZ = 150.0 * 20.0 / KERNEL_WIDTH_MS

#: DAF white noise is mixed at this multiple of song RMS: ~95 dBSPL WN vs ~80 dBSPL song.
#:
#: This scales the *figure's* DAF column, where the noise is added into a window of the
#: song and the relative level therefore matters. It does NOT affect the K2/K3 metrics:
#: ``evaluate.build_stimuli`` sends its white noise through the same RMS-normalising
#: encoder path as every other stimulus, so the amplitude divides straight back out.
#: See the note in :mod:`spiking_ven.evaluate`.
DAF_WN_AMPLITUDE = 5.6

#: Auditory population mean firing rate during song (Hz), the encoder's spike-rate
#: calibration target.
#:
#: Schneider & Woolley 2013, Neuron 79:141-152 (DOI 10.1016/j.neuron.2013.04.038) measure
#: 2.4 +/- 2.7 Hz for the broad-spiking neurons of the higher-level auditory cortex -- the
#: sparse, selective, background-invariant population this encoder models. 2.5 realises
#: 2.30 Hz after the refractory period. It was 15.0, an unsourced default repeated across
#: seven call sites, which made the auditory input ~4x too fast.
AUD_MEAN_RATE_HZ = 2.5

#: Window within the motif that the DAF stimulus perturbs (seconds).
#:
#: 50 ms, matching the noise bursts in Mandelblat-Cerf et al. 2014 -- the experiment the
#: responder statistics come from. It was 200 ms, which is long enough that the modelled
#: response keeps building through the burst instead of being the transient the paper
#: measures (peak 118 ms in at 200 ms, against 88 ms at 50 ms).
DAF_WINDOW_S = (0.400, 0.450)

#: Cochlea -> AIV-E conduction delay (ms).
#:
#: NOT a measured conduction delay. Mandelblat-Cerf et al. 2014 report 23 +/- 12 ms as the
#: AIV *response latency* -- noise onset to the first 2 ms bin followed by five
#: significant bins -- which is the end-to-end observable this model should PREDICT, not
#: consume. Using it as the conduction delay double-counts: the network then integrates
#: for a further ~43 ms, putting the modelled latency at ~66 ms, and makes the 23 ms
#: comparison circular.
#:
#: Now 5 ms, a plausible synaptic value, which is what every validated run uses. The
#: latency is therefore PREDICTED, not consumed: the model lands at 27-34 ms against the
#: bird's 23 +/- 12, with the residual being genuine network integration time. At 23 ms it
#: was double-counted and the comparison was circular.
AUD_DELAY_MS = 5

#: HVC -> AIV-I conduction delay (ms). Estimated -- no direct measurement exists. Zero in
#: the validated operating point: HVC drives I directly and any delay here shifts the
#: motor prediction relative to the auditory signal it has to cancel.
HVC_DELAY_MS = 0


# =============================================================================
# 2. The network operating point -- what the fit landed on
# =============================================================================

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

TAU_S = 20.0
TAU_E = 30.0

#: STDP coincidence window, pinned independently of the synaptic ``TAU_S``.
TAU_STDP = 20.0

#: I -> E, the only plastic weights in the model. Zero-initialised, so only song-driven
#: E cells ever acquire inhibition.
R_I_TH = 6.0
J_MAX_IE = 1.0

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


# =============================================================================
# 3. The causal minimum-phase encoder operating point
# =============================================================================


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

CAUSAL_LR = 10.0
CAUSAL_EPOCHS = 80
CAUSAL_N_ISTA = 20
