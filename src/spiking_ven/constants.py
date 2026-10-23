"""Every shared constant: what describes the preparation, and what describes the fit.

Three sections, kept apart because they change for different reasons and carry different
authority:

1. **The preparation.** Sample rate, the DAF window, conduction delays, the auditory
   firing rate. These describe the experiment being modelled; several are measured and
   carry citations, and changing one means claiming something different about the bird.
2. **The network operating point.** The tuned parameters the shipped model was trained
   with. Empirical, not measured: they are what the fit landed on.
3. **The causal encoder operating point.** The same, for the minimum-phase filter bank.

All three live here rather than being restated at each use site: the stimulus the figure
draws has to be the stimulus the network was trained and scored on, or the figure
describes a different experiment.

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
    "TAU_I", "C_JEI", "NOISE_E", "NOISE_I", "TARGET_RATE_E", "TARGET_RATE_I", "T_REFR_MS",
    "G_NMDA", "NMDA_TAU_RISE", "NMDA_TAU_DECAY", "NMDA_ALPHA", "MG_MM", "E_NMDA_MV",
    "E_L_MV", "V_TH_MV",
    "T_POST", "T_BURN", "N_ISTA",
    "a_jie_for",
    # 3. the causal encoder operating point
    "CAUSAL_N_BASES", "CAUSAL_K_FRAMES", "CAUSAL_N_COCH_CH", "CAUSAL_FRAME_RATE",
    "CAUSAL_LO_HZ", "CAUSAL_HI_HZ", "CAUSAL_RMS_REF",
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
#: It sets the DAF stimulus wherever one is built -- the figure's DAF column and the
#: responder metrics alike. The noise is mixed into a window of the song, and both
#: normalisation stages take the song as their reference, so this relative level is what
#: reaches the network.
DAF_WN_AMPLITUDE = 5.6

#: Auditory population mean firing rate during song (Hz), the encoder's spike-rate
#: calibration target.
#:
#: Schneider & Woolley 2013, Neuron 79:141-152 (DOI 10.1016/j.neuron.2013.04.038) measure
#: 2.4 +/- 2.7 Hz for the broad-spiking neurons of the higher-level auditory cortex -- the
#: sparse, selective, background-invariant population this encoder models. The target is
#: a mean over the whole rendition (song plus the silence after it). Refractory thinning
#: removes some spikes, so the realised rate is 2.16 Hz over the rendition and 2.66 Hz
#: during the song itself.
AUD_MEAN_RATE_HZ = 2.5

#: Window within the motif that the DAF stimulus perturbs (seconds).
#:
#: 50 ms, matching the noise bursts in Mandelblat-Cerf et al. 2014 -- the experiment the
#: responder statistics come from. A much longer burst lets the modelled response keep
#: building instead of being the transient the paper measures.
DAF_WINDOW_S = (0.400, 0.450)

#: Cochlea -> AIV-E conduction delay (ms), on top of the encoder's own processing time.
#:
#: Not measured: a plausible synaptic value. It must not be set to the 23 +/- 12 ms of
#: Mandelblat-Cerf et al. 2014, which is the AIV *response latency* -- the end-to-end
#: observable the model predicts (~30 ms here), not an input to it.
#:
#: The encoder adds delay of its own: a cochleagram frame is 2 ms and its analysis
#: window 4 ms, and the code at t uses only frames that end by t. Changing the encoder's
#: frame timing therefore moves the predicted latency, and this value is what keeps the
#: total in place.
AUD_DELAY_MS = 2

#: HVC -> AIV-I conduction delay (ms). Estimated -- no direct measurement exists. Zero in
#: the validated operating point: HVC drives I directly and any delay here shifts the
#: motor prediction relative to the auditory signal it has to cancel.
HVC_DELAY_MS = 0


# =============================================================================
# 2. The network operating point -- what the fit landed on
# =============================================================================
# Units: times in ms; membrane potentials, thresholds, drives, weights and noise are
# dimensionless (the membrane relaxes toward its input, and a spike is a crossing of
# theta); rates in Hz unless a name says otherwise. The simulation step is 1 ms.

#: Population sizes (E projection neurons, I interneurons, HVC inputs) and training length.
N_E = 600
N_I = 150
N_HVC = 60
N_RENDITIONS = 600

#: Auditory -> E. Each E cell connects to each encoder channel with probability ``C_B``,
#: so it samples only a few channels. The mean weight per connection is
#: ``B_SCALE / n_aud`` (n_aud = the encoder's channel count).
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

#: Input synaptic time constant (ms) for aud -> E and HVC -> I, and the membrane time
#: constants (ms) of E and I.
TAU_S = 20.0
TAU_E = 30.0
TAU_I = 10.0

#: NMDA component of the aud -> E synapse, as in Wang (2002, Neuron 36:955) and Brunel &
#: Wang (2001). Off by default (``G_NMDA = 0``). It uses the same connections as the fast
#: weights ``B``; ``G_NMDA`` is its conductance relative to them, in units of the leak.
#: Gating per auditory channel: dx/dt = -x / NMDA_TAU_RISE + spikes,
#: ds/dt = -s / NMDA_TAU_DECAY + NMDA_ALPHA x (1 - s)   (ms, 1/ms).
#: Mg2+ block (Jahr & Stevens 1990): B(V) = 1 / (1 + MG_MM exp(-0.062 V) / 3.57), V in mV.
#: Current: G_NMDA (B s) B(V) (E_NMDA - V).
G_NMDA = 0.0
NMDA_TAU_RISE = 2.0
NMDA_TAU_DECAY = 100.0
NMDA_ALPHA = 0.5
MG_MM = 1.0
E_NMDA_MV = 0.0

#: Voltage scale for the Mg2+ block and the NMDA driving force: the dimensionless E
#: membrane maps reset (0) to E_L and threshold (theta_e) to V_th, the values of Wang's
#: pyramidal cells (mV).
E_L_MV = -70.0
V_TH_MV = -50.0

#: E -> I connection probability (fixed weights).
C_JEI = 0.5

#: Standard deviation of the Gaussian noise added to the E and I input every 1 ms step.
NOISE_E = 0.1
NOISE_I = 0.0

#: Firing rates (spikes per ms) that set the thresholds: ``theta_e`` is chosen so the tonic
#: drive alone fires E at TARGET_RATE_E, and ``theta_i`` from TARGET_RATE_I.
TARGET_RATE_E = 0.025
TARGET_RATE_I = 0.05

#: Absolute refractory period (ms) of the E and I membranes and the input populations.
T_REFR_MS = 2.0

#: STDP coincidence window (ms), set independently of the synaptic ``TAU_S``.
TAU_STDP = 20.0

#: I -> E, the only plastic weights in the model. Zero-initialised, so only song-driven
#: E cells ever acquire inhibition. ``R_I_TH`` is the I rate (Hz) the STDP trace is
#: debiased against; ``J_MAX_IE`` caps each weight.
R_I_TH = 6.0
J_MAX_IE = 1.0

#: Firing-rate adaptation on E. ``TAU_SFA`` must stay shorter than the 50 ms DAF burst it
#: has to truncate; a timescale longer than the burst integrates over the motif and acts
#: as a tonic gain change instead. ``A_SFA`` was chosen by a sweep at this timescale --
#: see ``docs/design-notes.md``.
A_SFA = 1.0
TAU_SFA = 10.0

#: Protocol: silence after each motif (ms), silent burn-in before the first (ms), and ISTA
#: iterations per frame when encoding stimuli for the network.
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

#: Front end: filters trained, context window (frames), cochleagram channels, frame rate
#: (Hz; 500 -> 2 ms frames, so 16 frames = 32 ms of past), band edges (Hz), and the RMS
#: every waveform is scaled to before the cochleagram.
CAUSAL_N_BASES = 64
CAUSAL_K_FRAMES = 16
CAUSAL_N_COCH_CH = 40
CAUSAL_FRAME_RATE = 500
CAUSAL_LO_HZ = 200.0
CAUSAL_HI_HZ = 8000.0
CAUSAL_RMS_REF = 0.1

#: Code sparsity, and with it how many filters survive: 0.3 keeps 54 of 64 filters, with
#: ~11% of coefficients active during training. Larger values keep far fewer. Chosen on
#: song cancellation, which is what the network exists to do.
CAUSAL_LAMBDA = 0.3

#: A soft pull toward minimum phase, not a projection. The hard projection has a
#: degenerate fixed point -- minimum phase maps a white magnitude response to an impulse,
#: so each application concentrates energy at lag 0 and whitens the spectrum further.
#: Not tuned: values between 0.1 and 0.5 converge to the same bank.
CAUSAL_MP_STRENGTH = 0.25

#: L1 on kernel entries and a roughness penalty over channel and lag. ``CAUSAL_L1`` has to
#: be scaled against the gradient step: at ``CAUSAL_LR`` the soft threshold
#: ``lr * l1 = 7e-4`` is about the size of the median step (9e-4), whereas 0.03 would put it
#: over 300x above and zero the dictionary.
CAUSAL_L1 = 7e-5
CAUSAL_SMOOTH = 1e-3

#: Learning rate, epochs, and ISTA iterations per frame during training.
CAUSAL_LR = 10.0
CAUSAL_EPOCHS = 80
CAUSAL_N_ISTA = 20
