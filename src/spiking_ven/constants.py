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

Some of section 2 is DERIVED -- see :func:`rho_for`, which ties the STDP amplitude to
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
    "HVC_SPIKES_PER_BURST",
    "HVC_INTRA_BURST_HZ",
    "DAF_WN_AMPLITUDE",
    "DAF_WINDOW_S",
    "DAF_N_TRIALS",
    "AUD_DELAY_MS",
    "AUD_MEAN_RATE_HZ",
    "HVC_DELAY_MS",
    "HVC_INTRO",
    # 2. the network operating point
    "N_E", "N_I", "N_HVC", "N_RENDITIONS",
    "E_L_MV", "V_TH_MV", "V_RESET_MV", "C_E_PF", "C_I_PF",
    "DT_MS", "TAU_E", "TAU_I", "TAU_AMPA", "TAU_GABA",
    "TARGET_RATE_E_HZ", "I_DRIVE_I_PA", "NOISE_E", "NOISE_I",
    "E_AMPA_MV", "E_GABA_MV",
    "C_AUD", "AUD_MEAN_NS", "AUD_SD_NS",
    "C_HVC", "HVC_MEAN_NS", "HVC_SD_NS",
    "C_EI", "EI_MEAN_NS", "EI_SD_NS",
    "G_NMDA_MEAN_NS", "TAU_NMDA_RISE", "TAU_NMDA_DECAY", "ALPHA_NMDA", "MG_MM", "E_NMDA_MV",
    "T_REFR_MS", "TAU_STDP", "R_I_HZ", "BETA_ADAPT_PA", "TAU_ADAPT",
    "T_POST", "T_BURN", "N_ISTA",
    "RHO_NS_AT_10MS", "rho_for", "RHO_PRE_NS", "J_IE_ROW_MAX_NS",
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

#: HVC input: one stereotyped burst per neuron per motif -- finchsim's HVC_RA generator
#: (``ra.build_hvcra_spike_trains``, here ``common.generate_hvc_bursts``), onsets on a
#: jittered grid that tiles the motif. It matches documented HVC_RA statistics: ~1 burst
#: per motif, 3.9 +/- 1.9 spikes per burst, 6-10 ms long, 400-600 Hz within the burst,
#: sub-ms jitter across renditions (Hahnloser et al. 2002; Picardo et al. 2016).
#:
#: Spikes per HVC burst and the rate within it, as finchsim's ``params.HVCRA_RA``: 4 spikes
#: at 500 Hz span 6 ms.
HVC_SPIKES_PER_BURST = 4
HVC_INTRA_BURST_HZ = 500.0

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

#: DAF renditions per evaluation, each with an independent noise draw. The responder,
#: latency and duration tests are per neuron, so their power -- and every DAF metric --
#: depends on this count. Mandelblat-Cerf et al. 2014 had ~100 bursts per neuron (the
#: Fig. 7G and 7J rasters show 100 and 126 trials); with 20, responses read later,
#: shorter and larger per responder (docs/design-notes.md).
DAF_N_TRIALS = 100

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

#: HVC activity before each motif: ``(n_notes, note_ms, gap_ms)`` introductory notes, each
#: a short stereotyped HVC sequence repeated note to note, with no sound, and a gap after
#: each one (the last gap precedes the motif). Without it the HVC-driven inhibition of E
#: switches on only at motif onset, and E fires through its first ~50 ms. ``(0, 0, 0)``
#: disables it.
HVC_INTRO = (2, 80, 20)

#: HVC -> AIV-I conduction delay (ms). Estimated -- no direct measurement exists. Zero in
#: the validated operating point: HVC drives I directly and any delay here shifts the
#: motor prediction relative to the auditory signal it has to cancel.
HVC_DELAY_MS = 0


# =============================================================================
# 2. The network operating point -- what the fit landed on
# =============================================================================
# Units: times in ms, membrane potentials in mV, currents in pA, conductances in nS,
# capacitances in pF, rates in Hz. Synapses are conductances: each synaptic weight is the
# peak conductance (nS) one presynaptic spike opens, acting through its reversal
# potential. See spiking_ven.units and docs/MODEL.md for the membrane equation. The
# equations are in spiking_ven.network; docs/design-notes.md has the calibration history.

#: Simulation step (ms) of the Brian2 network. Inputs and outputs stay on a 1 ms grid.
#:
#: The operating point below was recalibrated for 0.1 ms when the network moved from a
#: 1 ms discrete-time numpy loop to Brian2 (docs/design-notes.md, "Brian2 port"). The 1 ms
#: loop held each conductance at its peak for a whole step, so a synapse decaying with tau
#: delivered ``1 / (1 - exp(-1 / tau))`` ms of peak conductance per spike instead of tau:
#: 18% more for AMPA (3 ms), 5% for adaptation, 3% for GABA_A. The AMPA weights are
#: scaled by that factor (x1.176) to deliver the same charge; adaptation and the STDP
#: baseline were then re-swept.
DT_MS = 0.1

#: Population sizes (E projection neurons, I interneurons, HVC inputs) and training length.
N_E = 600
N_I = 150
N_HVC = 600
N_RENDITIONS = 600

#: Membrane: rest, threshold and reset (mV), shared by E and I -- the values of Wang's
#: (2002) pyramidal cells and interneurons -- and capacitances (pF): 500 for E, 200 for I.
#: With the time constants these give leak conductances ``C / tau``: 16.7 nS for E,
#: 20 nS for I (Wang 2002's interneuron exactly).
E_L_MV = -70.0
V_TH_MV = -50.0
V_RESET_MV = -70.0
C_E_PF = 500.0
C_I_PF = 200.0

#: Membrane time constants (ms) of E and I.
TAU_E = 30.0
TAU_I = 10.0

#: Synaptic decays (ms). The auditory -> E and HVC -> I inputs and E -> I are AMPA
#: (EPSC decay 2-5 ms; 2, 3 and 5 calibrate alike, docs/design-notes.md); I -> E is GABA_A
#: (IPSC decay ~5-10 ms in adult cortex; 5 as in Brunel & Wang 2001). In songbirds, Luo &
#: Perkel (2002, J Neurophysiol) report developmental changes in GABA_A IPSC decay, 5-20 ms
#: in juveniles -- a different circuit, but the range the sweep covered. 15 ms, inside it:
#: at 5 ms the DAF response broke into ~15 ms volleys (E -> I feedback recruits I, shunting
#: silences E, E rebounds), which 15-20 ms smooths out (docs/design-notes.md). Each presynaptic
#: spike adds its peak conductance to one that decays with these. Learning does not depend on ``TAU_AMPA``: ``TAU_STDP`` sets it.
TAU_AMPA = 3.0
TAU_GABA = 15.0

#: Synaptic reversal potentials (mV): AMPA and NMDA at 0, GABA_A at rest (Wang 2002), so
#: I -> E inhibition is shunting -- it does nothing at rest and grows as E depolarises.
E_AMPA_MV = 0.0
E_GABA_MV = -70.0

#: Tonic drives. E's is derived inside the network so the drive alone fires E at
#: ``TARGET_RATE_E_HZ`` (453 pA here, ``units.tonic_drive_pa``). I's is a fixed current
#: (pA) far below I's threshold current (400 pA), so I fires only when HVC drives it.
TARGET_RATE_E_HZ = 25.0
I_DRIVE_I_PA = 53.7

#: Intensity (pA sqrt(ms)) of the white current noise on each E and I cell: numerically the
#: SD of the current averaged over a 1 ms step, whatever the simulation step. On E it gives
#: a free-membrane voltage SD of ~5.8 mV.
NOISE_E = 754.0
NOISE_I = 0.0

#: Auditory -> E, fast (AMPA) component: each E cell connects to each encoder channel with
#: probability ``C_AUD``, with log-normal peak conductances of this mean and SD (nS). Set
#: together with ``G_NMDA_MEAN_NS`` (the slow NMDA component carries part of the drive)
#: and with the encoder's output scale.
C_AUD = 0.05
AUD_MEAN_NS = 44.8
AUD_SD_NS = 22.3

#: HVC -> I, the motor prediction limb, and the only thing that moves the interneuron
#: rate; log-normal peak conductances of this mean and SD (nS) per connection.
C_HVC = 0.1
HVC_MEAN_NS = 7.48
HVC_SD_NS = 3.74

#: E -> I, fixed: log-normal peak conductances of this mean and SD (nS). Four times the
#: conductance that matches the earlier current-based E -> I: the faster feedback trims
#: the DAF response and moves its peak later (docs/design-notes.md).
C_EI = 0.5
EI_MEAN_NS = 0.220
EI_SD_NS = 0.0220

#: NMDA component of the aud -> E synapse, as in Wang (2002, Neuron 36:955) and Brunel &
#: Wang (2001), on the same connections as the fast weights ``B``. Current (pA):
#: ``I_nmda = (G s) mg(V) (E_NMDA - V)``, with ``G`` each connection's peak NMDA conductance
#: (nS) and ``s`` its channel's gating in [0, 1]. ``G`` is ``B`` rescaled so its mean is
#: ``G_NMDA_MEAN_NS``: a stronger AMPA synapse carries proportionally more NMDA. That mean is
#: the ceiling -- full gating, no Mg2+ block; the block lets ~4% through at rest (~111 nS)
#: and ~14% at threshold (~346 nS), against a leak of 16.7 nS. Per isolated spike near
#: threshold NMDA carries ~190x the AMPA charge (61x at rest), mostly because it lasts
#: ~73 ms against AMPA's ~3.5; its peak current is ~6x AMPA's. With AMPA-fast inputs it is
#: the only slow excitation, and what sustains the DAF response through and past the noise
#: burst; chosen with the auditory weights for duration and magnitude. 0 removes it.
#: Gating per auditory channel: dx/dt = -x / TAU_NMDA_RISE + spikes,
#: ds/dt = -s / TAU_NMDA_DECAY + ALPHA_NMDA x (1 - s)   (ms, 1/ms).
#: Mg2+ block (Jahr & Stevens 1990): mg(V) = 1 / (1 + MG_MM exp(-0.062 V) / 3.57), V in mV.
G_NMDA_MEAN_NS = 2500.0
TAU_NMDA_RISE = 2.0
TAU_NMDA_DECAY = 100.0
ALPHA_NMDA = 0.5
MG_MM = 1.0
E_NMDA_MV = 0.0

#: Absolute refractory period (ms) of the E and I membranes and the input populations.
T_REFR_MS = 2.0

#: STDP window (ms): the I trace's time constant, and the audio gate's (an E spike counts
#: only while the cell's auditory-input trace, decaying with this, exceeds 1/e: for this
#: long after an isolated input spike). Independent of ``TAU_AMPA``.
TAU_STDP = 20.0

#: I -> E, the only plastic weights in the model, as peak GABA_A conductance (nS).
#: Zero-initialised, so
#: only song-driven E cells ever acquire inhibition. ``R_I_HZ`` is the I rate (Hz) the
#: STDP trace is debiased against: I activity at that rate, unrelated to an E spike, gives
#: no net change. The numpy model used 46, just above its I rate during song (43.8 Hz),
#: making the rule mildly depressing. At 0.1 ms the I rate during song is ~42 Hz and the
#: DAF response came out too large at 46; 36, below it, makes the rule mildly
#: potentiating, which strengthens the learned inhibition and trims the response, at a
#: small cost in song specificity (reversed/trained 1.05-1.11 over training seeds, against
#: 1.04-1.10 before). Weights have no upper bound: the rule limits itself (inhibition stops
#: the gated E spikes that drive it).
R_I_HZ = 36.0

#: Firing-rate adaptation on E: a hyperpolarising current that steps up by ``BETA_ADAPT_PA``
#: on every spike and decays with ``TAU_ADAPT``. ``TAU_ADAPT`` must stay shorter than the
#: 50 ms DAF burst it has to truncate; a timescale longer than the burst integrates over
#: the motif and acts as a tonic gain change instead. ``BETA_ADAPT_PA`` was chosen by a sweep
#: at this timescale -- see ``docs/design-notes.md`` -- and re-swept at 0.1 ms (7540 pA at
#: 1 ms; x1.35).
BETA_ADAPT_PA = 10180.0
TAU_ADAPT = 10.0

#: Protocol: silence after each motif (ms), silent burn-in before the first (ms), and ISTA
#: iterations per frame when encoding stimuli for the network.
T_POST = 200
T_BURN = 500
N_ISTA = 50

#: Presynaptic depression of I -> E per I spike while the E cell's audio gate is open (nS).
#: Off (0) in the shipped model; an option for making the learning competitive.
RHO_PRE_NS = 0.0

#: Cap on each E cell's total I -> E weight (nS), applied after every training rendition
#: by scaling the row down. Off (0) in the shipped model.
J_IE_ROW_MAX_NS = 0.0

#: STDP step (peak GABA_A conductance, nS) per pairing at a 10 ms window; see
#: :func:`rho_for`.
RHO_NS_AT_10MS = 7.55


def rho_for(tau_stdp: float = TAU_STDP) -> float:
    """STDP step (peak GABA_A conductance, nS) for a given kernel width.

    Derived, not a constant: the per-spike STDP integral goes as ``rho * tau``, so the
    amplitude scales inversely with the window to keep that integral fixed. Storing the
    computed number instead would desynchronise the learning rule the first time the
    window changed -- which is the reason this module is Python and not a data file.

    The window must be positive; the synaptic ``TAU_AMPA`` plays no part in learning.
    """
    if tau_stdp <= 0.0:
        raise ValueError(f"tau_stdp must be > 0 ms, got {tau_stdp}")
    return RHO_NS_AT_10MS * (10.0 / tau_stdp)


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
