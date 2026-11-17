# The model

What the code computes, written for someone who wants to read or change it. Everything here
is taken from `src/spiking_ven/vocal_error_net.py` (the network), `common.py` (input spike
trains, weight initialisation), `constants.py` (the operating point) and the encoder modules
(`filterbank.py`, `cochleagram.py`, `causal_filters.py`). Where this page and the code
disagree, the code is right. Why the parameters have the values they do is in
[`design-notes.md`](design-notes.md).

## Pipeline

A song waveform is scaled to a reference RMS and turned into a cochleagram: 40 ERB-spaced
gammatone channels (200-8000 Hz), per-band RMS in 4 ms windows advancing every 2 ms, square-root
compressed. A bank of causal spectro-temporal filters, each spanning the 16 frames (32 ms) strictly
before now, sparse-codes that cochleagram frame by frame with ISTA, so the code at time `t` depends
on no audio at or after `t`. The rectified code is upsampled to 1 ms and converted to Poisson spike
trains, one *auditory channel* per surviving filter (`n_aud` = the encoder's `n_channels`),
calibrated to a population mean of 2.5 Hz (2.16 Hz realised over a rendition, 2.66 Hz during the
song, after refractory thinning). Separately, a synthetic HVC population supplies the premotor signal: each
HVC neuron bursts once per motif, at its own time. The auditory spikes drive excitatory (E) neurons,
the HVC spikes drive inhibitory (I) neurons, E drives I, and I inhibits E through the plastic
weights `JIE`. During training (`fit`), STDP strengthens I->E synapses where I firing predicts
auditory-driven E firing, so after training the HVC-timed inhibition cancels the E response to the
bird's own song. What remains -- the E response to sound that HVC did not predict, such as
distorted auditory feedback -- is the error signal.

## Network structure

| Matrix | Shape (post x pre) | From -> to | Plastic |
|---|---|---|---|
| `B` | `n_e x n_aud` | auditory channels -> E | no |
| `B_hvc` | `n_i x n_hvc` | HVC -> I | no |
| `JEI` | `n_i x n_e` | E -> I | no |
| `JIE` | `n_e x n_i` | I -> E (inhibitory) | yes, during `fit` only |

Shipped sizes (`constants.py`): `n_e = 600`, `n_i = 150`, `n_hvc = 600`; `n_aud = 54` for the
shipped encoder (64 filters trained, the ones that never fire pruned).

## Discrete-time dynamics

The simulation runs in 1 ms steps (`dt = 1`). Everything below is `_simulate` and `_step` in
`vocal_error_net.py`. Potentials are in mV, currents in pA, conductances in nS, capacitances in
pF (pA / nS = mV, nS / pF = 1 / ms), times in ms.

### Membrane

Each E and I cell is a leaky integrate-and-fire membrane whose synapses are all conductances,
acting through their reversal potentials:

```
C dV/dt = -g_L (V - E_L) - sum_k g_k(t) (V - E_k) + I
```

with `g_L = C / tau` and `I` the tonic drive, adaptation (E) and noise currents. It is stepped by
**exponential Euler** (`_step`): with every conductance held over the 1 ms step, V relaxes exactly
toward the conductance-weighted reversal,

```
g_tot = g_L + sum_k g_k
V_inf = (g_L E_L + sum_k g_k E_k + I) / g_tot
V    <- V_inf + (V - V_inf) * exp(-g_tot / C)
```

which is stable however large the synaptic conductances get, and with none reduces exactly to the
leaky update `V <- E_L + d (V - E_L) + (1 - d) I / g_L`, `d = exp(-1 / tau)`. Shipped values:

| | E | I |
|---|---|---|
| rest `E_L`, reset `V_reset` | -70 mV | -70 mV |
| threshold `V_th` | -50 mV | -50 mV |
| capacitance `C` | 500 pF | 200 pF |
| time constant at rest `tau` | 30 ms | 10 ms |
| leak conductance `g_L = C / tau` | 16.7 nS | 20 nS |

The capacitances are Wang's (2002). Synaptic conductance shortens the effective time constant
`C / g_tot` below `tau` whenever synapses are open.

### Synapses

Every synaptic conductance is a trace per presynaptic cell (or auditory/HVC channel) that jumps by
1 on a spike and decays with the synapse's time constant, times the weight's peak conductance:

```
h[t] = exp(-1 / tau_syn) * h[t-1] + spikes[t]          (h[-1] = 0)
g_syn = W_nS @ h
```

| Synapse | Weight | Trace | `tau_syn` | Reversal |
|---|---|---|---|---|
| auditory -> E | `B` | `h_aud` (`spike_to_rate`, before the loop) | `tau_s` = 3 ms | AMPA, `E_AMPA` = 0 mV |
| HVC -> I | `B_hvc` | `h_hvc` (`spike_to_rate`, before the loop) | `tau_s` = 3 ms | AMPA, 0 mV |
| E -> I | `JEI` | `g_exc`, fed by `sE[t-1]` | `tau_exc` = 3 ms | AMPA, 0 mV |
| I -> E | `JIE` | `g_inh`, fed by `sI[t-1]` | `tau_inh` = 5 ms | GABA_A, `E_GABA` = -70 mV |
| auditory -> E, NMDA | `G_nmda` | gating `s` (below) | 2 ms rise, 100 ms decay | `E_NMDA` = 0 mV |

Each weight is the **peak conductance (nS)** one presynaptic spike opens. A recurrent `tau` of 0
is a one-step pulse.

**Inhibition is shunting.** `E_GABA` equals rest, so I -> E inhibition passes no current at rest;
it pulls a depolarised cell back toward -70 mV (never below), with a driving force of up to 20 mV
at threshold, and it adds to the membrane's conductance, dividing the effect of the excitation
rather than subtracting a fixed current from it. The learned inhibition cancels the song through
that division.

### Conduction delays

At step `t` the network sees the input traces from `aud_delay_ms` and `hvc_delay_ms` steps
earlier, and zero before the delay has elapsed:

```
h_aud_t = h_aud[:, t - aud_delay_ms]    if t >= aud_delay_ms, else 0
h_hvc_t = h_hvc[:, t - hvc_delay_ms]    if t >= hvc_delay_ms, else 0
```

### E neurons

At each step `t`, in this order:

```
g_inh  = exp(-1 / tau_inh) * g_inh + sI[t-1]            GABA_A trace, one per I cell
w_sfa  = exp(-1 / tau_sfa) * w_sfa                      adaptation current decays
conductances:  B @ h_aud_t  (E_AMPA),  JIE @ g_inh  (E_GABA),  (G_nmda @ s_t) * mg(V_e)  (E_NMDA)
I_e    = I_drive_e - w_sfa + noise_e * N(0, 1)          currents, pA
refr_e = max(refr_e - 1, 0)
V_e    = _step(V_e, ...)                                exponential Euler, above
V_e    = V_reset                       where refr_e > 0        (clamped while refractory)
sE[t]  = 1                             where V_e > V_th and refr_e == 0
on a spike:  V_e = V_reset,  refr_e = t_refr_ms,  w_sfa += a_sfa
```

- All state starts at rest (`V = E_L`), and only `JIE` carries over between calls.
- **Tonic drive.** `I_drive_e` is derived from `TARGET_RATE_E_HZ` (`units.tonic_drive_pa`): the
  current that, alone and without noise, takes the membrane from reset to threshold in
  `1000 / 25 = 40` ms, so it fires E at 25 Hz: 453 pA.
- **Noise** is a fresh Gaussian current per neuron per step, SD `noise_e` = 754 pA (a free-membrane
  voltage SD of about 5.8 mV).
- The refractory countdown is decremented before the threshold test, so with `t_refr_ms = 2` a
  cell can fire at `t` and again at `t + 2`. `t_refr_ms = 0` disables the refractory period.
- **Spike-frequency adaptation** (`a_sfa > 0`): each spike adds `a_sfa` = 7540 pA to a
  hyperpolarising current that decays with `tau_sfa` = 10 ms, so a spike holds a cell quiet for
  several `tau_sfa` unless its auditory drive is strong.

### NMDA component

The auditory synapses onto E also carry an NMDA conductance, following Wang (2002). Each auditory
channel `j` has a gating variable, computed before the loop (`nmda_gating`):

```
x_j[t] = exp(-1 / nmda_tau_rise) * x_j[t-1] + spikes_j[t]
s_j[t] = d * s_j[t-1] + (1 - d * s_j[t-1]) * (1 - exp(-nmda_alpha * x_j[t]))     d = exp(-1 / nmda_tau_decay)
```

a step of `ds/dt = -s / tau_decay + alpha x (1 - s)` that applies the decay and the saturating
increment each exactly, in turn, so `s` stays in [0, 1] and saturates near 1 under dense input.
At each step the E conductances include

```
g_nmda = (G_nmda @ s_t) * mg_block(V_e)                       reversal E_nmda = 0 mV
G_nmda = B * g_nmda_mean_ns / b_aud_mean_ns                   (nmda_conductances())
mg_block(V) = 1 / (1 + mg_mm * exp(-0.062 * V) / 3.57)        (Jahr & Stevens 1990, V in mV)
```

where `s_t` is delayed like `h_aud_t` and `V_e` is the potential from the previous step. `G_nmda`
is each connection's peak NMDA conductance: the AMPA weights `B` rescaled so their mean is
`g_nmda_mean_ns` = 3000 nS, so a stronger AMPA synapse carries proportionally more NMDA. That is
the ceiling -- full gating, no Mg2+ block. The block lets ~4% through at rest (~133 nS) and ~14% at
threshold (~416 nS), against a leak of 16.7 nS. Per isolated spike, against the mean AMPA synapse:

| V (mV) | unblocked | NMDA / AMPA charge | NMDA / AMPA peak current |
|---|---|---|---|
| -70 (rest) | 0.044 | 73x | 2.4x |
| -60 | 0.080 | 130x | 4.3x |
| -50 (threshold) | 0.139 | 226x | 7.4x |

The charge ratio is large mostly because one spike's NMDA gating lasts ~73 ms (peak 0.68 at
+6 ms) against the AMPA trace's ~3.5; a single spike nearly saturates a channel's gating, so NMDA
reports whether a channel is active more than how fast it fires. `g_nmda_mean_ns = 0` removes it.
The STDP audio gate reads input spikes, not either conductance, so NMDA changes learning only
through the E spikes it causes.

### I neurons

At the same step `t`, after E:

```
g_exc  = exp(-1 / tau_exc) * g_exc + sE[t-1]            AMPA trace, one per E cell
conductance:  JEI @ g_exc + B_hvc @ h_hvc_t  (E_AMPA)
I_i    = I_drive_i + noise_i * N(0, 1)
refr_i = max(refr_i - 1, 0)
V_i    = _step(V_i, ...)
V_i    = V_reset                       where refr_i > 0
sI[t]  = 1                             where V_i > V_th and refr_i == 0
on a spike:  V_i = V_reset,  refr_i = t_refr_ms
```

I has no adaptation. Its tonic drive `I_drive_i` = 53.7 pA holds it 2.7 mV above rest, far below
the 400 pA (20 mV) it would need to reach threshold, so I fires only when HVC drives it. Each step
draws the E noise first and then the I noise, from the network's own generator
(`default_rng(seed + 10000)`) unless `transform(..., rng=g)` supplies another.

### Plasticity (STDP on JIE)

Only in `fit`. After the E and I updates of step `t` (`JIE` in peak conductance, nS):

```
tau     = tau_stdp                                      STDP window, > 0 (20 ms)
x_i     = exp(-1 / tau) * x_i + sI[t]                   presynaptic (I) trace
gate_e  = 1 where B @ n_aud > 0, else 0                 audio gate; n_aud = auditory spikes
                                                        per channel in (t - tau, t], delayed
                                                        by aud_delay_ms like the drive
JIE    += a_jie * outer(sE[t] * gate_e, x_i - xi_th)
JIE     = clip(JIE, 0, j_max_ie)
xi_th   = r_i_th * tau * 1e-3
```

- **Direction.** An E spike reads the trace of recent I spikes (including an I spike in the same
  step). I firing shortly before an E spike potentiates that I->E synapse; there is no term for an
  E spike followed by I firing, so the window is causal (I-before-E only).
- **Audio gate.** Only E spikes within `tau` ms of an auditory input spike to that cell update its
  row (`stdp_audio_gate`), so cells the song never reaches keep `JIE = 0`, and E spikes from tonic
  drive or noise away from the song's input never change `JIE`. The gate is defined on input spike
  events with the STDP window, so `tau_stdp` alone sets the learning window and `tau_s` plays no
  part in it.
- **Threshold `xi_th`.** `r_i_th` (Hz) times the window `tau` (ms, hence `1e-3`) is approximately
  the mean trace of an I cell firing irregularly at `r_i_th`. An E spike potentiates synapses from I
  cells whose trace is above that level and depresses those below it, so I activity that is merely
  at its baseline rate gives no net change. The shipped `r_i_th = 40` Hz is just below the I rate
  during song (43.1 Hz), which makes the rule mildly potentiating; far below it (an early 6 Hz)
  nearly every gated E spike potentiates whatever I has been doing, and the inhibition loses its
  song specificity. With `r_i_th = 0` the rule only potentiates.
- **Bounds.** Weights are clipped to `[0, j_max_ie]` (503 nS) after every step, so depression
  stops at zero. In the shipped model 62% of the I->E weights are nonzero, with mean 47 nS, 90th
  percentile 109 nS; 0.04% reach the bound.
- **Amplitude.** `a_jie` defaults to `constants.a_jie_for(tau_stdp) = 7.55 nS * 10 / tau`, so
  the integral of the learning window stays fixed when its width changes.
- Updates are applied online, so later steps of the same `fit` call already feel them. `JIE` is
  stored as float32 between calls.

## Weight initialisation

Fixed weights are sparse and log-normal (`common._lognormal_sparse`): a fraction `c` of the entries,
placed at random, are nonzero, with log-normal values of the given mean and standard deviation
(peak conductance, nS). All three are drawn from `default_rng(seed)`, in the order below, and
stored as float32.

| Matrix | Connection probability | Mean (nS) | SD (nS) | Peak current at rest (pA) |
|---|---|---|---|---|
| `B` | `c_B` (0.05) | `b_aud_mean_ns` (38.1) | `b_aud_sd_ns` (19.0) | ~2700 |
| `B_hvc` | `c_hvc` (0.1) | `b_hvc_mean_ns` (6.36) | `b_hvc_sd_ns` (3.18) | ~450 |
| `JEI` | `c_JEI` (0.5) | `j_ei_mean_ns` (0.14) | `j_ei_sd_ns` (0.014) | ~10 |
| `JIE` | -- | 0 (all zero) | -- | -- |

The auditory synapse is strong because each E cell samples only ~2.7 of the 54 sparse auditory
channels. Starting `JIE` at zero means only E cells that song drives ever acquire inhibition;
cells that song does not reach keep zero rows and respond fully to novel sound.

**Earlier model files** had current-based synapses (and before that a dimensionless membrane);
they cannot be represented exactly here, so `load` refuses them. spiking-ven at commit `50f5304`
or earlier runs them.

## Input spike trains

- **Auditory** (`common.acts_to_spikes`): the encoder activations are rectified (negative
  coefficients produce no spikes), scaled by one population-wide factor chosen so that the mean
  rate over the training rendition (the song plus the silence after it) is `AUD_MEAN_RATE_HZ`
  (2.5 Hz), and sampled as Poisson counts per 1 ms bin. Every other stimulus (reversed song, DAF) is scaled by the factor from the training
  song, not its own.
- **HVC** (`common.generate_hvc_bursts`): finchsim's HVC_RA generator
  (`ra.build_hvcra_spike_trains`), with 600 neurons. The motif is cut into `n_hvc`
  equal slots; neuron `i` starts one burst at a uniformly jittered time in slot `i` and fires
  `HVC_SPIKES_PER_BURST` (4) spikes at `HVC_INTRA_BURST_HZ` (500 Hz), a 6 ms burst, rounded down
  to the 1 ms grid. Nothing is random within a burst, and the same onsets repeat every rendition.
  This matches documented HVC_RA statistics: about 1 burst per motif, 3.9 +/- 1.9 spikes per
  burst, 6-10 ms long, 400-600 Hz within the burst, under 1 ms jitter across motifs (Hahnloser et
  al. 2002; Picardo et al. 2016). At 600 neurons a burst starts every ~1.4 ms and each I cell
  has ~60 HVC inputs, so the HVC drive to I is nearly continuous through the motif. The model's
  stimulus record names the generator; models trained on the earlier Gaussian-envelope input
  (60 neurons) are rejected on load.
- **Intro notes** (`common.generate_hvc_intro`, `HVC_INTRO = (2, 80, 20)`): before each motif,
  two 80 ms introductory notes of HVC activity, each followed by 20 ms of silence, with no
  sound. Each note is one sparse HVC sequence at the motif's burst density (about 57 of 600 neurons
  bursting at evenly spaced times), repeated identically. They are passed to `fit`/`transform` as
  `hvc_prelude`: the network runs them first, with zero auditory input, and the returned spikes
  start at motif onset. No learning happens during them, because the STDP gate needs auditory
  drive. Their purpose is to have I active when the song starts.
- The auditory spikes are passed through `apply_refractory` (2 ms), which keeps at most one spike per bin and
  drops spikes closer than 2 ms to the previous kept one. HVC bursts need no thinning: their
  spikes are already 2 ms apart.

**Training** (`cli/train_ven.py`): `N_RENDITIONS = 600` calls to `fit`, one per rendition of the
motif plus `T_POST = 200` ms of silence, each preceded by the intro notes; the first is preceded
by `T_BURN = 500` ms of silence, which ends with them.
Each call starts the membranes at rest; only `JIE` carries over. The HVC spike train is the same
every rendition, and the auditory input cycles through encodings of the template motif and the
corpus's other renditions of the song.

## The encoder, briefly

- **Level.** The waveform is divided by the RMS of a reference signal (the training song) and
  multiplied by `CAUSAL_RMS_REF`, so every stimulus is encoded at one calibrated level.
- **Cochleagram** (`filterbank.gammatone_spectrogram`, `cochleagram.cochleagram`): gammatone
  filters on an ERB scale; frame `t` holds the RMS over the window spanning frames `[t-1, t+1)`;
  then a square root.
- **Mean subtraction** (`causal_filters.causal_meansub`): one number per frame -- the mean over the
  trailing 16 frames, averaged over channels -- is subtracted from every channel.
- **Sparse code** (`causal_filters.infer`): with `X_past(t)` the 16 frames before `t`, the
  coefficients `s(t)` minimise `0.5 * ||X_past(t) - sum_k s_k V_k||^2 + lam * sum_k |s_k|` by ISTA
  (`CAUSAL_N_ISTA` iterations in training, `N_ISTA = 50` when encoding stimuli for the network).
  Correlated filters explain each other away rather than firing together.
- **No gain step.** The activations passed on are the ISTA coefficients themselves. Sound level
  reaches them only through the spectral contrast that survives mean subtraction, which grows
  with loudness; the spike-rate calibration then sets the overall scale.
- **Filters** (`causal_filters.learn`): learned to reconstruct the past window from the sparse
  code, pulled toward minimum phase (energy as early in the window as the magnitude response
  allows, which keeps the encoder's latency short), then pruned to those that fire.

## Units

| Quantity | Unit | Examples |
|---|---|---|
| Time step | 1 ms | the loop index `t`, `dt` |
| Time constants, delays, refractory period | ms | `tau_e`, `tau_i`, `tau_s`, `tau_inh`, `tau_exc`, `tau_stdp`, `tau_sfa`, `nmda_tau_rise`, `nmda_tau_decay`, `aud_delay_ms`, `t_refr_ms` |
| Membrane and reversal potentials | mV | `V_e`, `V_i`, `e_l_mv`, `v_th_mv`, `v_reset_mv`, `e_ampa_mv`, `e_gaba_mv`, `e_nmda_mv` |
| Currents: drives, adaptation, noise SD | pA | `i_drive_e_pa`, `i_drive_i_pa`, `a_sfa_pa`, `noise_e_pa` |
| Capacitance, leak conductance | pF, nS | `c_e_pf`, `c_i_pf`, `g_l_e`, `g_l_i` |
| Synaptic weights, their mean / SD, the I->E bound and STDP step | nS (peak conductance) | `B`, `B_hvc`, `JEI`, `JIE`, `b_aud_mean_ns`, `j_max_ie_ns`, `a_jie_ns` |
| NMDA conductance | nS (peak, per connection, unblocked) | `g_nmda_mean_ns`, `nmda_conductances()` |
| NMDA gating | in [0, 1] | `s_nmda` |
| NMDA rate constant | 1 / ms | `nmda_alpha` |
| Mg2+ concentration | mM | `mg_mm` |
| Synaptic traces | spikes (a sum of decaying unit jumps) | `h_aud`, `h_hvc`, `g_inh`, `g_exc` |
| Rates | Hz | `target_rate_hz`, `r_i_th`, `AUD_MEAN_RATE_HZ`, `HVC_INTRA_BURST_HZ` |
| Spike matrices | 0/1 per neuron per 1 ms bin | `transform` output, `(n_e, T)` |

## Glossary

- **HVC** -- a premotor nucleus of the songbird forebrain (a proper name, not an acronym). Its
  projection neurons burst sparsely, each at a fixed time in the song, so together they act as a
  clock of where the bird is in the song. Here, the motor prediction input to I.
- **AIV** -- ventral intermediate arcopallium, an auditory region that responds to errors in the
  bird's own song and projects to the dopaminergic VTA. The E population models its projection
  neurons.
- **DAF** -- distorted auditory feedback: a burst of noise played during singing so the bird hears
  an error. The model's test of the error signal (a 50 ms window, `DAF_WINDOW_S`).
- **VEN** -- vocal error network: this model.
- **E / I** -- excitatory and inhibitory populations.
- **STDP** -- spike-timing-dependent plasticity: a weight change set by the relative timing of pre-
  and postsynaptic spikes. Here, I (pre) before E (post) potentiates I->E.
- **NMDA** -- a glutamate receptor whose current is slow (~100 ms decay) and blocked by Mg2+
  near rest, so it flows mainly in depolarised cells. Here, an optional slow component of the
  auditory -> E synapse.
- **ISTA** -- iterative shrinkage-thresholding algorithm: a gradient step on the reconstruction
  error followed by a soft threshold, repeated; it solves the L1-penalised sparse coding problem.
- **STRF** -- spectro-temporal receptive field: the time-frequency pattern of sound a neuron
  responds to. Each encoder filter `V_k` (channels x 16 frames) is one.
