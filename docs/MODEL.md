# The model

What the code computes, written for someone who wants to read or change it. Everything here
is taken from `src/spiking_ven/network.py` (the equations), `vocal_error_net.py` (parameters,
training and inference), `common.py` (input spike trains, weight initialisation), `constants.py` (the operating point) and the encoder modules
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
weights `J_IE`. During training (`fit`), STDP strengthens I->E synapses where I firing predicts
auditory-driven E firing, so after training the HVC-timed inhibition cancels the E response to the
bird's own song. What remains -- the E response to sound that HVC did not predict, such as
distorted auditory feedback -- is the error signal.

## Network structure

| Matrix | Shape (post x pre) | From -> to | Synapse | Plastic |
|---|---|---|---|---|
| `A` | `n_e x n_aud` | auditory channels -> E | AMPA | no |
| `G` | `n_e x n_aud` | auditory channels -> E | NMDA (`A` rescaled) | no |
| `B` | `n_i x n_hvc` | HVC -> I | AMPA | no |
| `J_EI` | `n_i x n_e` | E -> I | AMPA | no |
| `J_IE` | `n_e x n_i` | I -> E | GABA_A | yes, during `fit` only |

Shipped sizes (`constants.py`): `n_e = 600`, `n_i = 150`, `n_hvc = 600`; `n_aud = 54` for the
shipped encoder (64 filters trained, the ones that never fire pruned).

## Dynamics

The network is a set of Brian2 equations, in `src/spiking_ven/network.py` -- the code *is* the
model, and this section transcribes it. It runs in continuous time, integrated by exponential
Euler at `DT_MS` (`constants.py`); the inputs and the returned spike matrices stay on a 1 ms grid.
Brian2 checks the units of every equation: potentials are in mV, currents in pA, conductances in
nS, capacitances in pF, times in ms.

### Membranes

Each E and I cell is a leaky integrate-and-fire membrane whose synapses are conductances acting
through their reversal potentials. For E,

```
C_E dV/dt = g_L (E_L - V) + g_ampa (E_AMPA - V) + g_nmda B(V) (E_NMDA - V)
            + g_gaba (E_GABA - V) + I_E - I_adapt + I_noise
```

and for I,

```
C_I dV/dt = g_L (E_L - V) + g_ampa (E_AMPA - V) + I_I + I_noise
```

with `g_L = C / tau`. A cell spikes when `V > V_th`, is reset to `V_reset` and held there for the
refractory period `t_ref`.

| | E | I |
|---|---|---|
| rest `E_L`, reset `V_reset` | -70 mV | -70 mV |
| threshold `V_th` | -50 mV | -50 mV |
| capacitance `C` | 500 pF | 200 pF |
| time constant at rest `tau` | 30 ms | 10 ms |
| leak conductance `g_L = C / tau` | 16.7 nS | 20 nS |
| refractory period `t_ref` | 2 ms | 2 ms |

- **Tonic drive.** `I_E` is derived from `TARGET_RATE_E_HZ` (`units.tonic_drive_pa`): the current
  that, alone and without noise, takes the membrane from reset to threshold in `1000 / 25 = 40` ms,
  `I_E / g_L = (V_th - V_reset e^(-T/tau)) / (1 - e^(-T/tau))` relative to rest (453 pA). `I_I` =
  53.7 pA holds I 2.7 mV above rest, far below the 400 pA it would need to reach threshold, so I
  fires only when HVC drives it.
- **Noise.** `I_noise` is white current noise of intensity `sigma` (pA sqrt(ms)), held for each
  step: `I_noise = sigma * N(0, 1) / sqrt(dt)`. The diffusion is the same at any `dt`; numerically,
  `sigma` = 754 is the SD of the noise current averaged over 1 ms (a free-membrane voltage SD of
  about 5.8 mV on E). I has none.
- **Adaptation** (E only): `dI_adapt/dt = -I_adapt / tau_adapt`, and each spike adds `beta`.
- **Integration.** Exponential Euler is exact for a membrane linear in `V` with its conductances
  held over the step, and stable however large they get. The Mg2+ block `B(V)` (below) is the one
  nonlinearity, so it is evaluated once per step, at the step's starting potential (Brian2's
  `constant over dt`), as is the noise.

### Synapses

Each synaptic conductance jumps by the weight -- the **peak conductance (nS)** one presynaptic
spike opens -- and decays:

```
dg_ampa/dt = -g_ampa / tau_ampa        on a spike from j:  g_ampa += w_ij
dg_gaba/dt = -g_gaba / tau_gaba
```

| Synapse | Weights | Conductance | Decay | Reversal | Delay |
|---|---|---|---|---|---|
| auditory -> E | `A` | `g_ampa` (E) | `tau_ampa` = 3 ms | `E_AMPA` = 0 mV | `d = AUD_DELAY_MS` |
| HVC -> I | `B` | `g_ampa` (I) | `tau_ampa` | `E_AMPA` | `HVC_DELAY_MS` (0) |
| E -> I | `J_EI` | `g_ampa` (I) | `tau_ampa` | `E_AMPA` | 0 |
| I -> E | `J_IE` | `g_gaba` (E) | `tau_gaba` = 15 ms | `E_GABA` = -70 mV | 0 |
| auditory -> E, NMDA | `G` | `g_nmda` (E) | gating `s` (below) | `E_NMDA` = 0 mV | `d` |

**Inhibition is shunting.** `E_GABA` equals rest, so I -> E inhibition passes no current at rest;
it pulls a depolarised cell back toward -70 mV (never below), and it adds to the membrane's
conductance, dividing the effect of the excitation rather than subtracting a fixed current from it.
The learned inhibition cancels the song through that division.

### NMDA component

Following Wang (2002), each auditory channel `j` has an NMDA gating variable, driven by the
channel's spikes after the conduction delay `d`:

```
dx_j/dt = -x_j / tau_nmda_rise                  on a spike:  x_j += 1
ds_j/dt = -s_j / tau_nmda_decay + alpha x_j (1 - s_j)
g_nmda,i = sum_j G_ij s_j
B(V) = 1 / (1 + [Mg2+] exp(-0.062 V) / 3.57)    (Jahr & Stevens 1990; V in mV, [Mg2+] in mM)
```

with `tau_nmda_rise` = 2 ms, `tau_nmda_decay` = 100 ms, `alpha` = 0.5 / ms, `[Mg2+]` = 1 mM. `G` is
each connection's peak NMDA conductance: the AMPA weights `A` rescaled so their mean is
`g_nmda_mean_ns`, so a stronger AMPA synapse carries proportionally more NMDA. That is the ceiling
-- full gating, no Mg2+ block; the block lets ~4% through at rest and ~14% at threshold. The NMDA
current is the only slow excitation, and it is what sustains the DAF response through and past the
noise burst. `g_nmda_mean_ns = 0` removes it.

### Plasticity (STDP on `J_IE`)

Only in `fit`. Each I cell keeps a presynaptic trace, and each E cell an auditory trace:

```
dx_j/dt = -x_j / tau_stdp          on an I spike:              x_j += 1
da_i/dt = -a_i / tau_stdp          on an auditory input spike: a_i += 1   (after the delay d)
sigma_i = [a_i > 1/e]                                                     the audio gate
```

and at every E spike,

```
J_IE[i, j] <- max(0, J_IE[i, j] + rho sigma_i (x_j - theta)),      theta = r_I tau_stdp
```

- **Direction.** An E spike reads the trace of recent I spikes, so I firing shortly before an E
  spike potentiates that I -> E synapse; there is no term for an E spike followed by I firing, so
  the window is causal (I before E only).
- **Audio gate.** `sigma_i` is open while the E cell's auditory trace exceeds 1/e: for `tau_stdp`
  after an isolated auditory input spike, longer after several. Only gated E spikes learn, so cells
  the song never reaches keep `J_IE = 0`, and spikes driven by tonic drive or noise away from the
  song never change `J_IE`. (Before the Brian2 port the gate was a box window -- any input spike in
  the last `tau_stdp` ms; the trace is the same for isolated spikes.)
- **Threshold `theta`.** `r_I tau_stdp` is the mean trace of an I cell firing irregularly at
  `r_I`. An E spike potentiates synapses from I cells whose trace is above it and depresses those
  below, so I activity merely at its baseline rate gives no net change.
- **Bounds.** Weights are clipped at zero, so depression stops there. They have no upper bound:
  the rule limits itself, because the inhibition it builds stops the gated E spikes that drive it.
- **Amplitude.** `rho` defaults to `constants.rho_for(tau_stdp) = 7.55 nS * 10 / tau_stdp`, so the
  integral of the learning window stays fixed when its width changes. Outside `fit` it is 0.
- Updates are applied online, so later spikes of the same `fit` call already feel them. All other
  state starts at rest on every call; only `J_IE` carries over.

## Weight initialisation

Fixed weights are sparse and log-normal (`common._lognormal_sparse`): a fraction `c` of the entries,
placed at random, are nonzero, with log-normal values of the given mean and standard deviation
(peak conductance, nS). All three are drawn from `default_rng(seed)`, in the order below. Brian2
creates a synapse for each nonzero entry; `J_IE` is all-to-all, starting at zero.

| Matrix | Connection probability | Mean (nS) | SD (nS) |
|---|---|---|---|
| `A` | `c_aud` | `aud_mean_ns` | `aud_sd_ns` |
| `B` | `c_hvc` | `hvc_mean_ns` | `hvc_sd_ns` |
| `J_EI` | `c_ei` | `ei_mean_ns` | `ei_sd_ns` |
| `J_IE` | 1 | 0 | -- |

(values in `constants.py`). The auditory synapse is strong because each E cell samples only ~2.7
of the 54 sparse auditory channels.

**Earlier model files** (the numpy engine of spiking-ven <= 0.6) are refused by `load`: retrain.

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
Each call starts the network at rest; only `J_IE` carries over. The HVC spike train is the same
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
| Simulation step | ms | `dt_ms` (`DT_MS`); inputs and outputs on a 1 ms grid |
| Time constants, delays, refractory period | ms | `tau_e`, `tau_i`, `tau_ampa`, `tau_gaba`, `tau_stdp`, `tau_adapt`, `tau_nmda_rise`, `tau_nmda_decay`, `aud_delay_ms`, `t_refr_ms` |
| Membrane and reversal potentials | mV | `v`, `e_l_mv`, `v_th_mv`, `v_reset_mv`, `e_ampa_mv`, `e_gaba_mv`, `e_nmda_mv` |
| Currents: drives, adaptation step | pA | `i_drive_e_pa`, `i_drive_i_pa`, `beta_adapt_pa` |
| Noise intensity | pA sqrt(ms) | `noise_e`, `noise_i` |
| Capacitance, leak conductance | pF, nS | `c_e_pf`, `c_i_pf`, `g_l_e`, `g_l_i` |
| Synaptic weights, their mean / SD, and the STDP step | nS (peak conductance) | `A`, `B`, `J_EI`, `J_IE`, `aud_mean_ns`, `rho_ns` |
| NMDA conductance | nS (peak, per connection, unblocked) | `g_nmda_mean_ns`, `nmda_conductances()` |
| NMDA gating, STDP traces | dimensionless | `s`, `x`, `x_stdp`, `a_aud` |
| NMDA rate constant | 1 / ms | `alpha_nmda` |
| Mg2+ concentration | mM | `mg_mm` |
| Rates | Hz | `target_rate_hz`, `r_i_hz`, `AUD_MEAN_RATE_HZ`, `HVC_INTRA_BURST_HZ` |
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
