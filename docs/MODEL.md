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

Shipped sizes (`constants.py`): `n_e = 600`, `n_i = 150`, `n_hvc = 60`; `n_aud = 54` for the
shipped encoder (64 filters trained, the ones that never fire pruned).

## Discrete-time dynamics

The simulation runs in 1 ms steps (`dt = 1`). Everything below is `_simulate` in
`vocal_error_net.py`. Membrane potentials, thresholds, drives, weights and noise are
dimensionless.

### Input filtering

Before the loop, each input spike train is filtered by `spike_to_rate`, a causal one-pole filter
with the synaptic time constant `tau_s`:

```
h[t] = exp(-1 / tau_s) * h[t-1] + spikes[t]          (per channel, h[-1] = 0)
```

so one input spike contributes a jump of 1 that decays with `tau_s` -- the postsynaptic current of
that synapse. This gives `h_aud` (from auditory spikes) and `h_hvc` (from HVC spikes).

### Conduction delays

At step `t` the network sees the filtered inputs from `aud_delay_ms` and `hvc_delay_ms` steps
earlier, and zero before the delay has elapsed:

```
h_aud_t = h_aud[:, t - aud_delay_ms]    if t >= aud_delay_ms, else 0
h_hvc_t = h_hvc[:, t - hvc_delay_ms]    if t >= hvc_delay_ms, else 0
```

### E neurons

With `decay_e = exp(-1 / tau_e)`, at each step `t`, in this order:

```
a_e    = B @ h_aud_t                                    audio drive (also the STDP gate)
w_sfa  = decay_sfa * w_sfa                              adaptation decays (decay_sfa = exp(-1/tau_sfa))
u_e    = drive_e + a_e - JIE @ sI[t-1] - w_sfa + noise_e * N(0, 1)
refr_e = max(refr_e - 1, 0)
V_e    = decay_e * V_e + (1 - decay_e) * u_e
V_e    = v_reset                       where refr_e > 0        (clamped while refractory)
sE[t]  = 1                             where V_e > theta_e and refr_e == 0
on a spike:  V_e = v_reset,  refr_e = t_refr_ms,  w_sfa += a_sfa
```

- `V_e` relaxes toward its input `u_e` with time constant `tau_e`; a spike is a crossing of the
  per-neuron threshold `theta_e`. `v_reset = 0`, and all state starts at zero.
- Inhibition uses the I spikes of the **previous** step, `sI[t-1]`. A single I spike acts on E
  for one step only; its lasting effect comes from the E membrane integrating it.
- The noise is a fresh Gaussian draw per neuron per step, added to the input (so its effect on
  `V_e` is scaled by `1 - decay_e`).
- The refractory countdown is decremented before the threshold test, so with `t_refr_ms = 2` a
  cell can fire at `t` and again at `t + 2`. `t_refr_ms = 0` disables the refractory period.
- Spike-frequency adaptation (`a_sfa > 0`): each spike adds `a_sfa` to a hyperpolarising current
  `w_sfa` that decays with `tau_sfa`. The shipped `a_sfa = 1.0` is about 23x `theta_e`, so a spike
  holds a cell quiet for several `tau_sfa` unless its auditory drive is strong.

### NMDA component

The auditory synapses onto E also carry an NMDA conductance (`g_nmda = 60`), following Wang
(2002). Each auditory channel `j` has a gating variable, computed before the loop
(`nmda_gating`):

```
x_j[t] = exp(-1 / nmda_tau_rise) * x_j[t-1] + spikes_j[t]
s_j[t] = d * s_j[t-1] + (1 - d * s_j[t-1]) * (1 - exp(-nmda_alpha * x_j[t]))     d = exp(-1 / nmda_tau_decay)
```

a step of `ds/dt = -s / tau_decay + alpha x (1 - s)` that applies the decay and the saturating
increment each exactly, in turn, so `s` stays in [0, 1] and saturates near 1 under dense input. At each step, before the membrane update, `u_e` gains

```
g_nmda * (B @ s_t) * mg_block(V_mV) * (E_nmda - V_e)
V_mV = e_l_mv + V_e * (v_th_mv - e_l_mv) / theta_e
mg_block(V_mV) = 1 / (1 + mg_mm * exp(-0.062 * V_mV) / 3.57)          (Jahr & Stevens 1990)
```

where `s_t` is delayed like `h_aud_t`, `V_e` is the potential from the previous step, and
`E_nmda` is `e_nmda_mv` expressed in membrane units by the same map. The map sends reset (0) to
`e_l_mv` (-70 mV) and each cell's threshold to `v_th_mv` (-50 mV), so the Mg2+ block and the
driving force act on a physiological voltage scale. NMDA uses the same connections as `B`;
`g_nmda` is its conductance relative to them. At threshold, `g_nmda = 1` carries about 0.055 of
the fast component's charge per isolated spike, so the shipped 60 carries ~3.3x the fast
charge (much less at rest, where Mg2+ blocks it). `g_nmda = 0` removes the term. The STDP audio gate stays on the fast drive
`a_e`, so NMDA changes learning only through the E spikes it causes.

### I neurons

With `decay_i = exp(-1 / tau_i)`, at the same step `t`, after E:

```
u_i    = drive_i + JEI @ sE[t-1] + B_hvc @ h_hvc_t + noise_i * N(0, 1)
refr_i = max(refr_i - 1, 0)
V_i    = decay_i * V_i + (1 - decay_i) * u_i
V_i    = v_reset                       where refr_i > 0
sI[t]  = 1                             where V_i > theta_i and refr_i == 0
on a spike:  V_i = v_reset,  refr_i = t_refr_ms
```

I has no adaptation, one scalar threshold `theta_i`, and sees the E spikes of the previous step.
Each step draws the E noise first and then the I noise, from the network's own generator
(`default_rng(seed + 10000)`) unless `transform(..., rng=g)` supplies another.

### Plasticity (STDP on JIE)

Only in `fit`. After the E and I updates of step `t`:

```
tau     = tau_stdp if tau_stdp > 0 else tau_s
x_i     = exp(-1 / tau) * x_i + sI[t]                   presynaptic (I) trace
gate_e  = 1 where a_e > 0, else 0                       audio gate
JIE    += A_jie * outer(sE[t] * gate_e, x_i - xi_th)
JIE     = clip(JIE, 0, J_max_ie)
xi_th   = r_i_th * tau * 1e-3
```

- **Direction.** An E spike reads the trace of recent I spikes (including an I spike in the same
  step). I firing shortly before an E spike potentiates that I->E synapse; there is no term for an
  E spike followed by I firing, so the window is causal (I-before-E only).
- **Audio gate.** Only E spikes that occur while the cell receives auditory drive (`B @ h_aud > 0`)
  update their row. E spikes driven by tonic drive or noise alone never change `JIE`. Because
  `h_aud` decays with `tau_s`, the gate stays open for a while after the cell's last auditory
  input, so shortening `tau_s` narrows the window in which learning can happen.
- **Threshold `xi_th`.** `r_i_th` (Hz) times the window `tau` (ms, hence `1e-3`) is approximately
  the mean trace of an I cell firing irregularly at `r_i_th`. An E spike potentiates synapses from I
  cells whose trace is above that level and depresses those below it, so I activity that is merely
  at its baseline rate gives no net change. With `r_i_th = 0` the rule only potentiates.
- **Bounds.** Weights are clipped to `[0, J_max_ie]` after every step, so depression stops at zero.
- **Amplitude.** `A_jie` defaults to `constants.a_jie_for(tau_stdp, tau_s) = 3e-3 * 10 / tau`, so
  the integral of the learning window stays fixed when its width changes.
- Updates are applied online, so later steps of the same `fit` call already feel them. `JIE` is
  stored as float32 between calls.

## Thresholds

Both are set once, in the constructor.

**`theta_e`** -- the level the E membrane, started at 0 with no noise and constant input
`drive_e`, reaches after `1 / target_rate` steps. Tonic drive alone therefore fires each E cell every
`1 / target_rate` ms (40 ms, 25 Hz, at the operating point):

```
theta_e = max(drive_e * (1 - decay_e ** (1 / target_rate)), 1e-3)     # same for every E cell
```

At the operating point `theta_e = 0.06 * (1 - exp(-40/30)) = 0.044`. It is stored per neuron and
saved per neuron, so it can be set cell by cell.

**`theta_i`** -- the same construction for a fixed reference input `target_rate_i * 10` (the 10 is a
fixed 10 ms reference, so `theta_i` does not move with `tau_s`):

```
theta_i = max(target_rate_i * 10 * (1 - decay_i ** (1 / target_rate_i)), 1e-3)
```

At the operating point `theta_i = 0.5 * (1 - exp(-2)) = 0.43`, far above `drive_i = 0.058`, so I
fires only when HVC drives it.

## Weight initialisation

Fixed weights are sparse and log-normal (`common._lognormal_sparse`): a fraction `c` of the entries,
placed at random, are nonzero, with log-normal values of the given mean and standard deviation.
All three are drawn from `default_rng(seed)`, in the order below, and stored as float32.

| Matrix | Connection probability | Mean weight per connection | Std |
|---|---|---|---|
| `B` | `c_B` (0.05) | `B_scale / n_aud` (10 / 54 = 0.19) | half the mean |
| `B_hvc` | `c_hvc` (0.1) | `B_hvc_scale` (0.333), not divided by `n_hvc` | half the mean |
| `JEI` | `c_JEI` (0.5) | `0.17 / sqrt(n_e * c_JEI)` (0.0098) | 10% of the mean |
| `JIE` | -- | 0 (all zero) | -- |

Starting `JIE` at zero means only E cells that song drives ever acquire inhibition; cells that
song does not reach keep zero rows and respond fully to novel sound.

## Input spike trains

- **Auditory** (`common.acts_to_spikes`): the encoder activations are rectified (negative
  coefficients produce no spikes), scaled by one population-wide factor chosen so that the mean
  rate over the training rendition (the song plus the silence after it) is `AUD_MEAN_RATE_HZ`
  (2.5 Hz), and sampled as Poisson counts per 1 ms bin. Every other stimulus (reversed song, DAF) is scaled by the factor from the training
  song, not its own.
- **HVC** (`common.generate_hvc_spikes`): neuron `i` of `n_hvc` has a Gaussian rate envelope
  centred at the `i`-th of `n_hvc` evenly spaced times across the motif, width `KERNEL_WIDTH_MS`
  (10 ms) and peak `PEAK_RATE_HZ` (300 Hz). Each 1 ms bin spikes with probability `rate * 1e-3`.
- **Intro notes** (`common.generate_hvc_intro`, `HVC_INTRO = (2, 80, 20)`): before each motif,
  two 80 ms introductory notes of HVC activity, each followed by 20 ms of silence, with no
  sound. Each note is one sparse HVC sequence at the motif's burst density (about 6 of 60 neurons
  bursting at evenly spaced times), repeated identically. They are passed to `fit`/`transform` as
  `hvc_prelude`: the network runs them first, with zero auditory input, and the returned spikes
  start at motif onset. No learning happens during them, because the STDP gate needs auditory
  drive. Their purpose is to have I active when the song starts.
- Both are passed through `apply_refractory` (2 ms), which keeps at most one spike per bin and
  drops spikes closer than 2 ms to the previous kept one.

**Training** (`cli/train_ven.py`): `N_RENDITIONS = 600` calls to `fit`, one per rendition of the
motif plus `T_POST = 200` ms of silence, each preceded by the intro notes; the first is preceded
by `T_BURN = 500` ms of silence, which ends with them.
Each call starts the membranes from zero; only `JIE` carries over. The HVC spike train is the same
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
- **Gain** (`CausalFilterEncoder.encode_signal`): each frame's code is divided by the norm of its
  reconstruction and multiplied by the window mean removed above, taken from the previous frame
  (the newest one the code read), so its size reports sound level rather than how well the filters
  matched.
- **Filters** (`causal_filters.learn`): learned to reconstruct the past window from the sparse
  code, pulled toward minimum phase (energy as early in the window as the magnitude response
  allows, which keeps the encoder's latency short), then pruned to those that fire.

## Units

| Quantity | Unit | Examples |
|---|---|---|
| Time step | 1 ms | the loop index `t`, `dt` |
| Time constants, delays, refractory period | ms | `tau_e`, `tau_i`, `tau_s`, `tau_stdp`, `tau_sfa`, `nmda_tau_rise`, `nmda_tau_decay`, `aud_delay_ms`, `t_refr_ms` |
| Potentials, thresholds, drives, noise std | dimensionless | `V_e`, `theta_e`, `drive_e`, `noise_e`, `v_reset` |
| Weights, adaptation increment | dimensionless | `B`, `JIE`, `J_max_ie`, `a_sfa`, `A_jie` |
| NMDA conductance, gating | dimensionless (conductance relative to `B`); `s` in [0, 1] | `g_nmda`, `s_nmda` |
| NMDA rate constant | 1 / ms | `nmda_alpha` |
| Voltages for the NMDA map only | mV | `e_l_mv`, `v_th_mv`, `e_nmda_mv` |
| Mg2+ concentration | mM | `mg_mm` |
| Filtered input `h` | spikes (a sum of decaying unit jumps) | `h_aud`, `h_hvc` |
| Rates used for thresholds | spikes per ms | `target_rate` (0.025 = 25 Hz), `target_rate_i` |
| Other rates | Hz | `r_i_th`, `AUD_MEAN_RATE_HZ`, `PEAK_RATE_HZ` |
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
