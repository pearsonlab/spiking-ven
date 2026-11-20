# Proposed changes to the paper's Methods

For `~/Overleaf/critic paper/main.tex`: sections *Auditory data*, *Sparse auditory coding*,
*Vocal error network*, *Model parameters*, *Training and plasticity* and *Model evaluation*.
The Overleaf file has not been edited. LaTeX is ready to paste.

All "Now:" quotes and line references were checked against `main.tex` as of 2026-11-20 13:53,
its last modification. Re-check them if the file has changed since. Items marked DONE are
already applied.

The changes fall into two groups:

- **Part A, shipped.** These are in release v0.7.0: the Brian2 port. Apply them whatever else
  is decided.
- **Parts B-F, experimental.** These describe the experiment baseline on branch
  `brian2-swac` (`configs/baseline.json`; history in `docs/contrast-experiments.md`). They are
  not merged and not the package defaults. Apply them only if the paper adopts that model.
  Each part says what it replaces.

| Part | Topic | Status |
|---|---|---|
| A | Simulation in Brian2 (time step, noise, gate, sign/unit fixes, retuned values) | shipped, v0.7.0 |
| B | Balanced E/I network with two inhibitory populations | experimental |
| C | I_HVC -> E plasticity: Vogels rate-target rule | experimental |
| D | Plastic HVC -> I_HVC: a learned temporal basis | experimental |
| E | Schneider & Woolley two-stage auditory front end | experimental |
| F | Evaluation: new measures and protocol changes | experimental |
| G | Parameter table for the baseline | experimental |
| H | Open issues to resolve before writing | -- |

---

## A. Simulation in Brian2 (shipped, v0.7.0)

### A1. Simulation (Vocal error network, first paragraph)

**Now:** "... leaky integrate-and-fire neuron pools, simulated at \qty{1}{\ms} resolution."

**Proposed:**

```latex
We modeled the vocal error network as a pair of excitatory ($N_E=600$) and inhibitory
($N_I=150$) conductance-based leaky integrate-and-fire neuron pools, simulated in Brian2
\parencite{stimberg2019brian2} by exponential Euler integration with a \qty{0.1}{\ms} time step.
```

Why: the network is now continuous-time equations integrated by Brian2, so the 1 ms discrete
update is gone. Cite Stimberg, Brette & Goodman (2019), eLife 8:e47314.

### A2. GABA sign and `V` -> `V_i` -- DONE

Applied in `main.tex` (checked 2026-11-20 13:53). The E equation (line 178) reads
`+ (\sum_j J^{IE}_{ij} s^I_j)(V_{GABA} - V_i)`, and the I equation (line 168)
`(V_{AMPA} - V_i)`. The one remaining bare `V` is `g(V)` in the NMDA term, which A3
rewrites as `m(V_i)`.

### A3. Units of the NMDA term (bug in the current text)

**Now:** `\sum_l A_{il} (s^{Aud}_l (V_{AMPA} - V_i) + s^{NMDA}_l g(V)\, (V_{NMDA} - V_i))`, with
`g(V) = g^{NMDA} / (1 + ...)`.

Here `A_il` (nS) multiplies `g(V)` (nS), which gives nS². In the model, each connection's NMDA
conductance is `A_il` rescaled so that the mean is `g^NMDA`.

**Proposed:**

```latex
\sum_l A_{il} \left(s^{Aud}_l (V_{AMPA} - V_i)
      + \frac{g^{NMDA}}{\bar{A}} s^{NMDA}_l\, m(V_i)\, (V_{NMDA} - V_i)\right)
...
m(V) &= \frac{1}{1 + \frac{[\mathrm{Mg}^{2+}]_o}{3.57} e^{-0.062 V}}
```

Here $\bar A$ is the mean nonzero entry of $A$. The table row for $g^{NMDA}$ is unchanged ("mean
peak NMDA conductance per connection"). In the part B-F baseline, NMDA is off ($g^{NMDA}=0$).

### A4. Noise

**Now:** "$\eta_{it}$ Gaussian white noise with standard deviation \qty{754}{\pA} for every
\qty{1}{\ms} timestep."

**Proposed:** "$\eta_i(t)$ Gaussian white noise current of intensity
$\sigma_\eta = \qty{754}{\pA\,\ms^{1/2}}$ (independent across neurons), i.e. a current whose
average over any \qty{1}{\ms} window has standard deviation \qty{754}{\pA}."

Why: a per-step SD is not a property of a continuous-time model. The noise intensity is, and it
gives the same membrane fluctuations at any time step. In the table: `\qty{754}{\pA\,\ms^{1/2}}`.

### A5. The STDP audio gate

**Now:** "$\sigma_i(t)$ is 1 if neuron $i$ received any auditory input spike within the previous
\qty{20}{\ms}"

**Proposed:**

```latex
where $\sigma_i(t) = \Theta\!\left(a_i(t) - e^{-1}\right)$ gates learning on recent auditory
input: $a_i$ is a trace of the auditory spikes neuron $i$ receives,
\begin{equation*}
    \dot{a}_i = -a_i/\tau_{STDP} + \sum_{l : A_{il} > 0} \sum_k \delta(t - t_{lk} - d),
\end{equation*}
so the gate stays open for $\tau_{STDP}$ after an isolated input spike (longer after several).
```

Why: the box window was replaced by an equivalent trace. The trace is a differential equation
like the rest of the model, and it matches the box exactly for isolated input spikes. (Part C
does not use the gate.)

### A6. Training

**Now:** "... prefaced by a \qty{300}{\ms} burn-in period ...". Unchanged: the code's 500 ms
burn-in includes the 200 ms of introductory notes.

Add after "Training cycled through motifs in a fixed order.":
"Each rendition started the network from rest, carrying over only the learned weights."

### A7. Parameter tables (shipped model)

These are the values that change in `constants.py` at v0.7.0. Everything else in Tables 2-3 is
unchanged.

| Table row | Now | Proposed | Note |
|---|---|---|---|
| (new) Simulation step | -- | \qty{0.1}{\ms} (exponential Euler, Brian2) | |
| $\eta$ Noise (E, I) | \qty{754}{\pA}, 0 per 1 ms step | \qty{754}{\pA\,\ms^{1/2}}, 0 | same number, now an intensity |
| $A$ Aud $\to$ E mean (s.d.) | \qty{38.1}{\nano\siemens} (19.0) | \qty{44.8}{\nano\siemens} (22.3) | $\times 1.176$ |
| $B$ HVC $\to$ I mean (s.d.) | \qty{6.36}{\nano\siemens} (3.18) | \qty{7.48}{\nano\siemens} (3.74) | $\times 1.176$ |
| $J^{EI}$ E $\to$ I mean (s.d.) | \qty{0.187}{\nano\siemens} (0.0187) | \qty{0.220}{\nano\siemens} (0.0220) | $\times 1.176$ |
| $\beta$ Adaptation step | \qty{7540}{\pA} | \qty{10180}{\pA} | |
| $r_I$ STDP threshold rate | \qty{46}{\Hz} | \qty{36}{\Hz} | |

The factor 1.176 restores the charge per spike that the 1 ms discrete update delivered:
$1/(1-e^{-1/3})$ ms against \qty{3}{\ms} of peak AMPA conductance. See
`docs/design-notes.md`, "Brian2 port". Suggested sentence for *Model evaluation*, after the list
of tunable parameters:

```latex
The model was first calibrated with a \qty{1}{\ms} discrete-time update and recalibrated for
continuous-time integration at \qty{0.1}{\ms}: AMPA weights were scaled by $1/[3(1-e^{-1/3})]
\approx 1.18$ to preserve the charge each spike delivers, and $\beta$ and $r_I$ re-tuned.
```

Also in the *Training and plasticity* table: $\rho$ = \qty{3.775}{\nano\siemens} is unchanged.

### A8. Loudness normalization (text does not match the shipped code)

**Now** (*Auditory data*, line 132): "Prior to encoding, signals were globally rescaled such
that the template had RMS value 0.1. During model training, all signals were normalized to 0.1
RMS to remove loudness as a confounding factor."

The shipped model (v0.7.0) does *not* normalize each rendition. Every rendition is scaled by
the template's RMS (`build_stimuli`, `rms_from=sig_train`), so quieter renditions arrive
quieter. Their RMS is 0.59-1.0 of the template's, and the template is the loudest. Either:

- **(v0.7.0)** drop the second sentence: "All signals were scaled by the same factor, which
  set the template's RMS to 0.1, so renditions kept their natural loudness differences."; or
- **(baseline, Parts B-F)** keep the second sentence. The baseline normalizes each training
  rendition by its own RMS (`rms_per_rendition`), which is what the sentence describes. Then
  add "The DAF and time-reversed test stimuli were scaled with the template's factor."

---

## B. Balanced E/I network with two inhibitory populations (experimental)

**Replaces:** the single inhibitory pool, which is driven by HVC and E and whose I -> E weights
start at zero and grow by audio-gated STDP. In the baseline that STDP is off ($\rho = 0$).

Sentences in the current text that become false, to remove or replace (*Vocal error network*):

- line 166: "tonic input currents, calibrated such that the E neuron population has a mean
  rate of \qty{25}{\Hz} absent other inputs and the I population only fires when driven by
  HVC input". In the baseline, E receives \qty{2500}{\pA} balanced by $I^{bal}$, giving a
  silent E rate of $\approx\qty{6}{\Hz}$. $I^{bal}$ receives no tonic current and fires on E
  input.
- line 168: the I equation's HVC term $\sum_l B_{il} s^{HVC}_l$. It moves to $I^{HVC}$; $I^{bal}$
  gets no HVC input.
- line 174: "Connection weights $J^{EI}$ are nonzero with probability 0.5 ... HVC input
  weights $B$ are nonzero with probability 0.1 ...". Becomes probability 0.05 (mean 11 nS,
  s.d. 5.5) for $J^{EI}$ and no $B$ (see below).
- line 192: "Connections in $J^{IE}$ ... were initialized to 0 and learned via plasticity
  rules". Becomes fixed, as below.
- Table 2 (`tab:params-net`): $I_E$ (\qty{453}{\pA}, "for a 25 Hz tonic rate"), $I_I$
  (\qty{53.7}{\pA}), $B$, $J^{EI}$ and the NMDA block all change. See Part G.

**Text (Vocal error network):**

```latex
The network contains two inhibitory populations. Balancing interneurons ($I^{bal}$, $N_I=150$)
receive only excitatory input from the E pool, through sparse heterogeneous synapses
($J^{EI}$, connection probability 0.05, lognormal weights of mean \qty{11}{\nano\siemens},
s.d. \qty{5.5}{\nano\siemens}), and inhibit E through fixed synapses ($J^{IE}$, connection
probability 0.5, lognormal, mean \qty{6}{\nano\siemens}, s.d. \qty{3}{\nano\siemens}),
holding the network in an E/I-balanced state. HVC-driven interneurons ($I^{HVC}$,
$N_{IH}=1200$) receive HVC input and no input from E (Section D), and inhibit E through
plastic synapses $W^{HE}$ (Section C). Each $I^{HVC}$ neuron contacts a random subset of
$\approx 40$ E neurons (connection probability $2/30$). E receives a tonic current of
\qty{2500}{\pA}. HVC does not project to $I^{bal}$ or to E.
```

Why: the balanced state quenches auditory transients, so the network tolerates the strong
auditory drive needed for the DAF response. Separating the HVC-driven inhibition makes the
cancellation pathway explicit and keeps HVC input from shifting the mean inhibitory rate.

**Equations:** the E equation gains a second inhibitory term:

```latex
+ \left(\sum_{j} J^{IE}_{ij} s^{I}_j + \sum_k W^{HE}_{ik} s^{IH}_k\right) (V_{GABA} - V_i)
```

$I^{HVC}$ neurons use the same membrane equation as $I^{bal}$.

---

## C. I_HVC -> E plasticity: the Vogels rate-target rule (experimental)

**Replaces:** audio-gated STDP on I -> E (*Training and plasticity*, lines 288-297: the
three-factor rule, its gate and kernel) for the HVC-driven pathway, and the plasticity
table `tab:params-train` ($\rho$, $r_I$, "Initial $J^{IE}$ 0"). Training-table rows that
change: training renditions, 600 -> 150 for the baseline (`configs/baseline.json`; it
converges by about 100, see round K).

**Text:**

```latex
Synapses from $I^{HVC}$ onto E learn by the inhibitory rate-target rule of
\textcite{vogels2011inhibitory}. Each neuron keeps a spike trace,
$\tau_{STDP}\,\dot{x}_n = -x_n + \tau_{STDP}\sum_k \delta(t - t_{nk})$, and
\begin{align*}
    \text{at each } I^{HVC} \text{ spike of } k: \quad
        W^{HE}_{ik} &\leftarrow \big[W^{HE}_{ik} + \eta\,(x^E_i - \alpha_i)\big]_+ \\
    \text{at each E spike of } i: \quad
        W^{HE}_{ik} &\leftarrow \big[W^{HE}_{ik} + \eta\, x^{IH}_k\big]_+ ,
\end{align*}
with $\alpha_i = 2\rho_i\tau_{STDP}$, $\tau_{STDP} = \qty{20}{\ms}$ and
$\eta = \qty{3}{\nano\siemens}$. The target $\rho_i$ is each E neuron's own firing rate in
silence (no song, no HVC input), measured before training, so learning drives each neuron's
rate during singing back to its non-singing rate. The rule is not gated by auditory input.
```

Why: it cancels the predictable auditory drive while holding each cell near its own baseline,
which reproduces the singing vs not-singing per-cell scatter (Mandelblat-Cerf et al. 2014,
Fig 7D). A gated version broke that per-cell match (`contrast-experiments.md`). $\tau_{STDP}$ =
20 ms follows the Vogels supplement.

Known bias, for the text or a footnote: song-locked $I^{HVC}$ firing correlates I and E spikes,
which pulls the fixed point slightly below $\rho_i$. The singing rate ends about 5% below the
silent rate (Section H).

---

## D. Plastic HVC -> I_HVC: a learned temporal basis (experimental)

**Replaces:** fixed sparse HVC -> inhibitory weights.

**Text:**

```latex
Each $I^{HVC}$ neuron initially receives input from 90\% of HVC neurons through weak, uniform
synapses ($B^{H}_{kj} = w_0 = \qty{0.9}{\nano\siemens}$), so that before learning it fires
sparsely ($\approx\qty{1}{\Hz}$) and without temporal preference while the bird sings, and
not at all in silence. These synapses learn which HVC inputs time each $I^{HVC}$ neuron's
spikes to the moments its E targets need inhibition. Each synapse carries a presynaptic trace
and an eligibility trace,
\begin{align*}
    \tau_h\,\dot{x}_{kj} &= -x_{kj} + \tau_h \textstyle\sum_m \delta(t - t^{HVC}_{jm}), \\
    \tau_e\,\dot{e}_{kj} &= -e_{kj} + \tau_e\, x_{kj} \textstyle\sum_m \delta(t - t^{IH}_{km}),
\end{align*}
($\tau_h = \qty{5}{\ms}$, $\tau_e = \qty{10}{\ms}$): an input becomes eligible when it fires
shortly before its $I^{HVC}$ neuron does. Each $I^{HVC}$ neuron $k$ accumulates a credit
$c_k$, the summed change of its outgoing weights under the rule of Section C,
$c_k = \sum_i \Delta W^{HE}_{ik}$, and every \qty{1}{\ms}
\begin{equation*}
    B^{H}_{kj} \leftarrow \big[B^{H}_{kj} + \kappa\, e_{kj}\, c_k\big]_+, \qquad \kappa = 0.02,
\end{equation*}
after which $c_k$ is reset. Inputs that drove spikes at times when the neuron's inhibition was
strengthened (its targets fired above their set point) grow; inputs that drove spikes where
the inhibition was weakened shrink. Between renditions, each neuron's total HVC input $S_k$ is
scaled homeostatically toward a target $I^{HVC}$ rate $r^* = \qty{8}{\Hz}$ during song,
\begin{equation*}
    S_k \leftarrow S_k \cdot \mathrm{clip}\!\left(1 + \gamma\,\frac{r^* - r_k}{r^*},\,
    0.5,\, 1.5\right), \qquad \gamma = 0.02,
\end{equation*}
($r_k$ the neuron's rate in that rendition), and its weights are rescaled divisively to that
total, $B^H_{kj} \leftarrow B^H_{kj}\, S_k / \sum_{j'} B^H_{kj'}$, so inputs that grew take
weight from the rest.
```

Why, and what the experiments showed (round K):

- Starting from near-uniform input, each $I^{HVC}$ neuron concentrates onto a few HVC burst
  times within about 30 renditions.
- Without homeostasis, most cells stayed silent and never learned.
- A 2 Hz target left too few spikes per window. 8 Hz cancels best.
- About 40 targets per I_HVC cell (not 10) are needed so that each E cell's inhibition covers
  the whole motif.
- Cancellation matches a hand-built single-window basis: flatness 52% vs 53%.

---

## E. Schneider & Woolley two-stage auditory front end (experimental)

**Replaces:** the *Sparse auditory coding* subsection (lines 139-158) and the "Sparse
auditory encoder" block of Table 1 (lines 222-233). The *Auditory data* and *Cochleagram*
subsections stay (but see A8). The same 40-channel cochleagram at 500 Hz frames feeds the new
encoder.

What carries over from the current text:

- **The three regularizers** (lines 141-153): $L_1$ on kernel entries, $L_2$ on kernel
  gradients, and the minimum-phase pull with $a = 0.25$. The new STRF training applies the
  same three steps after each update. Keep the itemized list, cepstrum explanation included,
  and change only the numbers. $L_1$: $7\times10^{-5}$ -> $2\times10^{-4}$. Smoothness:
  $10^{-3}$ -> $0.1$. Learning rate: 10 -> 0.1. The steps now act at the gradient step size.
- **The spike conversion** (line 158): Poisson counts in 1 ms bins, a 2 ms refractory period,
  rates calibrated to a 2.5 Hz mean over the rendition (motif plus 200 ms silence). This is
  unchanged.

What changes:

- the model, a rectified autoencoder rather than ISTA sparse coding;
- the context, 64 ms (32 frames) rather than 32 ms (16);
- 128 filters (126 active) rather than 64 (54);
- the input, which is centred by the corpus-mean window rather than locally de-meaned;
- the higher-AC stage, which is new.

Drop $\lambda$ = 0.3, the ISTA iterations and "Sparsity comes from the iterative soft
thresholding". Sparsity now comes from the per-unit thresholds of the second stage.

**Text** (new *Auditory coding* subsection, replacing lines 139-157; regularizer list kept):

```latex
We transformed the cochleagram into auditory cortical responses with a two-stage model after
\textcite{schneider2013sparse}: dense primary responses, and sparse responses in higher
auditory cortex as observed experimentally \parencite{keller2009neural,schneider2013sparse}.

\emph{Primary auditory cortex.} $K = 128$ units with spectrotemporal receptive fields
(STRFs) $V_k$ (40 channels $\times$ \qty{64}{\ms}) respond to the strictly past window $x_t$
of the cochleagram as $r_k(t) = g\,\big[V_k\cdot(x_t - \mu) + b_k\big]_+$, with $\mu$ the
mean window over the corpus. The STRFs were learned as the encoder of a rectified
autoencoder that reconstructs each window from the code through a linear decoder with
unit-norm rows, subject to the following regularizations, each applied after every gradient
step:
% keep the current itemize (L1 on kernel entries; L2 on kernel gradients; minimum-phase pull
% with the cepstrum explanation and a = 0.25) here unchanged
Training ran for 80 epochs over the corpus. Units that never responded were dropped (126
kept), and the gain $g$ was set so that the mean primary rate during song is \qty{32}{\Hz}.

\emph{Higher auditory cortex.} Each primary unit's rate is filtered by a causal
difference-of-Gaussians kernel (excitatory lobe centred \qty{13}{\ms}, s.d. \qty{6.4}{\ms};
inhibitory lobe centred \qty{60}{\ms}, s.d. \qty{42}{\ms}, equal area), starting from the
steady state the primary rate reaches in silence. The output is thresholded per unit at its
90th percentile on the song corpus and rectified, giving the activity $a_l(t)$ of 126
higher-auditory units. The population is divisively normalized,
\begin{equation*}
    \tilde a_l(t) = \frac{a_l(t)}{1 + \big(\sum_{l'} a_{l'}(t)/\sigma\big)^2},
    \qquad \sigma = 0.3,
\end{equation*}
and converted to spikes as before (keep line 158 as it is).
```

Table 1 "Sparse auditory encoder" block, replacement rows:

| Row | Now | Proposed |
|---|---|---|
| Filters trained (active) | 64 (54) | 128 (126) |
| Filter context | 16 frames (32 ms) | 32 frames (64 ms) |
| $\lambda$ code shrinkage | 0.3 | (remove) |
| $L_1$ on kernel entries | 7e-5 | 2e-4 |
| Smoothness on kernel gradients | 1e-3 | 0.1 |
| $a$ minimum-phase pull | 0.25 | 0.25 |
| Learning rate | 10 | 0.1 |
| Training epochs | 80 | 80 |
| ISTA iterations | 20 / 50 | (remove) |
| (new) Primary rate during song | -- | 32 Hz |
| (new) DoG kernel, exc. centre / s.d. | -- | 13 ms / 6.4 ms |
| (new) DoG kernel, inh. centre / s.d. | -- | 60 ms / 42 ms |
| (new) Higher-AC threshold | -- | 90th percentile per unit |
| (new) Normalization $\sigma$, exponent | -- | 0.3, 2 |
| Target mean auditory firing rate | 2.5 Hz | 2.5 Hz |
| Auditory refractory period | 2 ms | 2 ms |

Why:

- Sparse, syllable-locked higher-auditory responses at the reported rates: about 32 Hz primary,
  about 2.5 Hz higher.
- The DoG on *rates* rather than primary spikes is needed for responses that repeat across
  renditions. On spikes, primary Poisson noise dominated, with about 0.5 spikes per excitatory
  lobe.
- Per-unit thresholds give a spread of higher-AC rates, a median of 2.2 Hz with 10-90% of units
  at 0.2-4.9 Hz.
- The squared normalization brings the DAF response to target: broadband noise recruits the
  whole population at once.
- Per-rendition RMS removes a loudness confound: the template was the loudest rendition, and
  loudness correlated 0.82 with higher-AC rate.

Also drop the ISTA / dictionary-learning description and its parameters (Table 2 auditory
rows), and the 1.176 scaling of $A$ (A7) no longer applies. Part G gives the baseline's
$A$ instead.

---

## F. Evaluation: new measures and protocol changes (experimental)

**A sentence that becomes false** (*Model evaluation*, line 324): "the DAF response latency
and magnitude proved most constraining, requiring that auditory input currents to excitatory
cells be strongly NMDA-dominated and that E cells rapidly adapt ...". The baseline has no
NMDA. Proposed: "... the DAF response magnitude proved most constraining. It was set by the
strength of auditory input to E cells together with divisive normalization of the auditory
population, which compresses the broadband noise burst far more than song." The
tunable-parameter list in the same paragraph also changes: add $\sigma$, the $I^{HVC}$
target rate and $\kappa$; drop $g_{NMDA}$, $\rho$ and $r_I$.

**Targets table** (`tab:targets`), new row under *Vocal error neurons*:
`Singing / not-singing rate per neuron (Fig.~7D) & \numrange{0.5}{2} \\`.

Add to *Model evaluation*:

1. **Randomized DAF onsets:** "DAF noise bursts (\qty{50}{\ms}) were placed at random times
   inside syllables, as in the experiment, and each trial was aligned to its own onset."
2. **Singing vs not singing (new target, Fig 7D):** "For each E neuron we compared its rate
   while singing, averaged over all training renditions, with its rate in silence without HVC
   input. Target: population ratio 0.5-2, per-cell points near the diagonal." The first
   \qty{20}{\ms} of every run are excluded. A run without the HVC prelude starts from rest,
   with an onset transient that otherwise inflates the not-singing rate by about 0.4 Hz.
3. **Cancellation measures** (replacing the reversed/trained ratio as the main contrast
   measure):
   - cancellation against not singing,
     $1 - (\bar r_{\mathrm{aligned}} - \bar r_{\mathrm{silent}})/(\bar r_{\mathrm{no\,HVC}} -
     \bar r_{\mathrm{silent}})$;
   - temporal flatness, $1 - \mathrm{sd}_t(r_{\mathrm{aligned}})/\mathrm{sd}_t(r_{\mathrm{no\,HVC}})$,
     computed on each training rendition and averaged.

   Here $r$ is the E population rate, averaged over neurons with auditory input and over noise
   draws.
4. **HVC-shift test:** the cancellation that remains when HVC input is shifted by 150 ms
   measures how much of it is time-locked.

---

## G. Parameter table for the baseline (experimental)

| Parameter | Value |
|---|---|
| $N_E$, $N_I$ ($I^{bal}$), $N_{IH}$ ($I^{HVC}$) | 600, 150, 1200 |
| Tonic current to E | \qty{2500}{\pA} |
| $J^{EI}$: probability, mean (s.d.) | 0.05, \qty{11}{\nano\siemens} (5.5) |
| $J^{IE}$ ($I^{bal}\to$E, fixed): probability, mean (s.d.) | 0.5, \qty{6}{\nano\siemens} (3) |
| $A$ (Aud$\to$E): probability, mean (s.d.) | 0.05, \qty{400}{\nano\siemens} (200) |
| NMDA | off |
| HVC$\to I^{bal}$, HVC$\to$E | none |
| $E_{GABA}$, $\tau_{GABA}$, $\tau_{AMPA}$ | \qty{-70}{\mV}, \qty{15}{\ms}, \qty{3}{\ms} |
| Auditory delay | \qty{2}{\ms} |
| Vogels rule: $\eta$, $\tau_{STDP}$, target | \qty{3}{\nano\siemens}, \qty{20}{\ms}, own silent rate |
| $I^{HVC}\to$E connection probability | 0.0667 ($\approx 40$ targets) |
| HVC$\to I^{HVC}$: probability, $w_0$ | 0.9, \qty{0.9}{\nano\siemens} |
| $\tau_h$, $\tau_e$, $\kappa$ | \qty{5}{\ms}, \qty{10}{\ms}, 0.02 |
| $I^{HVC}$ target rate $r^*$, scaling gain $\gamma$ | \qty{8}{\Hz}, 0.02 |
| Auditory front end | Part E (126 units; $\sigma = 0.3$, squared normalization) |
| Training | 150 renditions cycling 38 motifs |

---

## H. Open issues to resolve before writing

- **Residual peaks.** About half of the song-locked modulation remains on each rendition
  (flatness about 52%). This traces to rendition-to-rendition variability of single higher-AC
  units relayed through one ~500 nS synapse, and to multi-window $I^{HVC}$ cells. The paper
  should either report it or wait for a fix.
- **Singing rate.** It sits about 5% below silent (pooled ratio 0.94, median per cell 0.98).
  This is within target, but could be tuned by scaling the Vogels targets.
- **DAF peak latency.** 76-86 ms in recent models, against the 35-75 ms target.
- **Biological plausibility to check:**
  - 1200 $I^{HVC}$ cells firing about 8 Hz during song;
  - \qty{400}{\nano\siemens} auditory synapses;
  - per-rendition loudness normalization;
  - the homeostatic scaling of HVC input.
- **Rate-target bias.** `set_rate_targets` includes the same onset transient (+6% on
  $\rho_i$). It is left as is because changing it changes every trained model.
