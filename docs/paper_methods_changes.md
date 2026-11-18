# Proposed changes to the paper's Methods (Brian2 port)

For `~/Overleaf/critic paper/main.tex`, sections *Vocal error network*, *Model parameters*,
*Training and plasticity*. The Overleaf file has not been edited. Each item gives the current
text, the proposed text, and why. LaTeX is ready to paste.

## 1. Simulation (Vocal error network, first paragraph)

**Now:** "... leaky integrate-and-fire neuron pools, simulated at \qty{1}{\ms} resolution."

**Proposed:**

```latex
We modeled the vocal error network as a pair of excitatory ($N_E=600$) and inhibitory
($N_I=150$) conductance-based leaky integrate-and-fire neuron pools, simulated in Brian2
\parencite{stimberg2019brian2} by exponential Euler integration with a \qty{0.1}{\ms} time step.
```

Why: the network is now continuous-time equations integrated by Brian2 (the 1 ms discrete
update is gone). Cite Stimberg, Brette & Goodman (2019), eLife 8:e47314.

## 2. Sign of the GABA term in the E equation (bug in the current text)

**Now:** `... - \left(\sum_{j} J^{IE}_{ij} s^I_j\right) (V_{GABA} - V) ...`

**Proposed:** `... + \left(\sum_{j} J^{IE}_{ij} s^I_j\right) (V_{GABA} - V_i) ...`

Why: with the driving force written as `(V_GABA - V)`, every conductance term enters with a plus
sign, as the AMPA and NMDA terms already do. The minus makes inhibition depolarising. (Also
`V` -> `V_i` throughout, for consistency.)

## 3. Units of the NMDA term (bug in the current text)

**Now:** `\sum_l A_{il} (s^{Aud}_l (V_{AMPA} - V_i) + s^{NMDA}_l g(V)\, (V_{NMDA} - V_i))` with
`g(V) = g^{NMDA} / (1 + ...)`.

Here `A_il` (nS) multiplies `g(V)` (nS), which gives nS². In the model the NMDA conductance of a
connection is `A_il` rescaled so the mean is `g^NMDA`. **Proposed:**

```latex
\sum_l A_{il} \left(s^{Aud}_l (V_{AMPA} - V_i)
      + \frac{g^{NMDA}}{\bar{A}} s^{NMDA}_l\, m(V_i)\, (V_{NMDA} - V_i)\right)
...
m(V) &= \frac{1}{1 + \frac{[\mathrm{Mg}^{2+}]_o}{3.57} e^{-0.062 V}}
```

with $\bar A$ the mean nonzero entry of $A$, and the table row for $g^{NMDA}$ unchanged ("mean
peak NMDA conductance per connection").

## 4. Noise

**Now:** "$\eta_{it}$ Gaussian white noise with standard deviation \qty{754}{\pA} for every
\qty{1}{\ms} timestep."

**Proposed:** "$\eta_i(t)$ Gaussian white noise current of intensity
$\sigma_\eta = \qty{754}{\pA\,\ms^{1/2}}$ (independent across neurons), i.e. a current whose
average over any \qty{1}{\ms} window has standard deviation \qty{754}{\pA}."

Why: a per-step SD is not a property of a continuous-time model; the intensity is, and it gives
the same membrane fluctuations at any time step. In the table: `\qty{754}{\pA\,\ms^{1/2}}`.

## 5. The STDP audio gate

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

Why: the box window was replaced by an equivalent trace, which is a differential equation like the
rest of the model (and identical to the box for isolated input spikes).

## 6. Training

**Now:** "... prefaced by a \qty{300}{\ms} burn-in period ..." -- unchanged (the code's 500 ms
burn-in includes the 200 ms of intro notes).

Add after "Training cycled through motifs in a fixed order.":
"Each rendition started the network from rest, carrying over only the learned weights $J^{IE}$."

## 7. Parameter tables

Values that change (`constants.py` at this commit). Everything else in Tables 2-3 is unchanged.

| Table row | Now | Proposed | Note |
|---|---|---|---|
| (new) Simulation step | -- | \qty{0.1}{\ms} (exponential Euler, Brian2) | |
| $\eta$ Noise (E, I) | \qty{754}{\pA}, 0 per 1 ms step | \qty{754}{\pA\,\ms^{1/2}}, 0 | same number, now an intensity |
| $A$ Aud $\to$ E mean (s.d.) | \qty{38.1}{\nano\siemens} (19.0) | \qty{44.8}{\nano\siemens} (22.3) | $\times 1.176$ |
| $B$ HVC $\to$ I mean (s.d.) | \qty{6.36}{\nano\siemens} (3.18) | \qty{7.48}{\nano\siemens} (3.74) | $\times 1.176$ |
| $J^{EI}$ E $\to$ I mean (s.d.) | \qty{0.187}{\nano\siemens} (0.0187) | \qty{0.220}{\nano\siemens} (0.0220) | $\times 1.176$ |
| $\beta$ Adaptation step | \qty{7540}{\pA} | \qty{10180}{\pA} | |
| $r_I$ STDP threshold rate | \qty{46}{\Hz} | \qty{36}{\Hz} | |

The factor 1.176 restores the charge per spike that the 1 ms discrete update delivered
($1/(1-e^{-1/3})$ ms vs \qty{3}{\ms} of peak AMPA conductance); see `docs/design-notes.md`,
"Brian2 port". Suggested sentence for *Model evaluation* (after the list of tunable parameters):

```latex
The model was first calibrated with a \qty{1}{\ms} discrete-time update and recalibrated for
continuous-time integration at \qty{0.1}{\ms}: AMPA weights were scaled by $1/[3(1-e^{-1/3})]
\approx 1.18$ to preserve the charge each spike delivers, and $\beta$ and $r_I$ re-tuned.
```

Also in the *Training and plasticity* table: $\rho$ = \qty{3.775}{\nano\siemens} is unchanged.

**Results that change** (Table 4 is targets only, so nothing to edit there, but any results text
quoting model numbers should use the README's "Expected numbers", regenerated for this model).
