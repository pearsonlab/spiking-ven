# To do

## Singing should barely change the E rate (new target, unmet)

Mandelblat-Cerf et al. 2014, Fig. 7D: AIV rates change little between singing and not
singing (15/37 units significant, units along the diagonal of a log-log plot within ~2x),
and only 4/37 are modulated in time with the motif. The model instead cancels song by
inhibiting E throughout singing: ~2.5 Hz while singing vs ~10 Hz in silence without singing.
A shift test (HVC timing moved against the song) shows ~96% of the cancellation is not
time-locked: song drive of +14.4 Hz above baseline is cancelled to +0.05 Hz with HVC
aligned and to +0.5 Hz with HVC shifted 50-200 ms. With a singing ~ not-singing constraint,
only highly specific (time-locked) cancellation should be needed, since the auditory input
then modulates E strongly relative to its baseline. Target `singing_over_not_singing`
(0.5-2.0) is being added to `targets.py` / `metrics.rendition_rates` (WIP commit on
`brian2-contrast`), with an xfail reproduction test until the model meets it.

## Revisit HVC -> I sparsity for song-specific cancellation

Single I cells are time-modulated (median CV 1.1 over the motif in 20 ms bins; peak ~3.8x
mean) but the population is flat (CV 0.10), and each E cell sums many learned I inputs, so
its inhibition is nearly flat in time. Revisit the sparsity of HVC -> I connections (and the
sparsity of the learned I -> E weights) so that I activity, and the inhibition each E cell
learns, is time-specific -- together with the target above, which removes the blanket
inhibition the current solution relies on. Earlier sweeps (`docs/contrast-experiments.md`,
C2-C4) varied `C_HVC` alone, under the old targets, and found no robust gain.

## Blunted, delayed response to noise outside singing (Fig. 8)

Mandelblat-Cerf et al. 2014, Fig. 8: outside singing, AIV does not respond to noise bursts
during playback of the bird's own song (0/53 units), and responds to isolated noise bursts
in only 12/53 units, slowly and with long latency (only one within 150 ms of onset); during
singing every noise-responsive unit responds within 150 ms (latency 23 +/- 12 ms). The model
has no state dependence besides HVC drive and responds strongly to song heard while not
singing (~20-24 Hz). Reproducing the blunted, delayed response outside singing may need a
singing-state signal -- possibly neuromodulatory -- that changes AIV's auditory gain or
dynamics, not only HVC-timed inhibition.
