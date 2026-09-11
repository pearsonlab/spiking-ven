"""Loading the song corpus and choosing the template rendition.

The "template" is the highest-RMS rendition. That single rule decides which motif the
network trains on, which one the metrics score, and which one the figure draws -- so it
lives here once. It used to be re-derived in four places, and if those had ever diverged
the figure and the metrics would have silently described a different motif than training
used.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["Corpus", "load_corpus", "highest_rms_index"]


def highest_rms_index(signals) -> int:
    """Index of the loudest (highest-RMS) signal in an iterable of 1-D arrays.

    Takes raw arrays rather than a Corpus so the preprocessing stage, which runs before
    any ``motifs.npz`` exists, can apply the identical rule.
    """
    rms = [float(np.sqrt(np.mean(np.asarray(s, dtype=np.float64) ** 2))) for s in signals]
    if not rms:
        raise ValueError("no signals given")
    return int(np.argmax(rms))


@dataclass(frozen=True)
class Corpus:
    """A loaded ``motifs.npz``: DTW-aligned renditions plus their syllable timings."""

    audio: np.ndarray          # (n_motifs, n_samples), zero-padded to the template length
    lengths: np.ndarray        # (n_motifs,) valid samples per rendition
    song_Ts: np.ndarray        # (n_motifs,) rendition duration in ms
    sr: int

    @property
    def n_motifs(self) -> int:
        return int(self.audio.shape[0])

    @property
    def T_song(self) -> int:
        """Motif window in ms, rounded up -- the duration every stage works in."""
        return int(np.ceil(self.song_Ts.max()))

    def signal(self, i: int) -> np.ndarray:
        """Rendition ``i`` as float64, trimmed to its valid length."""
        return self.audio[i, : self.lengths[i]].astype(np.float64)

    def signals(self):
        """Every rendition, trimmed."""
        return (self.signal(i) for i in range(self.n_motifs))

    @property
    def template_index(self) -> int:
        """Index of the template (highest-RMS) rendition."""
        return highest_rms_index(self.signals())

    def template(self) -> tuple[np.ndarray, int]:
        """``(signal, index)`` for the template rendition."""
        i = self.template_index
        return self.signal(i), i


def load_corpus(path: str | Path) -> Corpus:
    """Load a ``motifs.npz`` written by ``sven-prep-data``."""
    d = np.load(str(path))
    return Corpus(audio=d["audio"], lengths=d["lengths"], song_Ts=d["song_Ts"],
                  sr=int(d["sr"]))
