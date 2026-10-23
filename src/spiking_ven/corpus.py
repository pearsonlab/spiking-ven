"""Loading the song corpus and choosing the template rendition.

The "template" is the rendition every other one is DTW-aligned to: the highest-RMS raw
rendition, chosen by the preprocessing stage. It decides which motif the network trains
on, which one the metrics score, and which one the figure draws -- so it lives here once.
A ``motifs.npz`` that records ``template_index`` supplies it directly; one that does not
falls back to the highest-RMS aligned rendition.
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
    # DTW reference recorded by the preprocessing stage; None means "compute from audio".
    stored_template_index: int | None = None

    def __post_init__(self):
        i = self.stored_template_index
        if i is not None and not 0 <= i < self.n_motifs:
            raise ValueError(
                f"stored template_index {i} out of range for {self.n_motifs} motifs")

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
        """Index of the template rendition.

        The stored DTW reference when the corpus records one; otherwise the highest-RMS
        aligned rendition.
        """
        if self.stored_template_index is not None:
            return self.stored_template_index
        return highest_rms_index(self.signals())

    def template(self) -> tuple[np.ndarray, int]:
        """``(signal, index)`` for the template rendition."""
        i = self.template_index
        return self.signal(i), i


def load_corpus(path: str | Path) -> Corpus:
    """Load a ``motifs.npz`` written by ``sven-prep-data``.

    Reads the stored ``template_index`` when the file has one.
    """
    d = np.load(str(path))
    tpl = int(d["template_index"]) if "template_index" in d.files else None
    return Corpus(audio=d["audio"], lengths=d["lengths"], song_Ts=d["song_Ts"],
                  sr=int(d["sr"]), stored_template_index=tpl)
