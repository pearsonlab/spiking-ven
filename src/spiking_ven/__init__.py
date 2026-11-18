"""Spiking sparse auditory encoder and vocal error network (VEN).

Song audio is turned into a sparse spike code by a gammatone cochleagram feeding a
sparse-coding encoder, and that spike code drives an excitatory/inhibitory network that
learns to cancel the predictable auditory response to the bird's own song -- leaving an
error signal on novel or perturbed sound.

Pipeline
--------
``audio -> filterbank -> cochleagram -> sparse encoder -> spikes -> VocalErrorNet``

The sparse encoder is :class:`CausalFilterEncoder`, a strictly causal minimum-phase
filter bank: minimum phase puts each filter's energy as early as its magnitude response
allows, which is what keeps the modelled DAF response latency inside the bird's
23 +/- 12 ms.

The network's dynamics are Brian2 equations (:mod:`spiking_ven.network`). Brian2 is
imported lazily, on the first simulation, so ``import spiking_ven`` stays fast and the
encoder stages never pull it in. To host the network inside a larger Brian2 circuit,
use :meth:`VocalErrorNet.build`.
"""

__version__ = "0.7.0"

from .causal_filters import CausalFilterEncoder
from .cochleagram import cochleagram, load_wav
from .common import acts_to_spikes, generate_hvc_bursts, spike_to_rate
from .filterbank import centre_frequencies, gammatone_spectrogram
from .vocal_error_net import VocalErrorNet

__all__ = [
    "__version__",
    # stage 0: audio -> time-frequency
    "centre_frequencies",
    "gammatone_spectrogram",
    "cochleagram",
    "load_wav",
    # stage 1: sparse encoder
    "CausalFilterEncoder",
    "acts_to_spikes",
    # stage 2: vocal error network
    "VocalErrorNet",
    "generate_hvc_bursts",
    "spike_to_rate",
]
