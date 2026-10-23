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

The public surface is deliberately split so that the model and training core import with
numpy and scipy alone:

- pure numpy/scipy (always available): everything re-exported below
- Brian2 runtime glue (requires the ``brian2`` extra): :mod:`spiking_ven.brian2_runtime`

``brian2_runtime`` is intentionally NOT imported here, so ``import spiking_ven`` never
pulls in Brian2. Import it explicitly when you need live NeuronGroups.
"""

__version__ = "0.6.0"

from .causal_filters import CausalFilterEncoder
from .cochleagram import cochleagram, load_wav
from .common import acts_to_spikes, generate_hvc_spikes, spike_to_rate
from .filterbank import centre_frequencies, gammatone_spectrogram
from .vocal_error_net import VocalErrorNet, VocalErrorNetV2

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
    "VocalErrorNetV2",
    "generate_hvc_spikes",
    "spike_to_rate",
]
