"""Spiking sparse auditory encoder and vocal error network (VEN).

The public surface is deliberately split so that the model and training core import with
numpy and scipy alone:

- pure numpy/scipy (always available): :mod:`~spiking_ven.filterbank`,
  :mod:`~spiking_ven.cochleagram`, :mod:`~spiking_ven.olshausen_field`,
  :mod:`~spiking_ven.smith_lewicki`, :mod:`~spiking_ven.common`,
  :mod:`~spiking_ven.vocal_error_net`
- Brian2 runtime glue (requires the ``brian2`` extra): :mod:`~spiking_ven.brian2_runtime`

``brian2_runtime`` is intentionally NOT imported here, so ``import spiking_ven`` never pulls
in Brian2. Import it explicitly when you need live NeuronGroups.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
