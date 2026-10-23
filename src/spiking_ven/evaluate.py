"""Evaluation of a trained vocal error network, in one import.

The code lives in three modules, re-exported here:

- :mod:`spiking_ven.stimuli` -- building the song, reversed-song and DAF stimuli, from
  waveform to spike trains, and reading back the settings a model was trained with.
- :mod:`spiking_ven.metrics` -- the numbers computed from the network's response: mean E
  rate per rendition, and the DAF responder analysis of Mandelblat-Cerf et al. 2014.
- :mod:`spiking_ven.targets` -- the literature values those numbers are compared with.
"""

from __future__ import annotations

from .metrics import (
    daf_responders,
    daf_response,
    daf_trials,
    format_rendition_rates,
    format_responders,
    rendition_rates,
)
from .stimuli import (
    DAF_WN_AMPLITUDE,
    build_stimuli,
    daf_waveform,
    encode_acts,
    encode_stimulus,
    resolve_stimulus_settings,
    stimulus_record,
)
from .targets import BIOLOGICAL_TARGETS, PEAK_BIN_MS

__all__ = [
    "daf_waveform",
    "encode_acts",
    "encode_stimulus",
    "build_stimuli",
    "stimulus_record",
    "resolve_stimulus_settings",
    "daf_trials",
    "daf_responders",
    "daf_response",
    "format_responders",
    "rendition_rates",
    "format_rendition_rates",
    "BIOLOGICAL_TARGETS",
    "PEAK_BIN_MS",
    "DAF_WN_AMPLITUDE",
]
