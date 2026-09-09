# Third-party notices

This package vendors (copies, with modifications) third-party source code. The notices below
are reproduced to satisfy the terms of the applicable licenses.

---

## ERB / gammatone filterbank — `src/spiking_ven/filterbank.py`

`src/spiking_ven/filterbank.py` contains code derived from the **gammatone** Python package by
Jason Heeris (<https://github.com/detly/gammatone>), which is itself a port of MATLAB
implementations by Malcolm Slaney and Dan Ellis. Upstream was archived on 2024-09-10 and is no
longer maintained; the code is vendored here rather than taken as a runtime dependency so that
this package installs from numpy and scipy alone.

**Functions derived from upstream:** `centre_freqs`, `make_erb_filters`, `erb_filterbank`,
`gtgram` (and the ERB-scale helpers they rely on).

**Modifications made:** repackaged under the `spiking_ven` namespace; module renamed from
`gammatone` to `filterbank` to avoid shadowing the upstream distribution name; unused entry
points and the upstream CLI/plotting helpers removed; type hints and docstrings adjusted to
this project's conventions.

Neither the names of the copyright holders nor the names of their contributors are used to
endorse or promote this package.

Upstream license, reproduced verbatim from the project's `COPYING` file:

```
Copyright (c) 1998, Malcolm Slaney <malcolm@interval.com>
Copyright (c) 2009, Dan Ellis <dpwe@ee.columbia.edu>
Copyright (c) 2014, Jason Heeris <jason.heeris@gmail.com>
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
