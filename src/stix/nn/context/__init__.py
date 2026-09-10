# Copyright 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context-vector construction for :class:`stix.nn.EncoderBackboneDecoderNetwork`.

A context module owns the strategy for turning ``(t, context_data,
context_mask)`` into the single context vector consumed by the backbone
and decoders.

- ``SumContextEncoder``: default summed-context strategy.
- ``TimeNoiseContextEncoder``: Fourier-feature time/noise context encoder,
  typically slotted into ``SumContextEncoder`` as the time encoder.
"""

from stix.nn.context.sum_context_encoder import SumContextEncoder
from stix.nn.context.time_noise_context_encoder import (
    TimeNoiseContextEncoder,
)

__all__ = [
    "SumContextEncoder",
    "TimeNoiseContextEncoder",
]
