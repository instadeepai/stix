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

"""Neural network components for stochastic interpolants.

Public API:
- ``Network``: abstract base defining the network call contract enforced by
  ``GenerativeModel``.
- ``EncoderBackboneDecoderNetwork``: default network, a thin wrapper around
  instantiated encoders, backbone, and decoders.
- ``NetworkDimsConfig``: shared dimensions (embedding/context/ffn widths + dtype)
  injected into each DiT component so they cannot drift out of sync.
- ``SumContextEncoder``: default context-building strategy (sum per-source contributions).
- ``TimeNoiseContextEncoder``: Fourier-feature time/noise context encoder.
- ``MultiHeadDecoder``: bundle multiple decoder heads for two-sided models.
- ``DiTEncoder``, ``DiTDecoder``, ``DiTBackbone``: DiT (transformer) components.
  Each takes a ``NetworkDimsConfig`` plus its own arguments directly (no
  per-component config object).
- ``MLPEncoder``, ``MLPDecoder``, ``MLPBackbone``: minimal linear/MLP components,
  a lightweight alternative to the DiT stack.
- ``fuse``, ``unfuse``, ``fuse_mask``: concatenate a per-modality pytree into one
  stream and split it back, for writing a backbone that mixes the modalities.

Building blocks live under ``stix.nn.layers``; the components are split across
``stix.nn.encoders``, ``stix.nn.decoders``, ``stix.nn.backbone``, and the
context encoders under ``stix.nn.context``.
"""

from stix.nn.backbone import DiTBackbone
from stix.nn.context import (
    SumContextEncoder,
    TimeNoiseContextEncoder,
)
from stix.nn.decoders import DiTDecoder
from stix.nn.decoders.multi_head_decoder import MultiHeadDecoder
from stix.nn.encoder_backbone_decoder_network import EncoderBackboneDecoderNetwork
from stix.nn.encoders import DiTEncoder
from stix.nn.fusion import fuse, fuse_mask, unfuse
from stix.nn.mlp import MLPBackbone, MLPDecoder, MLPEncoder
from stix.nn.network import Network
from stix.nn.network_dims import NetworkDimsConfig

__all__ = [
    "DiTBackbone",
    "DiTDecoder",
    "DiTEncoder",
    "EncoderBackboneDecoderNetwork",
    "MLPBackbone",
    "MLPDecoder",
    "MLPEncoder",
    "MultiHeadDecoder",
    "Network",
    "NetworkDimsConfig",
    "SumContextEncoder",
    "TimeNoiseContextEncoder",
    "fuse",
    "fuse_mask",
    "unfuse",
]
