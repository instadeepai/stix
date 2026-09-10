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

"""Reusable DiT building blocks (layers).

Low-level modules composed by the DiT encoder, decoder, and backbone:

- ``SinusoidalFourierFeatures``: learned Fourier features for scalar inputs.
- ``AdaptiveLayerNorm``: DiT-style AdaLN modulation.
- ``SwiGLUFeedForward``: SwiGLU feed-forward network.
- ``MultiHeadSelfAttention``: standard multi-head self-attention.
- ``TransformerBlock``: DiT block with AdaLN-Zero modulation.
- ``RegressionHead``: decoder output head (AdaLN + SwiGLU FFN).
"""

from stix.nn.layers.adaptive_layer_norm import AdaptiveLayerNorm
from stix.nn.layers.attention import MultiHeadSelfAttention
from stix.nn.layers.feed_forward import SwiGLUFeedForward
from stix.nn.layers.fourier_features import SinusoidalFourierFeatures
from stix.nn.layers.regression_head import RegressionHead
from stix.nn.layers.transformer_block import TransformerBlock

__all__ = [
    "AdaptiveLayerNorm",
    "MultiHeadSelfAttention",
    "RegressionHead",
    "SinusoidalFourierFeatures",
    "SwiGLUFeedForward",
    "TransformerBlock",
]
