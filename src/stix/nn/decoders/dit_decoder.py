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

"""Per-modality DiT decoder.

Wraps a :class:`~stix.nn.layers.RegressionHead` (AdaLN + SwiGLU FFN with a
zero-initialised final layer) to project backbone outputs back to data space.
"""

from flax import nnx

from stix.nn.layers.regression_head import RegressionHead
from stix.nn.network_dims import NetworkDimsConfig
from stix.typing import Var


class DiTDecoder(nnx.Module):
    """Per-modality decoder: RegressionHead with context-dependent AdaLN.

    Wraps RegressionHead behind the ``decoder(x, context=...)`` call signature
    that :class:`~stix.nn.EncoderBackboneDecoderNetwork`
    expects from every per-modality decoder.
    Optionally squeezes the sequence dimension for single-token modalities.
    """

    def __init__(
        self,
        network_dims: NetworkDimsConfig,
        *,
        output_dim: int,
        squeeze_sequence: bool = True,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise regression head decoder.

        Args:
            network_dims: Shared network dimensions (uses ``embedding_dim``,
                ``context_dim``, ``ffn_hidden_dim`` and ``dtype``).
            output_dim: Raw output channel dimension for this modality.
            squeeze_sequence: Squeeze the token axis for single-token modalities.
            rngs: Random-number generators for parameter initialisation.
        """
        self.regression_head = RegressionHead(
            network_dims.embedding_dim,
            network_dims.context_dim,
            output_dim,
            network_dims.ffn_hidden_dim,
            dtype=network_dims.dtype,
            rngs=rngs,
        )
        self.squeeze_sequence = squeeze_sequence

    def __call__(self, x: Var, context: Var | None) -> Var:
        """Decode with context-dependent AdaLN.

        Args:
            x: Backbone output for this modality, shape ``(L, embedding_dim)``.
            context: Conditioning vector, or ``None`` to fall back to plain
                AdaLN-less normalisation inside the regression head.

        Returns:
            Decoded output, shape ``(output_dim,)`` if ``squeeze_sequence`` else
            ``(L, output_dim)``.
        """
        out = self.regression_head(x, context)
        if self.squeeze_sequence:
            out = out.squeeze(axis=0)
        return out
