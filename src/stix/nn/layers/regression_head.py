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

"""Decoder regression head: AdaLN + SwiGLU FFN."""

import jax.numpy as jnp
from flax import nnx

from stix.nn.layers.adaptive_layer_norm import AdaptiveLayerNorm
from stix.nn.layers.feed_forward import SwiGLUFeedForward
from stix.typing import Var


class RegressionHead(nnx.Module):
    """Decoder output head: AdaLN + SwiGLU FFN with zero-init final layer."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        output_dim: int,
        ffn_hidden_dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        """Initialise regression head with zero-init AdaLN and SwiGLU FFN.

        Args:
            input_dim: Backbone embedding width (AdaLN input).
            context_dim: Conditioning-vector width.
            output_dim: Raw output channel dimension for this modality.
            ffn_hidden_dim: SwiGLU hidden width.
            dtype: Parameter/compute dtype.
            rngs: Random-number generators for parameter initialisation.
        """
        self.adaptive_norm = AdaptiveLayerNorm(
            input_dim, context_dim, zero_init=True, dtype=dtype, rngs=rngs
        )
        self.ffn = SwiGLUFeedForward(
            input_dim,
            ffn_hidden_dim,
            output_dim,
            zero_init_output=True,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: Var, context: Var | None) -> Var:
        """Forward pass. x: (L, D), context: (L, C)/(C,)/None → (L, output_dim)."""
        return self.ffn(self.adaptive_norm(x, context))
