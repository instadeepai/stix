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

"""DiT-style Adaptive Layer Norm (AdaLN)."""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


class AdaptiveLayerNorm(nnx.Module):
    """DiT-style Adaptive Layer Norm (AdaLN).

    Computes: (1 + scale_proj(norm_c(c))) * norm_x(x) + offset_proj(norm_c(c))

    Both norm_x and norm_c are LayerNorm without learned scale/bias.
    When zero_init=True, scale and offset projections are zero-initialised
    so the block initially passes through norm_x(x) unchanged.
    """

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        *,
        zero_init: bool = False,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        """Initialise AdaLN with optional zero-init for scale/offset projections.

        Args:
            input_dim: Feature width of ``x`` being normalised.
            context_dim: Feature width of the conditioning vector ``context``.
            zero_init: If True, zero-initialise the scale/offset projections so
                the module starts as a pass-through LayerNorm on ``x``.
            dtype: Parameter/compute dtype.
            rngs: Random-number generators for parameter initialisation.
        """
        self.norm_x = nnx.LayerNorm(
            input_dim, use_scale=False, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.norm_c = nnx.LayerNorm(
            context_dim, use_scale=False, use_bias=False, dtype=dtype, rngs=rngs
        )

        kernel_init = (
            nnx.initializers.zeros if zero_init else nnx.initializers.lecun_normal()
        )
        self.scale_projection = nnx.Linear(
            context_dim, input_dim, kernel_init=kernel_init, dtype=dtype, rngs=rngs
        )
        self.offset_projection = nnx.Linear(
            context_dim, input_dim, kernel_init=kernel_init, dtype=dtype, rngs=rngs
        )

    def __call__(self, x: Array, context: Array | None) -> Array:
        """Apply AdaLN. x: (L, D), context: (L, C)/(C,)/None → (L, D).

        ``context=None`` degrades to plain LayerNorm (no scale/shift).
        """
        if context is None:
            return self.norm_x(x)
        normed_context = self.norm_c(context)
        scale = self.scale_projection(normed_context)
        offset = self.offset_projection(normed_context)
        return (1.0 + scale) * self.norm_x(x) + offset
