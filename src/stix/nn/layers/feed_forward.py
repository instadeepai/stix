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

"""SwiGLU feed-forward network."""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


class SwiGLUFeedForward(nnx.Module):
    """SwiGLU feed-forward network.

    Computes: output_proj(silu(gate_proj(x)) * up_proj(x))
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        zero_init_output: bool = False,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        """Initialise SwiGLU with gate, up, and output projections.

        Args:
            input_dim: Input feature width.
            hidden_dim: Width of the gate/up projections.
            output_dim: Output feature width.
            zero_init_output: If True, zero-initialise the output projection so
                the block starts as an identity contribution to a residual stream.
            dtype: Parameter/compute dtype.
            rngs: Random-number generators for parameter initialisation.
        """
        self.gate_projection = nnx.Linear(input_dim, hidden_dim, dtype=dtype, rngs=rngs)
        self.up_projection = nnx.Linear(input_dim, hidden_dim, dtype=dtype, rngs=rngs)

        output_kernel_init = (
            nnx.initializers.zeros
            if zero_init_output
            else nnx.initializers.lecun_normal()
        )
        self.output_projection = nnx.Linear(
            hidden_dim,
            output_dim,
            kernel_init=output_kernel_init,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(self, x: Array) -> Array:
        """Forward pass. x: (L, D) → (L, D_out)."""
        return self.output_projection(
            jax.nn.silu(self.gate_projection(x)) * self.up_projection(x)
        )
