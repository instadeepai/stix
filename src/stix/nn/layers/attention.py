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

"""Multi-head self-attention."""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array


class MultiHeadSelfAttention(nnx.Module):
    """Standard multi-head self-attention.

    No bias on Q/K projections, bias on V/output projections (standard DiT).
    Uses jax.nn.dot_product_attention for the attention computation.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        """Initialise Q/K/V/output projections for multi-head self-attention.

        Args:
            embedding_dim: Total embedding width, split evenly across heads.
            num_heads: Number of attention heads (must divide ``embedding_dim``).
            dtype: Parameter/compute dtype.
            rngs: Random-number generators for parameter initialisation.

        Raises:
            ValueError: If ``embedding_dim`` is not divisible by ``num_heads``.
        """
        if embedding_dim % num_heads != 0:
            msg = f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            raise ValueError(msg)

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.q_projection = nnx.Linear(
            embedding_dim, embedding_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.k_projection = nnx.Linear(
            embedding_dim, embedding_dim, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.v_projection = nnx.Linear(
            embedding_dim, embedding_dim, dtype=dtype, rngs=rngs
        )
        self.output_projection = nnx.Linear(
            embedding_dim, embedding_dim, dtype=dtype, rngs=rngs
        )

    def __call__(self, x: Array, attention_mask: Array | None = None) -> Array:
        """Self-attention. x: (L, D), mask: (L, L) | None → (L, D)."""
        seq_len = x.shape[0]

        # Project and reshape: (L, D) → (L, num_heads, head_dim)
        q = self.q_projection(x).reshape(seq_len, self.num_heads, self.head_dim)
        k = self.k_projection(x).reshape(seq_len, self.num_heads, self.head_dim)
        v = self.v_projection(x).reshape(seq_len, self.num_heads, self.head_dim)

        # jax.nn.dot_product_attention expects mask broadcastable to
        # (num_heads, q_length, kv_length). Expand (L, L) → (1, L, L).
        if attention_mask is not None:
            attention_mask = attention_mask[None, :, :]  # (1, L, L)

        out = jax.nn.dot_product_attention(q, k, v, mask=attention_mask)
        # (L, num_heads, head_dim) → (L, D)
        out = out.reshape(seq_len, -1)
        return self.output_projection(out)
