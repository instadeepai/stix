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

"""DiT transformer block with AdaLN-Zero modulation."""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from stix.nn.layers.attention import MultiHeadSelfAttention
from stix.nn.layers.feed_forward import SwiGLUFeedForward


class TransformerBlock(nnx.Module):
    """DiT block with AdaLN-Zero modulation (Peebles & Xie, 2023).

    A single zero-initialised MLP produces all 6 modulation parameters
    (shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn)
    from the conditioning vector. This ensures all modulation starts at zero
    and co-evolves during training.

    Forward: modulate(norm(x)) → Attention → gate → residual
           → modulate(norm(x)) → FFN → gate → residual.
    """

    def __init__(
        self,
        embedding_dim: int,
        context_dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        *,
        dtype: jnp.dtype = jnp.float32,
        rngs: nnx.Rngs,
    ):
        """Initialise transformer block with shared AdaLN-Zero modulation.

        Args:
            embedding_dim: Token feature width.
            context_dim: Conditioning-vector width fed to the AdaLN modulation MLP.
            num_heads: Number of self-attention heads.
            ffn_hidden_dim: SwiGLU hidden width.
            dtype: Parameter/compute dtype.
            rngs: Random-number generators for parameter initialisation.
        """
        self.embedding_dim = embedding_dim

        # Plain LayerNorm (no learned scale/bias) for attention and FFN sub-blocks
        self.attention_norm = nnx.LayerNorm(
            embedding_dim, use_scale=False, use_bias=False, dtype=dtype, rngs=rngs
        )
        self.ffn_norm = nnx.LayerNorm(
            embedding_dim, use_scale=False, use_bias=False, dtype=dtype, rngs=rngs
        )

        self.attention = MultiHeadSelfAttention(
            embedding_dim, num_heads, dtype=dtype, rngs=rngs
        )
        self.ffn = SwiGLUFeedForward(
            embedding_dim, ffn_hidden_dim, embedding_dim, dtype=dtype, rngs=rngs
        )

        # Shared AdaLN-Zero modulation: SiLU → Linear → 6 * embedding_dim
        # Zero-init so all modulation (shift, scale, gate) starts at zero,
        # making each block an identity at initialisation.
        self.adaln_modulation = nnx.Linear(
            context_dim,
            6 * embedding_dim,
            kernel_init=nnx.initializers.zeros,
            bias_init=nnx.initializers.zeros,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self, x: Array, context: Array | None, attention_mask: Array | None = None
    ) -> Array:
        """Forward pass. x: (L, D), context: (L, C)/(C,)/None → (L, D).

        ``context=None`` degrades to a plain pre-norm transformer block (no
        AdaLN modulation): shift=0, scale=0, gate=1, i.e. ``x + attn(LN(x))``
        and ``x + ffn(LN(x))`` with no conditioning influence.
        """
        if context is None:
            # Defaults that recover a plain pre-norm transformer block.
            shift_attn = scale_attn = shift_ffn = scale_ffn = 0.0
            gate_attn = gate_ffn = 1.0
        else:
            modulation = self.adaln_modulation(jax.nn.silu(context))
            (
                shift_attn,
                scale_attn,
                gate_attn,
                shift_ffn,
                scale_ffn,
                gate_ffn,
            ) = jnp.split(modulation, 6, axis=-1)

        # Attention sub-block: modulate → attend → gate → residual
        h = (1.0 + scale_attn) * self.attention_norm(x) + shift_attn
        x = x + gate_attn * self.attention(h, attention_mask)

        # FFN sub-block: modulate → FFN → gate → residual
        h = (1.0 + scale_ffn) * self.ffn_norm(x) + shift_ffn
        x = x + gate_ffn * self.ffn(h)
        return x
