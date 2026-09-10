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

"""DiT transformer backbone with AdaLN-Zero conditioning."""

import jax
from flax import nnx
from jaxtyping import PyTree

from stix.nn.fusion import fuse, fuse_mask, unfuse
from stix.nn.layers.transformer_block import TransformerBlock
from stix.nn.network_dims import NetworkDimsConfig
from stix.typing import Mask, Time, Var


class DiTBackbone(nnx.Module):
    """Transformer backbone with AdaLN-Zero conditioning.

    Concatenates per-modality encoded inputs along the sequence axis, runs through
    :class:`~stix.nn.layers.TransformerBlock` layers conditioned on a context
    vector, then splits back to per-modality outputs.

    The modalities are fused, so this backbone works on their flattened leaves
    rather than leaf by leaf, and hands back the structure it was given.
    """

    def __init__(
        self,
        network_dims: NetworkDimsConfig,
        *,
        modality_num_tokens: PyTree[int],
        num_heads: int = 4,
        num_layers: int = 4,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialise transformer blocks.

        Args:
            network_dims: Shared network dimensions (uses ``embedding_dim``,
                ``context_dim``, ``ffn_hidden_dim`` and ``dtype``).
            modality_num_tokens: Per-modality token count along the sequence
                axis, shaped like the modalities.
            num_heads: Number of attention heads per transformer block.
            num_layers: Number of transformer blocks.
            rngs: Random-number generators for parameter initialisation.
        """
        self.modality_num_tokens = modality_num_tokens

        # Transformer backbone
        self.transformer_blocks = nnx.List(
            [
                TransformerBlock(
                    network_dims.embedding_dim,
                    network_dims.context_dim,
                    num_heads,
                    network_dims.ffn_hidden_dim,
                    dtype=network_dims.dtype,
                    rngs=rngs,
                )
                for _ in range(num_layers)
            ]
        )

    def __call__(
        self,
        encoded_modalities: PyTree[Var],
        t: Time,
        context: Var | None,
        attention_mask: PyTree[Mask | None] | None = None,
    ) -> PyTree[Var]:
        """Forward pass through transformer backbone.

        Args:
            encoded_modalities: Per-modality encoded inputs (L_k, embedding_dim).
            t: Time value in [0, 1] (unused; conditioning is via context).
            context: Conditioning vector from the context encoder; ``None``
                degrades each ``TransformerBlock`` to plain pre-norm
                (no AdaLN modulation).
            attention_mask: Optional per-modality boolean masks. A whole ``None``
                attends over everything; a ``None`` in place of one modality's
                mask leaves that modality fully attended to.

        Returns:
            Per-modality embeddings (same structure that was passed in).

        Raises:
            ValueError: If the encoded token counts disagree with
                ``modality_num_tokens``.
        """
        # (L_total, embedding_dim)
        h, treedef, lengths = fuse(encoded_modalities)

        declared = jax.tree.leaves(self.modality_num_tokens)
        if lengths != declared:
            msg = (
                f"encoded token counts {lengths} do not match the declared "
                f"modality_num_tokens {declared}"
            )
            raise ValueError(msg)

        combined_mask = fuse_mask(attention_mask, treedef, lengths)  # (L_total,)
        backbone_mask = (
            None
            if combined_mask is None
            else combined_mask[:, None] & combined_mask[None, :]  # (L_total, L_total)
        )

        for block in self.transformer_blocks:
            h = block(h, context, backbone_mask)

        return unfuse(h, treedef, lengths)
