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

"""MLP encoder, decoder, and backbone for the EncoderBackboneDecoderNetwork.

Simple linear encoder/decoder with a pre-norm residual MLP backbone
that fuses per-modality embeddings via concatenation and time conditioning.
"""

import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree
from pydantic import BaseModel

from stix.nn.fusion import fuse, unfuse
from stix.typing import Mask, Time, Var


class MLPEncoder(nnx.Module):
    """Encode continuous input into an embedding space via a linear layer."""

    class Config(BaseModel):
        """Configuration for MLPEncoder."""

        input_dim: int
        embedding_dim: int

    def __init__(self, cfg: Config, rngs: nnx.Rngs) -> None:
        """Initialise the encoder."""
        self.encoding_layer = nnx.Linear(cfg.input_dim, cfg.embedding_dim, rngs=rngs)

    def __call__(self, x: Var, t: Time | None = None) -> Var:
        """Forward pass. x: (..., input_dim) -> (..., embedding_dim).

        Time t is accepted for interface compatibility but ignored.
        """
        return self.encoding_layer(x)


class MLPDecoder(nnx.Module):
    """Decode an embedding back to the original space via a linear layer."""

    class Config(BaseModel):
        """Configuration for MLPDecoder."""

        embedding_dim: int
        output_dim: int

    def __init__(self, cfg: Config, rngs: nnx.Rngs) -> None:
        """Initialise the decoder."""
        self.decoding_layer = nnx.Linear(cfg.embedding_dim, cfg.output_dim, rngs=rngs)

    def __call__(self, x: Var, context: Var | None = None) -> Var:
        """Forward pass. Context is accepted for interface compatibility but ignored."""
        return self.decoding_layer(x)


class MLPBackbone(nnx.Module):
    """MLP backbone that fuses per-modality embeddings with a time embedding.

    Concatenates all modality embeddings along the feature axis, applies
    pre-norm residual blocks with time conditioning (SiLU activation),
    then splits back to per-modality outputs.

    Context and attention_mask are accepted for interface compatibility
    but ignored: the MLP backbone does its own time conditioning.
    """

    class Config(BaseModel):
        """Configuration for MLPBackbone."""

        embedding_dim: int
        backbone_depth: int
        time_embedding_dim: int

    def __init__(self, cfg: Config, rngs: nnx.Rngs) -> None:
        """Initialise the backbone."""
        self.time_embedding = nnx.Linear(1, cfg.time_embedding_dim, rngs=rngs)
        self.linear_layers = nnx.List(
            [
                nnx.Linear(
                    cfg.embedding_dim + cfg.time_embedding_dim,
                    cfg.embedding_dim,
                    rngs=rngs,
                )
                for _ in range(cfg.backbone_depth)
            ]
        )
        self.layers_norm = nnx.List(
            [
                nnx.LayerNorm(cfg.embedding_dim, rngs=rngs)
                for _ in range(cfg.backbone_depth)
            ]
        )

    def __call__(
        self,
        encoded_modalities: PyTree[Var],
        t: Time,
        context: Var | None = None,
        attention_mask: PyTree[Mask | None] | None = None,
    ) -> PyTree[Var]:
        """Forward pass: fuse embeddings with time and apply residual MLP layers.

        Args:
            encoded_modalities: Per-modality encoded inputs to fuse along the
                feature axis.
            t: Time scalar used for the backbone's own time conditioning.
            context: Ignored; accepted only for interface compatibility with
                the ``Network`` contract.
            attention_mask: Ignored; accepted only for interface compatibility.

        Returns:
            Per-modality outputs (same structure as ``encoded_modalities``).
        """
        x, treedef, widths = fuse(encoded_modalities, axis=-1)

        t_emb = nnx.silu(self.time_embedding(jnp.atleast_1d(t)))
        t_emb = jnp.broadcast_to(t_emb, x.shape[:-1] + t_emb.shape[-1:])

        for norm, layer in zip(self.layers_norm, self.linear_layers):
            residual = x
            x = norm(x)
            x = nnx.silu(layer(jnp.concatenate((x, t_emb), axis=-1)))
            x = x + residual

        return unfuse(x, treedef, widths, axis=-1)
