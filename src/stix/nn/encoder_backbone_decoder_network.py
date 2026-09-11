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

"""Configurable encoder -> backbone -> decoder network.

The ``EncoderBackboneDecoderNetwork`` class is a thin wrapper that orchestrates
data flow between already-instantiated components:

- **Context encoder** (optional): produces the single context vector that
  conditions the backbone and decoders. ``EncoderBackboneDecoderNetwork`` is
  agnostic to how the context is built: it just calls ``context_encoder(t,
  context_data, context_mask)`` and forwards the result. See
  :class:`stix.nn.SumContextEncoder` for the default summed-context
  strategy and its masking semantics.
- **Encoders**: per-modality modules that project raw inputs into embedding space.
- **Backbone**: shared module that fuses per-modality embeddings (with time
  and optional context conditioning) and produces per-modality outputs.
- **Decoders**: per-modality modules that project backbone outputs back to
  data space (with optional context conditioning). A decoder's return type
  is opaque to the network: return a ``Var`` for one-sided, a ``tuple[Var, Var]``
  for two-sided (e.g. via ``MultiHeadDecoder``), or anything else downstream
  consumers expect.

Each component defines its own call signature; the network orchestrates the data
flow. Encoders and decoders are held one per pytree leaf, so an ``nnx.Module``
keeps its parameters tracked.
"""

import jax
from flax import nnx
from jaxtyping import PyTree

from stix.nn.network import Network
from stix.typing import Mask, Time, Var


def _is_module_leaf(leaf: object) -> bool:
    """Stop pytree traversal at nnx.Module boundaries."""
    return isinstance(leaf, nnx.Module)


class EncoderBackboneDecoderNetwork(Network):
    """Wraps instantiated encoders, backbone, and decoders.

    Data flow:
    1. ``context_encoder`` (if any) builds the context vector from ``t``, ``context_data``, and ``context_mask``.
    2. Per-modality encoders project inputs into embedding space.
    3. Backbone fuses embeddings with time/context, returns per-modality outputs.
    4. Per-modality decoders project outputs back to data space with context.

    The encoders, decoders and ``z_t`` share one per-modality pytree structure.
    """

    def __init__(
        self,
        encoders: PyTree[nnx.Module],
        backbone: nnx.Module,
        decoders: PyTree[nnx.Module],
        context_encoder: nnx.Module | None = None,
    ) -> None:
        """Initialise the network with already-instantiated components.

        Args:
            encoders: Per-modality input encoders. Must share ``decoders``'
                pytree structure.
            backbone: Shared module fusing per-modality embeddings.
            decoders: Per-modality output decoders.
            context_encoder: Module that builds the context vector from
                ``(t, context_data, context_mask)``. ``None`` means no
                context is built; the backbone and decoders receive
                ``context=None``. See :class:`stix.nn.SumContextEncoder`
                for the default strategy.

        Raises:
            ValueError: If ``encoders`` and ``decoders`` do not share one
                per-modality structure.
        """
        encoders_structure = jax.tree.structure(encoders, is_leaf=_is_module_leaf)
        decoders_structure = jax.tree.structure(decoders, is_leaf=_is_module_leaf)
        if encoders_structure != decoders_structure:
            msg = (
                "encoders and decoders must share pytree structure; got "
                f"encoders={encoders_structure} decoders={decoders_structure}"
            )
            raise ValueError(msg)

        self.context_encoder = context_encoder
        self.backbone = backbone
        # Marked as pytree data so nnx tracks each module's params while the
        # container stays a plain pytree to jax.tree.map over. The map rebuilds
        # the container, so later edits to the caller's own container do not
        # reach the network.
        self.encoders = nnx.data(
            jax.tree.map(lambda m: m, encoders, is_leaf=_is_module_leaf)
        )
        self.decoders = nnx.data(
            jax.tree.map(lambda m: m, decoders, is_leaf=_is_module_leaf)
        )

    def __call__(
        self,
        z_t: PyTree[Var],
        t: Time,
        context_data: PyTree[Var] | None = None,
        context_mask: PyTree[Mask | None] | None = None,
        attention_mask: PyTree[Mask | None] | None = None,
    ) -> PyTree[Var | tuple[Var, Var]]:
        """Forward pass: build context -> encode -> backbone -> decode.

        Args:
            z_t: Per-modality noisy state in embedded space.
            t: Time scalar.
            context_data: Context conditioning (a pytree) forwarded to
                ``context_encoder`` unchanged. Its structure is the context
                encoder's contract.
            context_mask: Context masks forwarded to ``context_encoder``
                unchanged.
            attention_mask: Forwarded to the backbone unchanged; semantics are
                backbone-specific (typically sequence-padding masks).

        Returns:
            Per-modality network output (same structure as ``z_t``); a single
            ``Var`` for one-sided decoders, or a ``tuple[Var, Var]`` for
            two-sided decoders (e.g. via ``MultiHeadDecoder``).
        """
        context = (
            self.context_encoder(t, context_data, context_mask)
            if self.context_encoder is not None
            else None
        )

        # Per-modality encode: map each encoder over its matching z_t leaf.
        encoded = jax.tree.map(
            lambda encoder, z: encoder(z, t=t),
            self.encoders,
            z_t,
            is_leaf=_is_module_leaf,
        )

        backbone_out = self.backbone(
            encoded, t, context=context, attention_mask=attention_mask
        )

        # Per-modality decode
        return jax.tree.map(
            lambda decoder, out: decoder(out, context=context),
            self.decoders,
            backbone_out,
            is_leaf=_is_module_leaf,
        )
