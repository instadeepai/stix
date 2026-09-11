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

"""Summed-context construction for :class:`stix.nn.EncoderBackboneDecoderNetwork`.

A context module owns the strategy for turning ``(t, context_data,
context_mask)`` into the single context vector consumed by the backbone
and decoders. The EncoderBackboneDecoderNetwork is agnostic to that strategy;
alternate strategies (concatenation, cross-attention, learned gating, ...) can
be added here as new module classes without touching EncoderBackboneDecoderNetwork.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PyTree

from stix.typing import Mask, Time, Var


class SumContextEncoder(nnx.Module):
    """Sum per-source context vectors into a single context.

    Pipeline: ``time_encoder(t)`` plus, for each encoder in ``context_encoders``,
    its ``jax.tree.map`` over the matching leaf of ``context_data``, gated by the
    matching leaf of ``context_mask``; all terms are then summed.
    ``context_encoders`` is an arbitrary pytree of modules and ``context_data`` /
    ``context_mask`` must share its structure, with the encoders as its leaves.
    All encoder outputs must share the same feature dim so the sum is
    well-defined.

    Conditioning semantics:

    - Whole-tree: pass ``context_data=None`` — every conditioning encoder is
      skipped, context is the time term alone. Cleanest path, used internally by
      :func:`~stix.sampling.guidance.get_classifier_free_guidance_generator`
      for its unconditional branch.
    - Per-leaf zero: a ``context_mask`` leaf of ``0`` with any (real-valued)
      ``context_data`` leaf — the encoder runs but its contribution is
      multiplied by zero. Useful when some sources are conditional and others
      not in the same forward.
    - Per-leaf None: a ``context_mask`` leaf of ``None`` (or whole ``None``)
      means fully conditional for that source.
    """

    def __init__(
        self,
        time_encoder: nnx.Module,
        context_encoders: PyTree[nnx.Module] | None = None,
    ) -> None:
        """Initialise with a mandatory time encoder and optional conditioning encoders.

        Args:
            time_encoder: Module fed the scalar ``t`` directly on every call.
            context_encoders: An arbitrary pytree of modules, each fed the
                matching leaf of ``context_data`` and gated by the matching leaf
                of ``context_mask``. ``None`` disables conditioning.
        """
        self.time_encoder = time_encoder
        # Marked as pytree data so nnx tracks the module leaves' params while the
        # container stays a plain pytree we can jax.tree.map over. E.g. an nnx.Dict
        # would itself be a Module and collapse to a single leaf under is_leaf.
        self.context_encoders = (
            nnx.data(context_encoders) if context_encoders is not None else None
        )

    def __call__(
        self,
        t: Time,
        context_data: PyTree[Var] | None = None,
        context_mask: PyTree[Mask | None] | None = None,
    ) -> Var:
        """Build the context vector.

        Args:
            t: Scalar time fed to ``time_encoder``.
            context_data: Conditioning data (a pytree) sharing
                ``context_encoders``' structure; each leaf is fed to its
                encoder. ``None`` is fully unconditional.
            context_mask: A pytree of masks with the same structure as ``context_data``.
                A ``None`` leaf (or whole ``None``) is fully conditional; a ``0``
                leaf zeros that source's contribution.

        Returns:
            The summed context vector consumed by the backbone and decoders.

        Raises:
            ValueError: If ``context_data``'s structure does not match
                ``context_encoders`` (e.g. a missing/extra source). Whole-tree
                ``None`` is fine.
            ValueError: If ``context_data`` is passed but ``context_encoders``
                is ``None``: likely user error (nothing would consume the data).
            ValueError: If ``context_mask`` is passed but ``context_data`` is
                ``None``: mask without data is nonsensical.
        """
        context = self.time_encoder(t)

        if context_data is None and context_mask is not None:
            msg = (
                "context_mask was passed without context_data. Pass both "
                "for the conditional path or neither for the unconditional path."
            )
            raise ValueError(msg)

        if self.context_encoders is None:
            if context_data is not None:
                msg = (
                    "context_data was passed but no context_encoders are "
                    "configured on this SumContextEncoder. Pass "
                    "context_data=None for the unconditional path, or "
                    "configure context_encoders."
                )
                raise ValueError(msg)
            return context

        if context_data is None:
            return context

        # Encode each source: map every encoder over the matching context_data
        # leaf. ``is_leaf`` stops the traversal at the encoder modules; a
        # structural mismatch between context_encoders and context_data raises a
        # ValueError from jax.
        encoded = jax.tree.map(
            lambda encoder, data: encoder(data),
            self.context_encoders,
            context_data,
            is_leaf=lambda leaf: isinstance(leaf, nnx.Module),
        )

        # Gate each source by its mask (a None leaf means fully conditional).
        # ``encoded`` (structure = context_encoders) drives the map; a None leaf
        # in context_mask arrives verbatim via flatten_up_to, so no is_leaf.
        if context_mask is not None:
            encoded = jax.tree.map(
                lambda ctx, mask: (
                    ctx if mask is None else ctx * mask[..., None].astype(ctx.dtype)
                ),
                encoded,
                context_mask,
            )

        # Sum the time term with every source's (gated) contribution.
        return jax.tree.reduce(jnp.add, encoded, context)
