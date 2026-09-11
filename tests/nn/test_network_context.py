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

"""Tests for :class:`stix.nn.EncoderBackboneDecoderNetwork`'s context-encoder plumbing.

Pins that ``EncoderBackboneDecoderNetwork`` is agnostic to context-construction strategy: it
just calls ``context_encoder(t, context_data, context_mask)`` and
forwards the result to backbone + decoders. Strategy-specific semantics
(masking, partial-dict errors, sum invariants) live with the strategy —
see ``tests/nn/test_sum_context_encoder.py``.
"""

from typing import cast

import jax
import jax.numpy as jnp
from conftest import (
    AddContextBackbone,
    BroadcastTimeEncoder,
    ConstantContextEncoder,
    EmbedConditionEncoder,
    IdentityDecoder,
    IdentityEncoder,
)
from flax import nnx
from jaxtyping import PyTree

from stix.nn import EncoderBackboneDecoderNetwork, SumContextEncoder

CTX_DIM = 3
MODALITY_DIM = 3  # matches CTX_DIM so the toy backbone can just add context


def _make_network(context_encoder) -> EncoderBackboneDecoderNetwork:
    rngs = nnx.Rngs(0)
    return EncoderBackboneDecoderNetwork(
        encoders={"mod_a": IdentityEncoder(rngs=rngs)},
        backbone=AddContextBackbone(rngs=rngs),
        decoders={"mod_a": IdentityDecoder(rngs=rngs)},
        context_encoder=context_encoder,
    )


_X_T = {
    "mod_a": jnp.arange(2 * MODALITY_DIM, dtype=jnp.float32).reshape(2, MODALITY_DIM)
}


def test_network_context_encoder_none_passes_through_None():
    """``context_encoder=None`` ⇒ backbone receives ``context=None`` ⇒ encoded input passes through unchanged."""
    net = _make_network(context_encoder=None)
    out = net(_X_T, jnp.float32(0.5))
    assert jnp.array_equal(cast(jax.Array, out["mod_a"]), _X_T["mod_a"])


def test_network_passes_t_and_context_data_to_context_encoder():
    """Pin network output equals ``encoded + time_term + condition_term``.

    With the toy modules wired below:
    - ``IdentityEncoder`` ⇒ encoded == ``_X_T``
    - ``BroadcastTimeEncoder`` ⇒ time_term broadcasts ``t`` to ``(CTX_DIM,)``
    - ``EmbedConditionEncoder(values=(1.0, 5.0))`` ⇒ condition_term is
      ``1.0`` for ``idx=0`` and ``5.0`` for ``idx=1`` (broadcast to feature dim)
    - ``SumContextEncoder`` adds the two terms; ``AddContextBackbone`` adds the
      result to encoded; ``IdentityDecoder`` passes through

    So ``net(_X_T, t, context_data={"label": k}) == _X_T + t + embed[k]``
    exactly, which pins both args are threaded to the context encoder (not
    dropped or replaced by constants) AND that the additive composition is
    correct. ``context_mask`` is covered transitively by
    ``SumContextEncoder``'s mask-semantics tests.
    """
    rngs = nnx.Rngs(0)
    embed_0 = 1.0
    embed_1 = 5.0
    net = EncoderBackboneDecoderNetwork(
        encoders={"mod_a": IdentityEncoder(rngs=rngs)},
        backbone=AddContextBackbone(rngs=rngs),
        decoders={"mod_a": IdentityDecoder(rngs=rngs)},
        context_encoder=SumContextEncoder(
            time_encoder=BroadcastTimeEncoder(shape=(CTX_DIM,), rngs=rngs),
            context_encoders={
                "label": EmbedConditionEncoder(
                    values=(embed_0, embed_1), shape=(CTX_DIM,), rngs=rngs
                ),
            },
        ),
    )
    cond0: PyTree[jnp.ndarray] = {"label": jnp.array([0, 0], dtype=jnp.int32)}
    cond1: PyTree[jnp.ndarray] = {"label": jnp.array([1, 1], dtype=jnp.int32)}

    t1 = jnp.float32(0.3)
    t2 = jnp.float32(0.7)

    out_a = net(_X_T, t1, context_data=cond0)
    out_b = net(_X_T, t2, context_data=cond0)  # different t
    out_c = net(_X_T, t1, context_data=cond1)  # different context_data

    assert jnp.allclose(cast(jax.Array, out_a["mod_a"]), _X_T["mod_a"] + t1 + embed_0)
    assert jnp.allclose(cast(jax.Array, out_b["mod_a"]), _X_T["mod_a"] + t2 + embed_0)
    assert jnp.allclose(cast(jax.Array, out_c["mod_a"]), _X_T["mod_a"] + t1 + embed_1)


def test_network_with_context_encoder_jits():
    """An ``EncoderBackboneDecoderNetwork`` containing a real ``SumContextEncoder`` traces under ``jax.jit`` via split/merge."""
    rngs = nnx.Rngs(0)
    net = EncoderBackboneDecoderNetwork(
        encoders={"mod_a": IdentityEncoder(rngs=rngs)},
        backbone=AddContextBackbone(rngs=rngs),
        decoders={"mod_a": IdentityDecoder(rngs=rngs)},
        context_encoder=SumContextEncoder(
            time_encoder=ConstantContextEncoder(value=0.5, shape=(CTX_DIM,), rngs=rngs),
        ),
    )
    graphdef, state = nnx.split(net)

    @jax.jit
    def call(state, t):
        merged = nnx.merge(graphdef, state)
        return merged(_X_T, t)

    out = call(state, jnp.float32(0.5))
    out_arr = cast(jax.Array, out["mod_a"])
    assert out_arr.shape == (2, MODALITY_DIM)
    assert jnp.all(jnp.isfinite(out_arr))
