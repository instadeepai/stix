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

"""Multi-modality tests for :class:`stix.nn.EncoderBackboneDecoderNetwork`.

These pin the ``jax.tree.map`` encode/decode path: each modality flows through
its own encoder and decoder; the network indexes nothing by modality name, so a
list or a nested pytree routes just as a flat dict does; the components stay
visible to nnx tracing and receive finite, nonzero gradients; and the arguments
the network only forwards actually arrive.

Every network here is multi-modality, with 2 modalities of different shapes.
"""

from typing import cast

import jax
import jax.numpy as jnp
import pytest
from conftest import AddContextBackbone
from flax import nnx

from stix.nn import EncoderBackboneDecoderNetwork


class ScaleEncoder(nnx.Module):
    """Encoder multiplying its modality by a learnable scalar."""

    def __init__(self, init: float, *, rngs: nnx.Rngs | None = None) -> None:
        del rngs
        self.scale = nnx.Param(jnp.asarray(init, jnp.float32))

    def __call__(self, x, t):
        del t
        return x * self.scale[...]


class BiasDecoder(nnx.Module):
    """Decoder adding a learnable scalar bias."""

    def __init__(self, init: float, *, rngs: nnx.Rngs | None = None) -> None:
        del rngs
        self.bias = nnx.Param(jnp.asarray(init, jnp.float32))

    def __call__(self, x, context=None):
        del context
        return x + self.bias[...]


class ContextBiasDecoder(nnx.Module):
    """Decoder adding the context vector it is handed."""

    def __call__(self, x, context=None):
        if context is None:
            raise AssertionError("decoder was not given the context vector")
        return x + context


class TwoHeadDecoder(nnx.Module):
    """Decoder returning a ``(Var, Var)`` pair, as two-sided models do."""

    def __call__(self, x, context=None):
        del context
        return x, x * 2.0


class RecordingBackbone(nnx.Module):
    """Pass-through backbone that records the arguments it was handed."""

    def __init__(self) -> None:
        self.seen: dict = nnx.static({})

    def __call__(self, encoded, t, context=None, attention_mask=None):
        self.seen["attention_mask"] = attention_mask
        return encoded


class TimeScaleEncoder(nnx.Module):
    """Encoder scaling its modality by the time it is given."""

    def __call__(self, x, t):
        return x * t


class FixedContextEncoder(nnx.Module):
    """Context encoder returning a fixed vector, ignoring its inputs."""

    def __call__(self, t, context_data=None, context_mask=None):
        del t, context_data, context_mask
        return jnp.full((2,), 5.0, jnp.float32)


def _multimodal_network() -> EncoderBackboneDecoderNetwork:
    """Two modalities with distinct encoder scales and decoder biases.

    ``AddContextBackbone`` is a pass-through when ``context is None``, so the
    end-to-end map is ``mod_a: x*10 + 1`` and ``mod_b: x*100 + 2``.
    """
    return EncoderBackboneDecoderNetwork(
        encoders={"mod_a": ScaleEncoder(10.0), "mod_b": ScaleEncoder(100.0)},
        backbone=AddContextBackbone(),
        decoders={"mod_a": BiasDecoder(1.0), "mod_b": BiasDecoder(2.0)},
    )


# Distinct content and feature dims (3 vs 5) per modality.
_X_T = {
    "mod_a": jnp.arange(2 * 3, dtype=jnp.float32).reshape(2, 3),
    "mod_b": jnp.arange(2 * 5, dtype=jnp.float32).reshape(2, 5) + 100.0,
}


def test_network_routes_each_modality_through_its_own_components():
    """Each modality flows through its own encoder and decoder, by position.

    The components are held in a list, so no modality names are involved, and
    the two modalities have different shapes so a permutation is a shape error
    as well as a wrong value.
    """
    net = EncoderBackboneDecoderNetwork(
        encoders=[ScaleEncoder(10.0), ScaleEncoder(100.0)],
        backbone=AddContextBackbone(),
        decoders=[BiasDecoder(1.0), BiasDecoder(2.0)],
    )
    first = jnp.arange(2 * 3, dtype=jnp.float32).reshape(2, 3)
    second = jnp.arange(2 * 5, dtype=jnp.float32).reshape(2, 5) + 100.0

    out = net([first, second], jnp.float32(0.5))
    assert isinstance(out, list)
    assert jnp.allclose(cast(jax.Array, out[0]), first * 10.0 + 1.0)
    assert jnp.allclose(cast(jax.Array, out[1]), second * 100.0 + 2.0)


def test_network_forwards_context_to_decoders():
    """Decoders receive the context vector the context encoder built."""
    net = EncoderBackboneDecoderNetwork(
        encoders={"mod_a": ScaleEncoder(1.0)},
        backbone=AddContextBackbone(),
        decoders={"mod_a": ContextBiasDecoder()},
        context_encoder=FixedContextEncoder(),
    )
    out = net({"mod_a": jnp.ones((2,), jnp.float32)}, jnp.float32(0.5))
    # Backbone adds context once, then the decoder adds it again: 1 + 5 + 5.
    assert jnp.allclose(cast(jax.Array, out["mod_a"]), 11.0)


def test_network_tuple_returning_decoders():
    """Two-headed decoders come back as a ``(Var, Var)`` leaf per modality."""
    net = EncoderBackboneDecoderNetwork(
        encoders={"mod_a": ScaleEncoder(1.0), "mod_b": ScaleEncoder(1.0)},
        backbone=AddContextBackbone(),
        decoders={"mod_a": TwoHeadDecoder(), "mod_b": TwoHeadDecoder()},
    )
    out = net(_X_T, jnp.float32(0.5))
    for modality in ("mod_a", "mod_b"):
        head_one, head_two = out[modality]
        assert head_one.shape == _X_T[modality].shape
        assert jnp.allclose(head_two, head_one * 2.0)


def test_network_encoder_decoder_params_trainable():
    """Encoder/decoder params get finite, nonzero gradients.

    Guards against the per-modality module leaves being dropped from nnx
    tracing, which would silently vanish their gradients.
    """
    net = _multimodal_network()

    def loss_fn(module):
        out = module(_X_T, jnp.float32(0.5))
        return sum(jnp.sum(v**2) for v in jax.tree.leaves(out))

    grads = nnx.grad(loss_fn)(net)
    grad_leaves = jax.tree.leaves(grads)
    assert len(grad_leaves) == 4
    assert all(jnp.all(jnp.isfinite(g)) for g in grad_leaves)
    assert all(jnp.any(g != 0.0) for g in grad_leaves)


def test_network_encoder_decoder_structure_mismatch_raises():
    """Encoders and decoders with different structure raise at construction."""
    with pytest.raises(ValueError, match="share pytree structure"):
        EncoderBackboneDecoderNetwork(
            encoders={"mod_a": ScaleEncoder(1.0), "mod_b": ScaleEncoder(1.0)},
            backbone=AddContextBackbone(),
            decoders={"mod_a": BiasDecoder(0.0)},  # missing mod_b
        )


def test_network_forwards_attention_mask_to_backbone():
    """The attention mask reaches the backbone unchanged."""
    backbone = RecordingBackbone()
    net = EncoderBackboneDecoderNetwork(
        encoders={"mod_a": ScaleEncoder(1.0)},
        backbone=backbone,
        decoders={"mod_a": BiasDecoder(0.0)},
    )
    mask = {"mod_a": jnp.ones((2,), dtype=jnp.bool_)}
    net({"mod_a": jnp.ones((2,), jnp.float32)}, jnp.float32(0.5), attention_mask=mask)
    seen = backbone.seen["attention_mask"]
    assert seen is not None
    assert jnp.array_equal(seen["mod_a"], mask["mod_a"])


def test_network_passes_time_to_encoders():
    """Encoders receive the time scalar, not a default.

    ``DiTEncoder`` standardises its input using ``t``, so an encoder that never
    sees it would silently produce the wrong embedding.
    """
    net = EncoderBackboneDecoderNetwork(
        encoders={"mod_a": TimeScaleEncoder()},
        backbone=AddContextBackbone(),
        decoders={"mod_a": BiasDecoder(0.0)},
    )
    x = {"mod_a": jnp.ones((2,), jnp.float32)}
    early = net(x, jnp.float32(0.25))
    late = net(x, jnp.float32(0.75))
    assert jnp.allclose(cast(jax.Array, early["mod_a"]), 0.25)
    assert jnp.allclose(cast(jax.Array, late["mod_a"]), 0.75)
