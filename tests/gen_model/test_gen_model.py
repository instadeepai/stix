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

"""Tests for the GenerativeModel base class."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from conftest import (
    SHAPE,
    IdentityNetwork,
    NoiseOneSidedGenerativeModel,
    build_one_sided_registry,
    one_sided_si,
)

from stix.core.embedder import IdentityEmbedder
from stix.core.gen_model.gen_model import GenerativeModel
from stix.core.modality import Modality, ModalityRegistry
from stix.typing import EmbeddedSourceTargetPair, RawSourceTargetPair, Var


# ── Helpers ──
class ToyGenerativeModel(GenerativeModel):
    """Minimal concrete subclass for testing the ABC's non-abstract methods."""

    def get_generator(self, net_out, z_t, t):
        return {}

    def get_loss(self, net_out, t, z_t, epsilon, embedded_pairs, raw_pairs, loss_mask):
        return jnp.array(0.0)


def toy_registry() -> ModalityRegistry:
    """Single-modality (``mod_a``) registry with an IdentityEmbedder."""
    return build_one_sided_registry()


def toy_gen_model() -> ToyGenerativeModel:
    return ToyGenerativeModel(
        network=IdentityNetwork(),
        modality_registry=toy_registry(),
    )


# ══════════════════════════════════════════════════════════════════════
# Construction
# ══════════════════════════════════════════════════════════════════════


def test_init_stores_components():
    """Constructor stores the network and the modality registry."""
    network = IdentityNetwork()
    modality_registry = toy_registry()
    model = ToyGenerativeModel(network=network, modality_registry=modality_registry)

    assert model.network is network
    assert model.modality_registry is modality_registry


# ══════════════════════════════════════════════════════════════════════
# get_network_output
# ══════════════════════════════════════════════════════════════════════


def test_get_network_output_passthrough():
    """get_network_output method should be matching the network's output."""
    gen_model = toy_gen_model()
    network = gen_model.network

    t = jnp.array(0.5)
    z_t = {"mod_a": jnp.array([1.0, 2.0, 3.0])}

    network_out = network(z_t, t, None, None, None)
    gen_model_out = gen_model.get_network_output(z_t, t, None, None, None)

    assert jnp.array_equal(gen_model_out["mod_a"], network_out["mod_a"])


# ══════════════════════════════════════════════════════════════════════
# get_embeddings
# ══════════════════════════════════════════════════════════════════════


def test_get_embeddings_two_sided():
    """Both source and target are embedded via the per-modality embedder.

    The result is a plain per-modality dict whose leaves are
    :class:`EmbeddedSourceTargetPair`.
    """
    gen_model = toy_gen_model()
    source = jnp.array([1.0, 2.0, 3.0])
    target = jnp.array([4.0, 5.0, 6.0])
    raw_batch = {"mod_a": RawSourceTargetPair(target=target, source=source)}

    embedded_pairs = gen_model.get_embeddings(raw_batch)

    # Pytree structure: dict keyed by modality, leaves are EmbeddedSourceTargetPair.
    assert set(embedded_pairs.keys()) == {"mod_a"}
    assert isinstance(embedded_pairs["mod_a"], EmbeddedSourceTargetPair)

    emb_source = embedded_pairs["mod_a"].source
    assert emb_source is not None

    # IdentityEmbedder -> output equals input
    assert jnp.array_equal(emb_source, source)
    assert jnp.array_equal(embedded_pairs["mod_a"].target, target)


def test_get_embeddings_one_sided_none_source():
    """When source is None, embedded_source stays None."""
    gen_model = toy_gen_model()
    target = jnp.array([4.0, 5.0, 6.0])
    raw_batch = {"mod_a": RawSourceTargetPair(target=target, source=None)}

    embedded_pairs = gen_model.get_embeddings(raw_batch)

    assert set(embedded_pairs.keys()) == {"mod_a"}
    assert isinstance(embedded_pairs["mod_a"], EmbeddedSourceTargetPair)

    assert embedded_pairs["mod_a"].source is None
    # IdentityEmbedder -> output equals input
    assert jnp.array_equal(embedded_pairs["mod_a"].target, target)


# ══════════════════════════════════════════════════════════════════════
# Construction validation
# ══════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("missing", ["interpolant", "embedder"])
def test_init_raises_when_modality_missing_required_field(missing):
    """Required fields are validated at construction (a real raise, not an assert),
    so downstream consumers can rely on them even under ``python -O``. The check
    lives on the base ``GenerativeModel.__init__``, reached here via inheritance."""
    fields = {
        "shape": SHAPE,
        "is_discrete": False,
        "interpolant": one_sided_si(),
        "embedder": IdentityEmbedder(dm_shape=SHAPE),
    }
    del fields[missing]  # omit exactly one required field
    modality_registry = ModalityRegistry({"mod_a": Modality(**fields)})
    with pytest.raises(ValueError, match=missing):
        ToyGenerativeModel(
            network=IdentityNetwork(),
            modality_registry=modality_registry,
        )


# ══════════════════════════════════════════════════════════════════════
# get_loss (exercised via the concrete noise-prediction NoiseOneSidedGenerativeModel)
# ══════════════════════════════════════════════════════════════════════


def _noise_gen_model():
    """Concrete one-sided noise-prediction model with IdentityNetwork and MSE loss."""
    return NoiseOneSidedGenerativeModel(
        network=IdentityNetwork(),
        modality_registry=build_one_sided_registry(),
    )


def _interpolant(gen_model):
    """The single-modality interpolant of a toy gen-model."""
    return gen_model.modality_registry.registry["mod_a"].interpolant


def test_get_loss_returns_finite_scalar():
    """Loss is a finite scalar."""
    gen_model = _noise_gen_model()
    si = _interpolant(gen_model)

    key = jr.PRNGKey(0)
    t = jnp.array(0.3)
    key_target, key_epsilon, key_out_var = jr.split(key, 3)

    target = jr.normal(key_target, SHAPE)
    epsilon = jr.normal(key_epsilon, SHAPE)
    out_var = jr.normal(key_out_var, SHAPE)

    raw_pairs = {"mod_a": RawSourceTargetPair(target=target, source=None)}
    emb_pairs = gen_model.get_embeddings(raw_pairs)
    z_t = {"mod_a": si.interpolate(emb_pairs["mod_a"], t, epsilon)}
    net_out: dict[str, Var | tuple[Var, Var]] = {"mod_a": out_var}

    loss = gen_model.get_loss(
        net_out=net_out,
        t=t,
        z_t=z_t,
        epsilon={"mod_a": epsilon},
        embedded_pairs=emb_pairs,
        raw_pairs=raw_pairs,
        loss_mask={"mod_a": jnp.ones(SHAPE)},
    )
    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_get_loss_zero_when_prediction_is_ground_truth():
    """Loss is zero when the network predicts the exact ground truth noise."""
    gen_model = _noise_gen_model()
    si = _interpolant(gen_model)

    key = jr.PRNGKey(42)
    t = jnp.array(0.3)
    key_target, key_epsilon = jr.split(key)

    target = jr.normal(key_target, SHAPE)
    epsilon = jr.normal(key_epsilon, SHAPE)

    raw_pairs = {"mod_a": RawSourceTargetPair(target=target, source=None)}
    emb_pairs = gen_model.get_embeddings(raw_pairs)
    z_t = {"mod_a": si.interpolate(emb_pairs["mod_a"], t, epsilon)}
    net_out: dict[str, Var | tuple[Var, Var]] = {"mod_a": epsilon}

    loss = gen_model.get_loss(
        net_out=net_out,
        t=t,
        z_t=z_t,
        epsilon={"mod_a": epsilon},
        embedded_pairs=emb_pairs,
        raw_pairs=raw_pairs,
        loss_mask={"mod_a": jnp.ones(SHAPE)},
    )
    assert jnp.allclose(loss, 0.0, atol=1e-6)
