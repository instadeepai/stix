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

"""Tests for the discrete DFM generative models.

Covers construction/validation of :class:`PosteriorMixtureGenerativeModel` and
:class:`VelocityAndPosteriorGenerativeModel`, the ``TransitionRates`` output of
:meth:`get_generator`, and a training-time weighted cross-entropy pass through
the :class:`LossPipeline` (the discrete source is implicit in the noise, so no
source variable is materialised).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from conftest import (
    BATCH_SIZE,
    DISCRETE_SHAPE,
    NUM_CATEGORIES,
    ToyLogitsNetwork,
    build_mixture_registry,
    mask_mixture_interpolant,
    one_sided_si,
    toy_multimodal_gen_model,
    toy_posterior_mixture_gen_model,
)

from stix.core.embedder import IdentityEmbedder
from stix.core.gen_model.factory import (
    PosteriorMixtureGenerativeModel,
    VelocityAndPosteriorGenerativeModel,
)
from stix.core.generator import TransitionRates
from stix.core.interpolant import (
    DiscreteInterpolant,
    MaskDiscreteInterpolant,
    UniformDiscreteInterpolant,
)
from stix.core.modality import Modality, ModalityRegistry
from stix.training.loss_pipeline import LossPipeline
from stix.typing import EmbeddedSourceTargetPair, RawSourceTargetPair
from stix.typing.data import Batch


def _discrete_batch(batch_size: int = BATCH_SIZE) -> Batch:
    """A raw batch of categorical indices (no source: the pipeline fills it)."""
    target = jr.randint(
        jr.PRNGKey(0), (batch_size, *DISCRETE_SHAPE), 0, NUM_CATEGORIES
    ).astype(jnp.int32)
    return Batch(
        raw_batch={"tokens": RawSourceTargetPair(target=target, source=None)},
        context_data=None,
        context_mask=None,
        attn_mask=None,
        loss_mask=None,
    )


# ── Construction / validation ──


def test_posterior_model_requires_target_posterior_rates():
    """An interpolant without ``rates_from_target_posterior`` is rejected."""
    registry = ModalityRegistry(
        {
            "tokens": Modality(
                shape=DISCRETE_SHAPE,
                is_discrete=True,
                interpolant=one_sided_si(),
                embedder=IdentityEmbedder(dm_shape=DISCRETE_SHAPE),
                num_categories=NUM_CATEGORIES,
            )
        }
    )
    with pytest.raises(TypeError, match="rates_from_target_posterior"):
        PosteriorMixtureGenerativeModel(
            network=ToyLogitsNetwork([0.0] * NUM_CATEGORIES),
            modality_registry=registry,
        )


def test_posterior_model_accepts_a_user_defined_interpolant():
    """The requirement is a `DiscreteInterpolant` *implementing the method*.

    A user subclassing the base and providing `rates_from_target_posterior` must be
    accepted without being one of the two shipped classes: the check stays a
    capability probe, not a closed list. This also guards the literal
    ``"rates_from_target_posterior"`` in that probe — misspell it and this test fails.
    """
    shipped = MaskDiscreteInterpolant(
        num_categories=NUM_CATEGORIES, kappa_fn=lambda t: t
    )

    class UserDefinedInterpolant(MaskDiscreteInterpolant):
        """Neither of the shipped classes, but a DiscreteInterpolant with the method."""

        def rates_from_target_posterior(
            self, target_posterior, z_t, t, *, backward=False
        ):
            return shipped.rates_from_target_posterior(
                target_posterior, z_t, t, backward=backward
            )

    interpolant = UserDefinedInterpolant(
        num_categories=NUM_CATEGORIES, kappa_fn=lambda t: t
    )
    assert type(interpolant) not in (
        MaskDiscreteInterpolant,
        UniformDiscreteInterpolant,
    )
    assert isinstance(interpolant, DiscreteInterpolant)
    assert hasattr(interpolant, "rates_from_target_posterior")

    registry = ModalityRegistry(
        {
            "tokens": Modality(
                shape=DISCRETE_SHAPE,
                is_discrete=True,
                interpolant=interpolant,
                embedder=IdentityEmbedder(dm_shape=DISCRETE_SHAPE),
                num_categories=NUM_CATEGORIES,
            )
        }
    )
    PosteriorMixtureGenerativeModel(
        network=ToyLogitsNetwork([0.0] * NUM_CATEGORIES),
        modality_registry=registry,
    )


def test_posterior_model_requires_num_categories():
    """A discrete modality without ``num_categories`` is rejected."""
    registry = ModalityRegistry(
        {
            "tokens": Modality(
                shape=DISCRETE_SHAPE,
                is_discrete=True,
                interpolant=mask_mixture_interpolant(),
                embedder=IdentityEmbedder(dm_shape=DISCRETE_SHAPE),
            )
        }
    )
    with pytest.raises((ValueError, AssertionError)):
        PosteriorMixtureGenerativeModel(
            network=ToyLogitsNetwork([0.0] * NUM_CATEGORIES),
            modality_registry=registry,
        )


def test_multimodal_model_rejects_unsupported_interpolant():
    """A modality that is neither one-sided-linear nor discrete is rejected."""
    from stix.core.interpolant.linear_interpolant import LinearStochasticInterpolant

    registry = ModalityRegistry(
        {
            "cont": Modality(
                shape=(3,),
                is_discrete=False,
                interpolant=LinearStochasticInterpolant(
                    gamma_fn=lambda t: (t * (1 - t)) ** 0.5,
                    alpha_fn=lambda t: 1 - t,
                    beta_fn=lambda t: t,
                ),
                embedder=IdentityEmbedder(dm_shape=(3,)),
            )
        }
    )
    with pytest.raises(TypeError, match="OneSidedLinearStochastic"):
        VelocityAndPosteriorGenerativeModel(
            network=ToyLogitsNetwork([0.0] * NUM_CATEGORIES),
            modality_registry=registry,
        )


# ── get_generator ──


@pytest.mark.parametrize("source", ["mask", "uniform"])
def test_get_generator_returns_transition_rates(source):
    """``get_generator`` yields ``TransitionRates`` of the right state width."""
    gen_model = toy_posterior_mixture_gen_model(source=source)
    expected_states = NUM_CATEGORIES + (1 if source == "mask" else 0)
    # z_t is a one-hot state over the CTMC states (every position on state 0).
    z_t = jax.nn.one_hot(jnp.zeros(DISCRETE_SHAPE, dtype=jnp.int32), expected_states)
    net_out = gen_model.get_network_output(
        {"tokens": z_t}, jnp.array(0.5), None, None, None
    )
    generators = gen_model.get_generator(net_out, {"tokens": z_t}, jnp.array(0.5))
    rates = generators["tokens"]
    assert isinstance(rates, TransitionRates)
    assert rates.forward_rates.shape == (*DISCRETE_SHAPE, expected_states)
    assert rates.backward_rates.shape == (*DISCRETE_SHAPE, expected_states)
    # A valid CTMC generator: rows sum to (approximately) zero.
    assert jnp.allclose(jnp.sum(rates.forward_rates, axis=-1), 0.0, atol=1e-3)
    assert jnp.allclose(jnp.sum(rates.backward_rates, axis=-1), 0.0, atol=1e-3)


# ── Training loss ──


def _posterior_ce_loss(source: str, logits, z_t, target, loss_mask):
    """Evaluate ``PosteriorMixtureGenerativeModel.get_loss`` on a single sample."""
    gen_model = PosteriorMixtureGenerativeModel(
        network=ToyLogitsNetwork([0.0] * NUM_CATEGORIES),
        modality_registry=build_mixture_registry(source=source),
    )
    dummy_raw = RawSourceTargetPair(
        target=jnp.zeros(z_t.shape[:-1], dtype=jnp.int32), source=None
    )
    dummy_epsilon = jnp.zeros((*z_t.shape[:-1], 2))
    return gen_model.get_loss(
        net_out={"tokens": logits},
        t=jnp.asarray(0.5),
        z_t={"tokens": z_t},
        epsilon={"tokens": dummy_epsilon},
        embedded_pairs={"tokens": EmbeddedSourceTargetPair(target=target, source=None)},
        raw_pairs={"tokens": dummy_raw},
        loss_mask={"tokens": loss_mask},
    )


def test_posterior_ce_skips_revealed_mask_positions():
    """On the mask path, frozen revealed tokens do not contribute CE."""
    registry = build_mixture_registry(source="mask")
    interpolant = registry._modality_leaves()[0].interpolant
    assert interpolant is not None
    seq_len = DISCRETE_SHAPE[0]
    logits = jnp.zeros((seq_len, NUM_CATEGORIES))
    target = jax.nn.one_hot(
        jnp.zeros((seq_len,), dtype=jnp.int32), interpolant.num_states
    )
    # Mixing coin < kappa_t selects the target atom, so every position is revealed.
    epsilon = jnp.full((seq_len, 2), 0.1)
    t = jnp.asarray(0.5)
    z_t = interpolant.interpolate(
        EmbeddedSourceTargetPair(target=target, source=None), t, epsilon
    )
    assert jnp.all(jnp.argmax(z_t, axis=-1) == 0)
    loss = _posterior_ce_loss("mask", logits, z_t, target, jnp.ones_like(target))
    assert jnp.allclose(loss, 0.0, atol=1e-5)


def test_posterior_ce_mask_only_on_still_masked():
    """Mask-path CE averages only over still-masked positions."""
    registry = build_mixture_registry(source="mask")
    interpolant = registry._modality_leaves()[0].interpolant
    assert interpolant is not None
    seq_len = DISCRETE_SHAPE[0]
    logits = jnp.zeros((seq_len, NUM_CATEGORIES))
    target_idx = jnp.zeros((seq_len,), dtype=jnp.int32)
    target = jax.nn.one_hot(target_idx, interpolant.num_states)
    # First half revealed (target atom), second half still masked.
    z_t = target.at[seq_len // 2 :].set(
        jax.nn.one_hot(
            jnp.full((seq_len - seq_len // 2,), interpolant.mask_index),
            interpolant.num_states,
        )
    )
    loss = _posterior_ce_loss("mask", logits, z_t, target, jnp.ones_like(target))
    assert jnp.allclose(loss, jnp.log(NUM_CATEGORIES), atol=1e-5)


def test_posterior_ce_uniform_includes_revealed_positions():
    """Uniform DFM still trains every position (rewrites remain possible)."""
    registry = build_mixture_registry(source="uniform")
    interpolant = registry._modality_leaves()[0].interpolant
    assert interpolant is not None
    seq_len = DISCRETE_SHAPE[0]
    logits = jnp.zeros((seq_len, NUM_CATEGORIES))
    target = jax.nn.one_hot(
        jnp.zeros((seq_len,), dtype=jnp.int32), interpolant.num_states
    )
    z_t = target
    loss = _posterior_ce_loss("uniform", logits, z_t, target, jnp.ones_like(target))
    assert jnp.allclose(loss, jnp.log(NUM_CATEGORIES), atol=1e-5)


@pytest.mark.parametrize("source", ["mask", "uniform"])
def test_posterior_model_training_loss_is_finite(source):
    """A LossPipeline pass over the discrete model returns a finite scalar."""
    gen_model = PosteriorMixtureGenerativeModel(
        network=ToyLogitsNetwork([0.0] * NUM_CATEGORIES),
        modality_registry=build_mixture_registry(source=source),
    )
    pipeline = LossPipeline(antithetical=False)
    loss, metrics = pipeline(gen_model, _discrete_batch(), jr.PRNGKey(0))
    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert "loss" in metrics


def test_multimodal_training_loss_is_finite():
    """A LossPipeline pass over the joint model returns a finite scalar."""
    gen_model = toy_multimodal_gen_model()
    cont_target = jr.normal(jr.PRNGKey(1), (BATCH_SIZE, 3))
    tokens_target = jr.randint(
        jr.PRNGKey(2), (BATCH_SIZE, *DISCRETE_SHAPE), 0, NUM_CATEGORIES
    ).astype(jnp.int32)
    batch = Batch(
        raw_batch={
            "cont": RawSourceTargetPair(target=cont_target, source=None),
            "tokens": RawSourceTargetPair(target=tokens_target, source=None),
        },
        context_data=None,
        context_mask=None,
        attn_mask=None,
        loss_mask=None,
    )
    pipeline = LossPipeline(antithetical=False)
    loss, _ = pipeline(gen_model, batch, jr.PRNGKey(0))
    assert loss.shape == ()
    assert jnp.isfinite(loss)
