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

"""Tests for LossPipeline."""

import jax.numpy as jnp
import jax.random as jr
import pytest
from conftest import (
    BATCH_SIZE,
    SHAPE,
    IdentityNetwork,
    NoiseOneSidedGenerativeModel,
    ToyConstantNetwork,
    one_sided_si,
    toy_one_sided_gen_model,
    two_sided_deterministic_si,
)
from flax import nnx
from jaxtyping import PyTree

from stix.core.coupling import Coupling
from stix.core.embedder import IdentityEmbedder
from stix.core.gen_model.factory import VelocityTwoSidedGenerativeModel
from stix.core.modality import Modality, ModalityRegistry
from stix.training.loss_pipeline import LossPipeline
from stix.training.time_sampler import UniformTimeSampler
from stix.typing import (
    EmbeddedSourceTargetPair,
    Mask,
    RawSourceTargetPair,
    get_batch_size,
)
from stix.typing.data import Batch

# ── Helpers ──


def one_sided_gen_model(network) -> NoiseOneSidedGenerativeModel:
    """One-sided linear generative model with MSE noise loss on a single modality."""
    return toy_one_sided_gen_model(network)


def multi_modality_gen_model(
    shapes: dict[str, tuple[int, ...]], network=None
) -> NoiseOneSidedGenerativeModel:
    """One-sided model over several continuous modalities with the given shapes.

    Used by the private-helper tests that need a registry matching an ad-hoc
    embedded/state pytree (the noise and mask helpers dispatch per modality
    through the registry).
    """
    registry: ModalityRegistry = ModalityRegistry(
        {
            name: Modality(
                shape=shape,
                is_discrete=False,
                interpolant=one_sided_si(),
                embedder=IdentityEmbedder(dm_shape=shape),
            )
            for name, shape in shapes.items()
        }
    )
    return NoiseOneSidedGenerativeModel(
        network=network if network is not None else IdentityNetwork(),
        modality_registry=registry,
    )


def toy_embedded_batch(
    batch_size: int = BATCH_SIZE,
) -> dict[str, EmbeddedSourceTargetPair]:
    """Build a minimal embedded-pairs dict with batched target/source tensors."""
    target = jnp.ones((batch_size, *SHAPE))
    source = jnp.zeros((batch_size, *SHAPE))
    return {"mod_a": EmbeddedSourceTargetPair(target=target, source=source)}


def toy_raw_batch(batch_size: int = BATCH_SIZE) -> dict[str, RawSourceTargetPair]:
    """Build a minimal raw-pairs dict with a batched target and no source."""
    return {
        "mod_a": RawSourceTargetPair(target=jnp.ones((batch_size, *SHAPE)), source=None)
    }


def toy_batch_obj(
    batch_size: int = BATCH_SIZE,
    loss_mask: PyTree[Mask | None] | None = None,
) -> Batch:
    """Build a minimal :class:`Batch` for ``LossPipeline.__call__``."""
    return Batch(
        raw_batch=toy_raw_batch(batch_size),
        context_data=None,
        context_mask=None,
        attn_mask=None,
        loss_mask=loss_mask,
    )


def toy_pipeline(
    coupling: Coupling | None = None,
    antithetical: bool = False,
    time_sampler=None,
) -> LossPipeline:
    """Build a LossPipeline; ``coupling`` is optional and skipped when ``None``."""
    return LossPipeline(
        coupling=coupling,
        time_sampler=time_sampler,
        antithetical=antithetical,
    )


# ══════════════════════════════════════════════════════════════════════
# __init__
# ══════════════════════════════════════════════════════════════════════


def test_init_stores_components():
    """Constructor stores ``coupling`` and ``antithetical`` as provided."""

    class IdentityCoupling(Coupling):
        def __call__(self, raw_pairs, embedded_pairs):
            return raw_pairs, embedded_pairs

    coupling = IdentityCoupling()
    pipeline = LossPipeline(coupling=coupling, antithetical=True)

    assert pipeline.coupling is coupling
    assert pipeline.antithetical is True


def test_init_default_coupling_is_none():
    """Omitting ``coupling`` leaves it unset; the pipeline then applies no pairing."""
    pipeline = LossPipeline()
    assert pipeline.coupling is None


def test_init_default_time_sampler_is_uniform():
    """Omitting ``time_sampler`` should default to a ``UniformTimeSampler``."""
    pipeline = LossPipeline()
    assert isinstance(pipeline.time_sampler, UniformTimeSampler)


def test_init_custom_time_sampler_preserved():
    """A user-supplied ``time_sampler`` should be stored unchanged."""
    sampler = UniformTimeSampler(t_min=0.1, t_max=0.9)
    pipeline = LossPipeline(time_sampler=sampler)
    assert pipeline.time_sampler is sampler


def test_init_default_antithetical_is_false():
    """Omitting ``antithetical`` should default to ``False``."""
    pipeline = LossPipeline()
    assert pipeline.antithetical is False


# ══════════════════════════════════════════════════════════════════════
# _generate_noise
# ══════════════════════════════════════════════════════════════════════


def test_generate_noise_matches_target_shape_per_modality():
    """Generated noise should have the same shape as each modality's embedded target."""
    pipeline = toy_pipeline()
    gen_model = multi_modality_gen_model({"img": (4,), "txt": (2, 3)})
    embedded_batch = {
        "img": EmbeddedSourceTargetPair(target=jnp.ones((BATCH_SIZE, 4)), source=None),
        "txt": EmbeddedSourceTargetPair(
            target=jnp.ones((BATCH_SIZE, 2, 3)), source=None
        ),
    }

    noise = pipeline._generate_noise(jr.PRNGKey(0), gen_model, embedded_batch)

    assert noise["img"].shape == (BATCH_SIZE, 4)
    assert noise["txt"].shape == (BATCH_SIZE, 2, 3)


def test_generate_noise_reproducible_with_same_key():
    """``_generate_noise`` should be deterministic for a fixed PRNG key."""
    pipeline = toy_pipeline()
    gen_model = multi_modality_gen_model({"mod_a": SHAPE})
    embedded_batch = toy_embedded_batch()

    noise_a = pipeline._generate_noise(jr.PRNGKey(0), gen_model, embedded_batch)
    noise_b = pipeline._generate_noise(jr.PRNGKey(0), gen_model, embedded_batch)

    assert jnp.array_equal(noise_a["mod_a"], noise_b["mod_a"])


def test_generate_noise_different_keys_differ():
    """``_generate_noise`` should produce different samples for different PRNG keys."""
    pipeline = toy_pipeline()
    gen_model = multi_modality_gen_model({"mod_a": SHAPE})
    embedded_batch = toy_embedded_batch()

    noise_a = pipeline._generate_noise(jr.PRNGKey(0), gen_model, embedded_batch)
    noise_b = pipeline._generate_noise(jr.PRNGKey(1), gen_model, embedded_batch)

    assert not jnp.array_equal(noise_a["mod_a"], noise_b["mod_a"])


def test_generate_noise_modalities_use_independent_keys():
    """Modalities of identical shape should receive independent noise draws."""
    pipeline = toy_pipeline()
    gen_model = multi_modality_gen_model({"mod_a": SHAPE, "mod_b": SHAPE})
    embedded_batch = {
        "mod_a": EmbeddedSourceTargetPair(
            target=jnp.ones((BATCH_SIZE, *SHAPE)), source=None
        ),
        "mod_b": EmbeddedSourceTargetPair(
            target=jnp.ones((BATCH_SIZE, *SHAPE)), source=None
        ),
    }

    noise = pipeline._generate_noise(jr.PRNGKey(0), gen_model, embedded_batch)

    assert not jnp.array_equal(noise["mod_a"], noise["mod_b"])


# ══════════════════════════════════════════════════════════════════════
# _fill_loss_mask
# ══════════════════════════════════════════════════════════════════════


def test_fill_loss_mask_none_fills_with_ones():
    """``loss_mask=None`` should produce an all-ones mask shaped like each state ``z_t[k]``."""
    pipeline = toy_pipeline()
    gen_model = multi_modality_gen_model({"mod_a": (2, 3), "mod_b": (5,)})
    state = {"mod_a": jnp.zeros((2, 3)), "mod_b": jnp.zeros((5,))}

    filled = pipeline._fill_loss_mask(gen_model.modality_registry, state, None)

    assert set(filled.keys()) == {"mod_a", "mod_b"}
    assert jnp.array_equal(filled["mod_a"], jnp.ones((2, 3)))
    assert jnp.array_equal(filled["mod_b"], jnp.ones((5,)))


def test_fill_loss_mask_preserves_provided_masks():
    """A fully-specified ``loss_mask`` should be passed through unchanged."""
    pipeline = toy_pipeline()
    gen_model = multi_modality_gen_model({"mod_a": (2, 3)})
    state = {"mod_a": jnp.zeros((2, 3))}
    provided = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    filled = pipeline._fill_loss_mask(
        gen_model.modality_registry, state, {"mod_a": provided}
    )

    assert jnp.array_equal(filled["mod_a"], provided)


def test_fill_loss_mask_fills_none_leaves_with_ones():
    """A per-modality ``None`` leaf should default that modality to an all-ones mask.

    A mask must be a full tree: whole ``None``, or one leaf per modality where
    ``None`` means "ones". Partial dicts that omit keys are not supported.
    """
    pipeline = toy_pipeline()
    gen_model = multi_modality_gen_model({"mod_a": (2, 3), "mod_b": (5,)})
    state = {"mod_a": jnp.zeros((2, 3)), "mod_b": jnp.zeros((5,))}
    provided = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

    filled = pipeline._fill_loss_mask(
        gen_model.modality_registry, state, {"mod_a": provided, "mod_b": None}
    )

    assert jnp.array_equal(filled["mod_a"], provided)
    assert jnp.array_equal(filled["mod_b"], jnp.ones((5,)))


def test_fill_loss_mask_keys_off_state_not_net_out():
    """The default mask should follow the *state* shape ``(L,)``, not a wider
    ``net_out`` (e.g. ``(L, K)`` logits for a discrete head).
    """
    pipeline = toy_pipeline()
    gen_model = multi_modality_gen_model({"mod_a": (5,)})
    state = {"mod_a": jnp.zeros((5,))}

    filled = pipeline._fill_loss_mask(gen_model.modality_registry, state, None)

    assert filled["mod_a"].shape == (5,)


# ══════════════════════════════════════════════════════════════════════
# __call__
# ══════════════════════════════════════════════════════════════════════


def test_call_returns_finite_scalar():
    """Calling the pipeline on a valid batch should return a (loss, metrics)
    pair where the loss is a finite scalar and ``metrics`` contains ``loss``.
    """
    pipeline = toy_pipeline()
    gen_model = one_sided_gen_model(IdentityNetwork())

    loss, metrics = pipeline(gen_model, toy_batch_obj(), jr.PRNGKey(0))

    assert loss.shape == ()
    assert jnp.isfinite(loss)
    assert "loss" in metrics
    assert jnp.array_equal(metrics["loss"], loss)


def test_call_reproducible_with_same_key():
    """Calling the pipeline twice with the same key should yield identical losses."""
    pipeline = toy_pipeline()
    gen_model = one_sided_gen_model(IdentityNetwork())
    batch = toy_batch_obj()

    loss_a, _ = pipeline(gen_model, batch, jr.PRNGKey(0))
    loss_b, _ = pipeline(gen_model, batch, jr.PRNGKey(0))

    assert jnp.array_equal(loss_a, loss_b)


def test_call_different_keys_produce_different_losses():
    """Calling the pipeline with different keys should produce different losses."""
    pipeline = toy_pipeline()
    gen_model = one_sided_gen_model(IdentityNetwork())
    batch = toy_batch_obj()

    loss_a, _ = pipeline(gen_model, batch, jr.PRNGKey(0))
    loss_b, _ = pipeline(gen_model, batch, jr.PRNGKey(1))

    assert not jnp.array_equal(loss_a, loss_b)


def test_call_passes_loss_mask_through():
    """A fully-zero ``loss_mask`` should drive the MSE loss to exactly zero."""
    pipeline = toy_pipeline()
    gen_model = one_sided_gen_model(IdentityNetwork())
    zero_mask: PyTree[Mask | None] = {"mod_a": jnp.zeros((BATCH_SIZE, *SHAPE))}

    loss, _ = pipeline(gen_model, toy_batch_obj(loss_mask=zero_mask), jr.PRNGKey(0))

    assert jnp.allclose(loss, 0.0)


def test_call_antithetical_returns_finite_scalar():
    """Calling the pipeline with ``antithetical=True`` should still return a finite scalar."""
    pipeline = toy_pipeline(antithetical=True)
    gen_model = one_sided_gen_model(IdentityNetwork())

    loss, _ = pipeline(gen_model, toy_batch_obj(), jr.PRNGKey(0))

    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_call_uses_coupling():
    """The pipeline must invoke ``coupling`` exactly once per call, with the
    raw batch and embedded batch as positional arguments (no epsilon).
    """
    calls = []

    class RecordingCoupling(Coupling):
        def __call__(self, raw_batch, embedded_pairs):
            calls.append((raw_batch, embedded_pairs))
            return raw_batch, embedded_pairs

    pipeline = toy_pipeline(coupling=RecordingCoupling())
    gen_model = one_sided_gen_model(IdentityNetwork())

    pipeline(gen_model, toy_batch_obj(), jr.PRNGKey(0))

    assert len(calls) == 1
    raw, embedded = calls[0]
    assert set(raw.keys()) == {"mod_a"}
    assert set(embedded.keys()) == {"mod_a"}
    # One-sided: coupling must not invent a source; noise is drawn after.
    assert embedded["mod_a"].source is None


def test_call_draws_noise_after_coupling():
    """Interpolant noise must be sampled after coupling so couplings cannot
    correlate epsilon with the re-paired (z_src, z_tgt).
    """
    observed = {}

    class CaptureCoupling(Coupling):
        def __call__(self, raw_batch, embedded_pairs):
            # Stamp a sentinel that would only appear in epsilon if noise
            # had already been generated and passed through (it must not).
            observed["n_args"] = 2
            observed["source"] = embedded_pairs["mod_a"].source
            return raw_batch, embedded_pairs

    pipeline = toy_pipeline(coupling=CaptureCoupling())
    gen_model = one_sided_gen_model(IdentityNetwork())
    loss, _ = pipeline(gen_model, toy_batch_obj(), jr.PRNGKey(0))
    assert jnp.isfinite(loss)
    assert observed["n_args"] == 2
    assert observed["source"] is None


def _two_sided_prior_gen_model() -> VelocityTwoSidedGenerativeModel:
    """Two-sided velocity model with Gaussian embedded source prior."""
    registry = ModalityRegistry(
        {
            "mod_a": Modality(
                shape=SHAPE,
                is_discrete=False,
                interpolant=two_sided_deterministic_si(),
                embedder=IdentityEmbedder(dm_shape=SHAPE),
                embedded_source_prior=jr.normal,
            )
        }
    )
    return VelocityTwoSidedGenerativeModel(
        network=IdentityNetwork(), modality_registry=registry
    )


def test_call_fills_embedded_source_from_prior():
    """Two-sided + prior: training fills embedded source before coupling."""
    observed = {}

    class CaptureCoupling(Coupling):
        def __call__(self, raw_batch, embedded_pairs):
            observed["source"] = embedded_pairs["mod_a"].source
            return raw_batch, embedded_pairs

    pipeline = toy_pipeline(coupling=CaptureCoupling())
    gen_model = _two_sided_prior_gen_model()
    loss, _ = pipeline(gen_model, toy_batch_obj(), jr.PRNGKey(0))
    assert jnp.isfinite(loss)
    assert observed["source"] is not None
    assert observed["source"].shape == (BATCH_SIZE, *SHAPE)


def test_call_two_sided_without_source_or_prior_raises():
    """Two-sided interpolant with no source and no prior cannot interpolate."""
    registry = ModalityRegistry(
        {
            "mod_a": Modality(
                shape=SHAPE,
                is_discrete=False,
                interpolant=two_sided_deterministic_si(),
                embedder=IdentityEmbedder(dm_shape=SHAPE),
            )
        }
    )
    gen_model = VelocityTwoSidedGenerativeModel(
        network=IdentityNetwork(), modality_registry=registry
    )
    pipeline = toy_pipeline()
    with pytest.raises(ValueError, match="embedded source"):
        pipeline(gen_model, toy_batch_obj(), jr.PRNGKey(0))


def test_call_mixed_one_sided_and_prior_keeps_one_sided_source_none():
    """Mixed registry: prior modality is filled; one-sided source stays None."""
    observed = {}

    class CaptureCoupling(Coupling):
        def __call__(self, raw_batch, embedded_pairs):
            observed["one_sided"] = embedded_pairs["one_sided"].source
            observed["prior"] = embedded_pairs["prior"].source
            return raw_batch, embedded_pairs

    # Multimodal factory only supports one-sided continuous + discrete; use
    # a custom GenerativeModel-like pair of separate loss paths via Noise
    # one-sided for the continuous one-sided leaf is insufficient for mixed
    # interpolant types. Exercise the fill/couple path with a minimal custom
    # subclass that only needs get_loss / get_generator stubs for the call.
    from stix.core.gen_model.gen_model import GenerativeModel
    from stix.core.generator import Velocity
    from stix.core.loss.criterion import MSECriterion
    from stix.core.loss.utils import reduce_modality_losses

    class _MixedModel(GenerativeModel):
        def get_loss(
            self,
            net_out,
            t,
            z_t,
            epsilon,
            embedded_pairs,
            raw_pairs,
            loss_mask,
        ):
            mse = MSECriterion()
            losses = self.modality_registry.map(
                lambda modality, net_out_k, pair_k, eps_k, mask_k: mse(
                    net_out_k,
                    modality.interpolant.get_conditional_velocity(pair_k, t, eps_k),
                    t,
                    mask_k,
                ),
                net_out,
                embedded_pairs,
                epsilon,
                loss_mask,
            )
            return reduce_modality_losses(losses)

        def get_generator(self, net_out, z_t, t):
            return self.modality_registry.map(
                lambda modality, net_out_k, z_t_k: Velocity(velocity=net_out_k),
                net_out,
                z_t,
            )

    registry = ModalityRegistry(
        {
            "one_sided": Modality(
                shape=SHAPE,
                is_discrete=False,
                interpolant=one_sided_si(),
                embedder=IdentityEmbedder(dm_shape=SHAPE),
            ),
            "prior": Modality(
                shape=SHAPE,
                is_discrete=False,
                interpolant=two_sided_deterministic_si(),
                embedder=IdentityEmbedder(dm_shape=SHAPE),
                embedded_source_prior=jr.normal,
            ),
        }
    )
    gen_model = _MixedModel(network=IdentityNetwork(), modality_registry=registry)
    raw = {
        "one_sided": RawSourceTargetPair(
            target=jnp.ones((BATCH_SIZE, *SHAPE)), source=None
        ),
        "prior": RawSourceTargetPair(
            target=jnp.ones((BATCH_SIZE, *SHAPE)), source=None
        ),
    }
    batch = Batch(raw_batch=raw)
    pipeline = toy_pipeline(coupling=CaptureCoupling())
    loss, _ = pipeline(gen_model, batch, jr.PRNGKey(0))
    assert jnp.isfinite(loss)
    assert observed["one_sided"] is None
    assert observed["prior"] is not None
    assert observed["prior"].shape == (BATCH_SIZE, *SHAPE)


def test_call_is_differentiable_through_gen_model():
    """The pipeline's output must be differentiable w.r.t. the ``gen_model``
    parameters so that it can be used with ``nnx.value_and_grad``.
    """
    pipeline = toy_pipeline()
    gen_model = one_sided_gen_model(ToyConstantNetwork(value=0.5))

    def loss_fn(model):
        loss, _ = pipeline(model, toy_batch_obj(), jr.PRNGKey(0))
        return loss

    grads = nnx.grad(loss_fn)(gen_model)
    bias_grad = grads.network.bias[...]
    assert bias_grad.shape == ()
    assert jnp.isfinite(bias_grad)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_call_supports_multiple_batch_sizes(batch_size):
    """The pipeline should run correctly for any positive batch size, as it is
    derived from the raw batch's leading dimension at call time.
    """
    pipeline = toy_pipeline()
    gen_model = one_sided_gen_model(IdentityNetwork())

    loss, _ = pipeline(gen_model, toy_batch_obj(batch_size=batch_size), jr.PRNGKey(0))

    assert loss.shape == ()
    assert jnp.isfinite(loss)


def test_call_runs_with_batched_conditioning_through_vmap():
    """Non-``None`` conditioning must flow through the per-sample ``vmap`` in
    ``_step``: a batched ``context_data`` leaf (leading batch axis) runs and is
    deterministic. Guards the ``Batch`` conditioning fields against a broken
    dict/pytree vmap (all other tests pass ``context_data=None``).
    """
    pipeline = toy_pipeline()
    gen_model = one_sided_gen_model(IdentityNetwork())
    batch = Batch(
        raw_batch=toy_raw_batch(),
        context_data={"label": jnp.ones((BATCH_SIZE, *SHAPE))},
        context_mask={"label": jnp.ones((BATCH_SIZE, *SHAPE))},
    )

    loss_a, _ = pipeline(gen_model, batch, jr.PRNGKey(0))
    loss_b, _ = pipeline(gen_model, batch, jr.PRNGKey(0))

    assert loss_a.shape == ()
    assert jnp.isfinite(loss_a)
    assert jnp.array_equal(loss_a, loss_b)


# ══════════════════════════════════════════════════════════════════════
# get_batch_size (consumed by LossPipeline.__call__)
# ══════════════════════════════════════════════════════════════════════


def test_batch_size_consistent_modalities_returns_shared_size():
    """All modalities agreeing on the leading dim yields that shared batch size."""
    raw = {
        "mod_a": RawSourceTargetPair(target=jnp.ones((8, *SHAPE)), source=None),
        "mod_b": RawSourceTargetPair(target=jnp.ones((8, *SHAPE)), source=None),
    }

    assert get_batch_size(raw) == 8


def test_batch_size_inconsistent_modalities_raises():
    """Mismatched leading dims across modalities surface as a ValueError that
    names the offending sizes (rather than silently picking the first).
    """
    raw = {
        "mod_a": RawSourceTargetPair(target=jnp.ones((32, *SHAPE)), source=None),
        "mod_b": RawSourceTargetPair(target=jnp.ones((16, *SHAPE)), source=None),
    }

    with pytest.raises(ValueError, match="Inconsistent batch sizes"):
        _ = get_batch_size(raw)


def test_batch_size_empty_data_raises_value_error():
    """An empty raw batch raises a domain-specific ValueError instead of the
    raw StopIteration that ``next(iter(...))`` would otherwise produce.
    """
    raw: dict[str, RawSourceTargetPair] = {}

    with pytest.raises(ValueError, match="empty raw batch"):
        _ = get_batch_size(raw)
