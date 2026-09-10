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

"""Tests for gen_model utility functions."""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from conftest import (
    SHAPE,
    build_one_sided_registry,
    one_sided_si,
    two_sided_deterministic_si,
)

from stix.core.embedder import IdentityEmbedder
from stix.core.gen_model.utils import (
    assert_two_sided_sources_available,
    fill_embedded_source_from_prior,
)
from stix.core.modality import Modality, ModalityRegistry
from stix.typing import EmbeddedSourceTargetPair

# ══════════════════════════════════════════════════════════════════════
# ModalityRegistry.sample_initial_state
# ══════════════════════════════════════════════════════════════════════


def test_sample_initial_state_shape_and_scale():
    """``sample_initial_state`` returns per-modality noise of shape
    ``(num_samples, *embedding_shape)``, scaled by ``gamma(0)``. The conftest
    one-sided interpolant uses ``gamma(t) = 1 - t`` (so ``gamma(0) = 1``) and an
    ``IdentityEmbedder`` (embedding_shape == SHAPE).
    """
    registry = build_one_sided_registry()
    num_samples = 4096
    z_init = registry.sample_initial_state(jax.random.key(0), num_samples=num_samples)

    assert set(z_init) == {"mod_a"}
    assert z_init["mod_a"].shape == (num_samples, *SHAPE)
    # gamma(0) = 1 => standard Gaussian.
    assert jnp.abs(z_init["mod_a"].std() - 1.0) < 0.05


def test_sample_initial_state_is_deterministic_in_key():
    """The same key yields the same draw; different keys differ."""
    registry = build_one_sided_registry()
    a = registry.sample_initial_state(jax.random.key(0), num_samples=8)["mod_a"]
    b = registry.sample_initial_state(jax.random.key(0), num_samples=8)["mod_a"]
    c = registry.sample_initial_state(jax.random.key(1), num_samples=8)["mod_a"]
    assert jnp.array_equal(a, b)
    assert not jnp.array_equal(a, c)


def _two_sided_prior_registry(*, prior=jr.normal) -> ModalityRegistry:
    """Two-sided deterministic modality with an embedded source prior."""
    return ModalityRegistry(
        {
            "mod_a": Modality(
                shape=SHAPE,
                is_discrete=False,
                interpolant=two_sided_deterministic_si(),
                embedder=IdentityEmbedder(dm_shape=SHAPE),
                embedded_source_prior=prior,
            )
        }
    )


def _two_sided_data_registry() -> ModalityRegistry:
    """Two-sided deterministic modality with no prior (data→data)."""
    return ModalityRegistry(
        {
            "mod_a": Modality(
                shape=SHAPE,
                is_discrete=False,
                interpolant=two_sided_deterministic_si(),
                embedder=IdentityEmbedder(dm_shape=SHAPE),
            )
        }
    )


def test_sample_initial_state_from_embedded_source_prior():
    """Two-sided + prior draws z_src, then z_0 = interpolant.sample_initial_state."""
    registry = _two_sided_prior_registry()
    num_samples = 8
    key = jax.random.key(0)
    z_init = registry.sample_initial_state(key, num_samples=num_samples)
    # Per-modality key is split into (src, noise); deterministic FM has
    # z_0 = z_src, so the draw matches the prior on the source key.
    expected_key = registry.split_and_project_key(key)["mod_a"]
    src_key, _noise_key = jr.split(expected_key)
    expected = jr.normal(src_key, (num_samples, *SHAPE))
    assert jnp.array_equal(z_init["mod_a"], expected)


def test_sample_initial_state_two_sided_without_source_or_prior_raises():
    """Data→data two-sided modalities need a raw source tree."""
    registry = _two_sided_data_registry()
    with pytest.raises(TypeError, match="raw source or embedded_source_prior"):
        registry.sample_initial_state(jax.random.key(0), num_samples=4)


def test_sample_initial_state_from_raw_source():
    """Data→data: embed the raw source and pass it to the interpolant."""
    registry = _two_sided_data_registry()
    raw = jnp.arange(6, dtype=jnp.float32).reshape(2, *SHAPE)
    z_init = registry.sample_initial_state(jax.random.key(0), raw_source={"mod_a": raw})
    assert jnp.array_equal(z_init["mod_a"], raw)


def test_sample_initial_state_infers_num_samples_from_raw_source():
    registry = _two_sided_data_registry()
    raw = jnp.ones((5, *SHAPE))
    z_init = registry.sample_initial_state(jax.random.key(0), raw_source={"mod_a": raw})
    assert z_init["mod_a"].shape[0] == 5


def test_sample_initial_state_num_samples_mismatch_raises():
    registry = _two_sided_data_registry()
    raw = jnp.ones((3, *SHAPE))
    with pytest.raises(ValueError, match="does not match"):
        registry.sample_initial_state(
            jax.random.key(0), raw_source={"mod_a": raw}, num_samples=4
        )


def test_sample_initial_state_requires_num_samples_when_no_source():
    registry = build_one_sided_registry()
    with pytest.raises(ValueError, match="requires num_samples"):
        registry.sample_initial_state(jax.random.key(0))


def test_sample_initial_state_raw_source_on_one_sided_raises():
    registry = build_one_sided_registry()
    raw = jnp.ones((2, *SHAPE))
    with pytest.raises(ValueError, match="one-sided interpolant"):
        registry.sample_initial_state(jax.random.key(0), raw_source={"mod_a": raw})


def test_sample_initial_state_raw_source_wins_over_prior():
    """An explicit raw source is used even when a prior is set."""
    registry = _two_sided_prior_registry()
    raw = jnp.full((2, *SHAPE), 7.0)
    z_init = registry.sample_initial_state(jax.random.key(0), raw_source={"mod_a": raw})
    assert jnp.array_equal(z_init["mod_a"], raw)


def test_sample_initial_state_mixed_one_sided_and_prior():
    """Mixed registries dispatch per modality (one-sided vs prior)."""
    registry = ModalityRegistry(
        {
            "noise": Modality(
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
    z_init = registry.sample_initial_state(jax.random.key(0), num_samples=4)
    assert z_init["noise"].shape == (4, *SHAPE)
    assert z_init["prior"].shape == (4, *SHAPE)


def test_sample_initial_state_mixed_one_sided_and_raw_source():
    """One-sided leaf stays None; two-sided leaf embeds the raw source."""
    registry = ModalityRegistry(
        {
            "noise": Modality(
                shape=SHAPE,
                is_discrete=False,
                interpolant=one_sided_si(),
                embedder=IdentityEmbedder(dm_shape=SHAPE),
            ),
            "data": Modality(
                shape=SHAPE,
                is_discrete=False,
                interpolant=two_sided_deterministic_si(),
                embedder=IdentityEmbedder(dm_shape=SHAPE),
            ),
        }
    )
    raw = jnp.full((3, *SHAPE), 2.0)
    z_init = registry.sample_initial_state(
        jax.random.key(0),
        raw_source={"noise": None, "data": raw},
    )
    assert z_init["noise"].shape == (3, *SHAPE)
    assert jnp.array_equal(z_init["data"], raw)


def test_fill_embedded_source_from_prior_fills_missing_source():
    """Prior fill sets source when unset; leaves existing sources alone."""
    registry = _two_sided_prior_registry()
    batch = {
        "mod_a": EmbeddedSourceTargetPair(target=jnp.ones((2, *SHAPE)), source=None)
    }
    filled = fill_embedded_source_from_prior(registry, batch, jr.PRNGKey(0))
    assert filled["mod_a"].source is not None
    assert filled["mod_a"].source.shape == (2, *SHAPE)

    existing = jnp.full((2, *SHAPE), 7.0)
    batch_existing = {
        "mod_a": EmbeddedSourceTargetPair(target=jnp.ones((2, *SHAPE)), source=existing)
    }
    kept = fill_embedded_source_from_prior(registry, batch_existing, jr.PRNGKey(0))
    assert jnp.array_equal(kept["mod_a"].source, existing)


def test_fill_embedded_source_leaves_one_sided_source_none():
    """One-sided modalities without a prior keep source=None after fill."""
    registry = build_one_sided_registry()
    batch = {
        "mod_a": EmbeddedSourceTargetPair(target=jnp.ones((2, *SHAPE)), source=None)
    }
    filled = fill_embedded_source_from_prior(registry, batch, jr.PRNGKey(0))
    assert filled["mod_a"].source is None


def test_fill_embedded_source_two_sided_without_source_or_prior_raises():
    """Two-sided interpolant with no source and no prior cannot be filled."""
    registry = _two_sided_data_registry()
    batch = {
        "mod_a": EmbeddedSourceTargetPair(target=jnp.ones((2, *SHAPE)), source=None)
    }
    with pytest.raises(ValueError, match="embedded source"):
        fill_embedded_source_from_prior(registry, batch, jr.PRNGKey(0))


def test_assert_two_sided_sources_available_raises_when_source_missing():
    """Two-sided + source=None is rejected; one-sided and present sources pass."""
    missing = {
        "mod_a": EmbeddedSourceTargetPair(target=jnp.ones((2, *SHAPE)), source=None)
    }
    with pytest.raises(ValueError, match="embedded source"):
        assert_two_sided_sources_available(_two_sided_data_registry(), missing)

    present = {
        "mod_a": EmbeddedSourceTargetPair(
            target=jnp.ones((2, *SHAPE)), source=jnp.zeros((2, *SHAPE))
        )
    }
    assert_two_sided_sources_available(_two_sided_data_registry(), present)
    assert_two_sided_sources_available(build_one_sided_registry(), missing)
