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

"""Tests for UniformTimeSampler."""

import jax
import jax.numpy as jnp
import jax.random as jr

from stix.training.time_sampler import UniformTimeSampler


def test_default_bounds():
    """The default ``UniformTimeSampler`` should produce samples within the
    unit interval ``[0, 1]`` regardless of the PRNG key.
    """
    sampler = UniformTimeSampler()
    keys = jr.split(jr.PRNGKey(0), 1000)
    samples = jax.vmap(sampler)(keys)
    assert jnp.all(samples >= 0.0)
    assert jnp.all(samples <= 1.0)


def test_custom_bounds_respected():
    """A ``UniformTimeSampler`` configured with custom ``t_min`` and ``t_max``
    bounds should never produce samples outside that interval.
    """
    sampler = UniformTimeSampler(t_min=0.2, t_max=0.8)
    keys = jr.split(jr.PRNGKey(42), 1000)
    samples = jax.vmap(sampler)(keys)
    assert jnp.all(samples >= 0.2)
    assert jnp.all(samples <= 0.8)


def test_samples_cover_range():
    """Drawing many samples from ``UniformTimeSampler`` over ``[0, 1]`` should
    cover essentially the full interval (min near 0, max near 1).
    """
    sampler = UniformTimeSampler(t_min=0.0, t_max=1.0)
    keys = jr.split(jr.PRNGKey(1), 10_000)
    samples = jax.vmap(sampler)(keys)
    assert 0.0 <= samples.min() < 0.01
    assert 1.0 >= samples.max() > 0.99


def test_output_is_scalar():
    """Each call to ``UniformTimeSampler`` should return a scalar tensor of
    shape ``()``.
    """
    sampler = UniformTimeSampler()
    sample = sampler(jr.PRNGKey(1))
    assert sample.shape == ()


def test_output_dtype_is_float32():
    """``UniformTimeSampler`` should produce samples with dtype ``float32``."""
    sampler = UniformTimeSampler()
    sample = sampler(jr.PRNGKey(2))
    assert sample.dtype == jnp.float32


def test_reproducible_with_same_key():
    """Calling ``UniformTimeSampler`` with the same PRNG key twice should
    produce identical samples.
    """
    sampler = UniformTimeSampler()
    key = jr.PRNGKey(99)
    assert jnp.array_equal(sampler(key), sampler(key))


def test_different_keys_produce_different_samples():
    """Calling ``UniformTimeSampler`` with two different PRNG keys should
    produce different samples.
    """
    sampler = UniformTimeSampler()
    key1, key2 = jr.split(jr.PRNGKey(0))
    assert not jnp.array_equal(sampler(key1), sampler(key2))
