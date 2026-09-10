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

"""Tests for ContinuousStochasticInterpolant / ContinuousDeterministicInterpolant."""

import jax.numpy as jnp
import jax.random as jr
import pytest

from stix.core.interpolant.continuous_interpolant import (
    ContinuousDeterministicInterpolant,
    ContinuousInterpolant,
    ContinuousStochasticInterpolant,
)
from stix.core.interpolant.discrete_interpolant import DiscreteInterpolant
from stix.core.interpolant.interpolant import Interpolant, OneSidedInterpolant
from stix.core.interpolant.linear_interpolant import (
    LinearStochasticInterpolant,
    OneSidedLinearStochasticInterpolant,
)
from stix.core.interpolant.standard_interpolants.discrete_diffusion import (
    MaskDiscreteInterpolant,
    UniformDiscreteInterpolant,
)
from stix.typing import EmbeddedSourceTargetPair


class _ConcreteOneSidedLinear(OneSidedLinearStochasticInterpolant):
    """Minimal concrete subclass for constructing custom schedules in tests."""


class _ToyDeterministicInterpolant(ContinuousDeterministicInterpolant):
    def interpolant_fn(self, embedded_pairs: EmbeddedSourceTargetPair, t):
        source = embedded_pairs.source if embedded_pairs.source is not None else 0.0
        return (1 - t) * source + t * embedded_pairs.target

    def sample_initial_state(self, z_src, epsilon):
        return 0.0 if z_src is None else z_src


class _ToyContinuousStochasticInterpolant(ContinuousStochasticInterpolant):
    def interpolant_fn(self, embedded_pairs: EmbeddedSourceTargetPair, t):
        source = embedded_pairs.source if embedded_pairs.source is not None else 0.0
        return (1 - t) * source + t * embedded_pairs.target

    def sample_initial_state(self, z_src, epsilon):
        source = 0.0 if z_src is None else z_src
        return source + self.gamma_fn(jnp.asarray(0.0)) * epsilon


@pytest.fixture()
def two_sided_pairs() -> tuple[EmbeddedSourceTargetPair, jnp.ndarray, jnp.ndarray]:
    src_vec = jnp.array([2.0, 3.0])
    tgt_vec = jnp.array([7.0, 11.0])
    return EmbeddedSourceTargetPair(target=tgt_vec, source=src_vec), src_vec, tgt_vec


def test_wrong_gamma_schedule_raises():
    with pytest.raises(
        ValueError, match="ContinuousStochasticInterpolant requires non-zero gamma_t"
    ):
        LinearStochasticInterpolant(
            gamma_fn=lambda t: jnp.zeros_like(t),
            alpha_fn=lambda t: 1 - t,
            beta_fn=lambda t: t,
        )


def test_lin_one_sided_si_gamma_zero_at_t_0_raises():
    with pytest.raises(
        ValueError,
        match="ContinuousOneSidedStochasticInterpolant requires non-zero gamma_t at t=0",
    ):
        _ConcreteOneSidedLinear(
            gamma_fn=lambda t: t,
            beta_fn=lambda t: t,
        )


def test_si_interpolate_with_noise_schedule(two_sided_pairs):
    pairs, src_vec, tgt_vec = two_sided_pairs
    interpolant = _ToyContinuousStochasticInterpolant(gamma_fn=lambda t: t + 1.0)
    epsilon = jnp.array([2.0, -1.0])
    t = jnp.array(0.5)
    out = interpolant.interpolate(pairs, t, epsilon)
    expected = (1 - t) * src_vec + t * tgt_vec + (t + 1.0) * epsilon
    assert jnp.allclose(out, expected)


def test_interpolant_interpolate_with_noise_schedule(two_sided_pairs):
    pairs, src_vec, tgt_vec = two_sided_pairs
    interpolant = _ToyDeterministicInterpolant()
    epsilon = jnp.array([2.0, -1.0])
    t = jnp.array(0.5)
    out = interpolant.interpolate(pairs, t, epsilon)
    expected = (1 - t) * src_vec + t * tgt_vec
    assert jnp.allclose(out, expected)
    epsilon_new = jr.normal(jr.PRNGKey(42), (2,))
    out_new = interpolant.interpolate(pairs, t, epsilon_new)
    assert jnp.allclose(out, out_new)


def test_si_get_conditional_velocity_field(two_sided_pairs):
    pairs, src_vec, tgt_vec = two_sided_pairs
    interpolant = _ToyContinuousStochasticInterpolant(gamma_fn=lambda t: t**2)
    t = jnp.array(0.3)
    epsilon = jnp.array([0.0, 0.0])
    out = interpolant.get_conditional_velocity_field(pairs, t, epsilon)
    expected = tgt_vec - src_vec
    assert jnp.allclose(out, expected)
    epsilon_new = jr.normal(jr.PRNGKey(42), (2,))
    out_new = interpolant.get_conditional_velocity_field(pairs, t, epsilon_new)
    assert jnp.allclose(out, out_new)


def test_interpolant_get_conditional_velocity_field(two_sided_pairs):
    pairs, src_vec, tgt_vec = two_sided_pairs
    interpolant = _ToyDeterministicInterpolant()
    t = jnp.array(0.3)
    epsilon = jnp.array([0.0, 0.0])
    out = interpolant.get_conditional_velocity_field(pairs, t, epsilon)
    expected = tgt_vec - src_vec
    assert jnp.allclose(out, expected)
    epsilon_new = jr.normal(jr.PRNGKey(42), (2,))
    out_new = interpolant.get_conditional_velocity_field(pairs, t, epsilon_new)
    assert jnp.allclose(out, out_new)


def test_si_get_conditional_score(two_sided_pairs):
    pairs, src_vec, tgt_vec = two_sided_pairs
    interpolant = _ToyContinuousStochasticInterpolant(gamma_fn=lambda t: t**2)
    t = jnp.array(0.3)
    epsilon = jnp.array([1.2, -0.5])
    out = interpolant.get_conditional_score(pairs, t, epsilon)
    expected = -1 * epsilon / (t**2)
    assert jnp.allclose(out, expected)


def test_deterministic_interpolant_has_no_score(two_sided_pairs):
    pairs, _, _ = two_sided_pairs
    interpolant = _ToyDeterministicInterpolant()
    assert not hasattr(interpolant, "get_conditional_score")


def test_si_get_conditional_velocity(two_sided_pairs):
    pairs, src_vec, tgt_vec = two_sided_pairs
    interpolant = _ToyContinuousStochasticInterpolant(gamma_fn=lambda t: t**2)
    t = jnp.array(0.3)
    epsilon = jnp.array([1.2, -0.5])
    out = interpolant.get_conditional_velocity(pairs, t, epsilon)
    expected = (tgt_vec - src_vec) + 2 * t * epsilon
    assert jnp.allclose(out, expected)


def test_interpolant_get_conditional_velocity(two_sided_pairs):
    pairs, src_vec, tgt_vec = two_sided_pairs
    interpolant = _ToyDeterministicInterpolant()
    t = jnp.array(0.3)
    epsilon = jnp.array([1.2, -0.5])
    out = interpolant.get_conditional_velocity(pairs, t, epsilon)
    expected = tgt_vec - src_vec
    assert jnp.allclose(out, expected)
    epsilon_new = jr.normal(jr.PRNGKey(42), (2,))
    out_new = interpolant.get_conditional_velocity(pairs, t, epsilon_new)
    assert jnp.allclose(out, out_new)


def test_hierarchy_axes():
    continuous_si = _ToyContinuousStochasticInterpolant(gamma_fn=lambda t: t + 1.0)
    assert isinstance(continuous_si, ContinuousInterpolant)
    assert isinstance(continuous_si, ContinuousStochasticInterpolant)
    assert isinstance(continuous_si, Interpolant)

    continuous_det = _ToyDeterministicInterpolant()
    assert isinstance(continuous_det, ContinuousDeterministicInterpolant)
    assert isinstance(continuous_det, ContinuousInterpolant)
    assert not isinstance(continuous_det, ContinuousStochasticInterpolant)

    mask = MaskDiscreteInterpolant(num_categories=4, kappa_fn=lambda t: t)
    uniform = UniformDiscreteInterpolant(num_categories=4, kappa_fn=lambda t: t)
    for discrete in (mask, uniform):
        assert isinstance(discrete, DiscreteInterpolant)
        assert isinstance(discrete, Interpolant)
        assert isinstance(discrete, OneSidedInterpolant)
        assert not isinstance(discrete, ContinuousInterpolant)
        assert not isinstance(discrete, ContinuousStochasticInterpolant)
        assert not isinstance(discrete, ContinuousDeterministicInterpolant)

    assert issubclass(DiscreteInterpolant, Interpolant)
    assert not issubclass(DiscreteInterpolant, OneSidedInterpolant)
    assert not issubclass(DiscreteInterpolant, ContinuousInterpolant)
    assert issubclass(ContinuousStochasticInterpolant, ContinuousInterpolant)
    assert issubclass(ContinuousDeterministicInterpolant, ContinuousInterpolant)
