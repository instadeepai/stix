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

"""Tests for discrete DFM interpolants (masked / uniform).

Covers :class:`MaskDiscreteInterpolant` and :class:`UniformDiscreteInterpolant`:

- :math:`I_t` equals a sample from the selected mixture conditional :math:`w_J`;
- the empirical interpolation law matches the two-point mixture;
- the two-point velocity reduces to
  :math:`\\dot\\kappa/(1-\\kappa)\\,(p_{\\mathrm{tgt}\\mid t}-\\delta_z)`;
- forward velocities conserve mass and freeze unmasked mask-path tokens;
- the corrector mix :math:`\\alpha\\hat u-\\beta\\check u` with
  :math:`\\alpha-\\beta=1` vs :math:`\\alpha=\\beta`.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from stix.core.interpolant.continuous_interpolant import ContinuousInterpolant
from stix.core.interpolant.discrete_interpolant import DiscreteInterpolant
from stix.core.interpolant.interpolant import Interpolant, OneSidedInterpolant
from stix.core.interpolant.standard_interpolants.discrete_diffusion import (
    MaskDiscreteInterpolant,
    UniformDiscreteInterpolant,
)
from stix.typing import EmbeddedSourceTargetPair

K = 4  # number of data categories


def _linear_kappa(t):
    """Linear mixing schedule (kappa_0 = 0, kappa_1 = 1)."""
    return t


@pytest.fixture
def mask_interpolant() -> MaskDiscreteInterpolant:
    return MaskDiscreteInterpolant(num_categories=K, kappa_fn=_linear_kappa)


@pytest.fixture
def uniform_interpolant() -> UniformDiscreteInterpolant:
    return UniformDiscreteInterpolant(num_categories=K, kappa_fn=_linear_kappa)


def _rate_matrix(interpolant, posterior, t, *, backward: bool = False):
    """Stack a single-position velocity over every start state into a matrix.

    ``rate_matrix[z, y]`` is the velocity ``z -> y``. Shape
    ``(num_states, num_states)``.
    """
    num_states = interpolant.num_states
    rows = [
        interpolant.rates_from_target_posterior(
            posterior, jax.nn.one_hot(z, num_states), t, backward=backward
        )
        for z in range(num_states)
    ]
    return jnp.stack(rows, axis=0)


def _corrector_rate_matrix(interpolant, posterior, t, lam):
    """Stack :math:`(1+\\lambda)\\hat u + \\lambda\\check u` over every start state."""
    u_hat = _rate_matrix(interpolant, posterior, t, backward=False)
    u_check = _rate_matrix(interpolant, posterior, t, backward=True)
    return (1.0 + lam) * u_hat + lam * u_check


def _target_pairs(interpolant, target_cat, n):
    """Build one-hot target-only pairs (source is implicit in the noise)."""
    num_states = interpolant.num_states
    x1 = jax.nn.one_hot(jnp.full((n,), target_cat, dtype=jnp.int32), num_states)
    return EmbeddedSourceTargetPair(target=x1, source=None)


# ── Basics ──


def test_num_states(mask_interpolant, uniform_interpolant):
    assert mask_interpolant.num_states == K + 1
    assert uniform_interpolant.num_states == K
    assert mask_interpolant.mask_index == K
    assert mask_interpolant.num_components == 2
    assert uniform_interpolant.num_components == 2


def test_not_continuous_interpolant(mask_interpolant, uniform_interpolant):
    assert isinstance(mask_interpolant, DiscreteInterpolant)
    assert isinstance(uniform_interpolant, DiscreteInterpolant)
    assert isinstance(mask_interpolant, Interpolant)
    assert isinstance(uniform_interpolant, Interpolant)
    assert isinstance(mask_interpolant, OneSidedInterpolant)
    assert isinstance(uniform_interpolant, OneSidedInterpolant)
    assert not isinstance(mask_interpolant, ContinuousInterpolant)
    assert not isinstance(uniform_interpolant, ContinuousInterpolant)
    assert not hasattr(mask_interpolant, "interpolant_fn")
    assert not hasattr(uniform_interpolant, "interpolant_fn")


def test_mask_sample_initial_state_all_mask(mask_interpolant):
    shape = (1000, mask_interpolant.num_states)
    epsilon = mask_interpolant.sample_noise(jr.PRNGKey(0), shape)
    z0 = mask_interpolant.sample_initial_state(None, epsilon)
    assert jnp.all(jnp.argmax(z0, axis=-1) == K)


def test_uniform_sample_initial_state_in_range(uniform_interpolant):
    shape = (5000, uniform_interpolant.num_states)
    epsilon = uniform_interpolant.sample_noise(jr.PRNGKey(0), shape)
    z0 = uniform_interpolant.sample_initial_state(None, epsilon)
    x0 = jnp.argmax(z0, axis=-1)
    assert jnp.all((x0 >= 0) & (x0 < K))
    assert jnp.unique(x0).shape[0] == K


def test_sample_initial_state_is_valid_one_hot(mask_interpolant, uniform_interpolant):
    for interpolant in (mask_interpolant, uniform_interpolant):
        shape = (128, interpolant.num_states)
        epsilon = interpolant.sample_noise(jr.PRNGKey(0), shape)
        z0 = interpolant.sample_initial_state(None, epsilon)
        assert z0.shape == shape
        assert jnp.allclose(jnp.sum(z0, axis=-1), 1.0)
        assert jnp.all((z0 == 0) | (z0 == 1))


# ── I_t equals the selected atom s_J ──


@pytest.mark.parametrize("t_val", [0.2, 0.5, 0.8])
def test_interpolate_equals_selected_atom(mask_interpolant, uniform_interpolant, t_val):
    t = jnp.asarray(t_val)
    n = 64
    target_cat = 2
    for interpolant in (mask_interpolant, uniform_interpolant):
        pairs = _target_pairs(interpolant, target_cat, n)
        epsilon = interpolant.sample_noise(jr.PRNGKey(1), (n, interpolant.num_states))
        conditionals = interpolant.mixture_conditionals(pairs)
        index = interpolant.draw_mixing_index(epsilon[..., 0], t)
        selected = jnp.take_along_axis(conditionals, index[..., None, None], axis=-2)[
            ..., 0, :
        ]
        epsilon_s = epsilon[..., 1]
        cdf = jnp.cumsum(selected, axis=-1)
        state_index = jnp.sum(epsilon_s[..., None] >= cdf, axis=-1)
        state_index = jnp.clip(state_index, 0, interpolant.num_states - 1)
        expected = jax.nn.one_hot(
            state_index, interpolant.num_states, dtype=selected.dtype
        )
        z_t = interpolant.interpolate(pairs, t, epsilon)
        assert jnp.allclose(z_t, expected)


# ── Interpolation law matches the mixture marginal ──


@pytest.mark.parametrize("t_val", [0.2, 0.5, 0.8])
def test_mask_interpolate_matches_marginal(mask_interpolant, t_val):
    t = jnp.asarray(t_val)
    n = 40000
    target_cat = 2
    num_states = mask_interpolant.num_states
    pairs = _target_pairs(mask_interpolant, target_cat, n)
    epsilon = mask_interpolant.sample_noise(jr.PRNGKey(1), (n, num_states))
    z_t = jnp.argmax(mask_interpolant.interpolate(pairs, t, epsilon), axis=-1)

    kappa = float(mask_interpolant.kappa_fn(t)[0])
    empirical_target = jnp.mean(z_t == target_cat)
    empirical_mask = jnp.mean(z_t == K)
    assert jnp.allclose(empirical_target, kappa, atol=0.01)
    assert jnp.allclose(empirical_mask, 1.0 - kappa, atol=0.01)


@pytest.mark.parametrize("t_val", [0.3, 0.7])
def test_uniform_interpolate_matches_marginal(uniform_interpolant, t_val):
    t = jnp.asarray(t_val)
    n = 40000
    target_cat = 1
    num_states = uniform_interpolant.num_states
    pairs = _target_pairs(uniform_interpolant, target_cat, n)
    epsilon = uniform_interpolant.sample_noise(jr.PRNGKey(1), (n, num_states))
    z_t = jnp.argmax(uniform_interpolant.interpolate(pairs, t, epsilon), axis=-1)

    kappa = float(uniform_interpolant.kappa_fn(t)[0])
    for category in range(K):
        expected = (1.0 - kappa) / K + kappa * float(category == target_cat)
        empirical = jnp.mean(z_t == category)
        assert jnp.allclose(empirical, expected, atol=0.01)


@pytest.mark.parametrize("t_val", [0.2, 0.5, 0.8])
def test_still_source_via_interpolate_matches_mixture(
    mask_interpolant, uniform_interpolant, t_val
):
    """Still-source positions are those with ``interpolate(t) == interpolate(0)``.

    For mask diffusion the source never equals a data target, so the empirical
    rate is :math:`1-\\kappa_t`. For uniform diffusion, source–target collisions
    make ``z_t == z_0`` with probability :math:`(1-\\kappa) + \\kappa/K`.
    """
    t = jnp.asarray(t_val)
    n = 20000
    target_cat = 0
    for interpolant, expected_rate in (
        (mask_interpolant, 1.0 - t_val),
        (uniform_interpolant, (1.0 - t_val) + t_val / K),
    ):
        num_states = interpolant.num_states
        pairs = _target_pairs(interpolant, target_cat, n)
        epsilon = interpolant.sample_noise(jr.PRNGKey(2), (n, num_states))
        z_0 = interpolant.interpolate(pairs, jnp.asarray(0.0), epsilon)
        z_t = interpolant.interpolate(pairs, t, epsilon)
        still_source = jnp.all(z_t == z_0, axis=-1)
        z_t_cat = jnp.argmax(z_t, axis=-1)
        assert jnp.all(z_t_cat[~still_source] == target_cat)
        assert jnp.allclose(
            jnp.mean(still_source.astype(jnp.float32)), expected_rate, atol=0.02
        )


# ── Two-point velocity ──


@pytest.mark.parametrize("t_val", [0.2, 0.6, 0.9])
def test_two_point_forward_velocity(mask_interpolant, uniform_interpolant, t_val):
    """Forward velocity matches :math:`\\dot\\kappa/(1-\\kappa)\\,(p_{\\mathrm{tgt}\\mid t}-\\delta_z)`."""
    t = jnp.asarray(t_val)
    posterior = jax.nn.softmax(jr.normal(jr.PRNGKey(0), (K,)))
    coeff = 1.0 / (1.0 - t_val)
    for interpolant in (mask_interpolant, uniform_interpolant):
        for z in range(interpolant.num_states):
            z_t = jax.nn.one_hot(z, interpolant.num_states)
            got = interpolant.rates_from_target_posterior(posterior, z_t, t)
            padded = interpolant.mixture_distributions_from_target_posterior(
                posterior, z_t
            )[..., 0, :]
            expected = coeff * (padded - z_t.astype(padded.dtype))
            assert jnp.allclose(got, expected, atol=1e-4)


@pytest.mark.parametrize("t_val", [0.2, 0.6, 0.9])
def test_two_point_backward_velocity(mask_interpolant, uniform_interpolant, t_val):
    """Backward velocity matches :math:`\\dot\\kappa/\\kappa\\,(p_{\\mathrm{src}} - \\delta_z)`."""
    t = jnp.asarray(t_val)
    posterior = jax.nn.softmax(jr.normal(jr.PRNGKey(1), (K,)))
    coeff = 1.0 / t_val
    for interpolant in (mask_interpolant, uniform_interpolant):
        for z in range(interpolant.num_states):
            z_t = jax.nn.one_hot(z, interpolant.num_states)
            got = interpolant.rates_from_target_posterior(
                posterior, z_t, t, backward=True
            )
            source = interpolant.mixture_distributions_from_target_posterior(
                posterior, z_t
            )[..., 1, :]
            expected = coeff * (source - z_t.astype(source.dtype))
            assert jnp.allclose(got, expected, atol=1e-4)


# ── Velocities conserve mass ──


@pytest.mark.parametrize("t_val", [0.2, 0.6, 0.9])
def test_forward_velocity_conserves_mass(mask_interpolant, uniform_interpolant, t_val):
    t = jnp.asarray(t_val)
    posterior = jax.nn.softmax(jr.normal(jr.PRNGKey(0), (K,)))
    for interpolant in (mask_interpolant, uniform_interpolant):
        rate_matrix = _rate_matrix(interpolant, posterior, t)
        assert jnp.allclose(jnp.sum(rate_matrix, axis=-1), 0.0, atol=1e-4)
        off_diagonal = rate_matrix - jnp.diag(jnp.diag(rate_matrix))
        assert jnp.all(off_diagonal >= -1e-5)


@pytest.mark.parametrize("t_val", [0.2, 0.6, 0.9])
def test_backward_velocity_conserves_mass(mask_interpolant, uniform_interpolant, t_val):
    """The reversed velocity is a rate matrix in its own right, not its negation."""
    t = jnp.asarray(t_val)
    posterior = jax.nn.softmax(jr.normal(jr.PRNGKey(0), (K,)))
    for interpolant in (mask_interpolant, uniform_interpolant):
        rate_matrix = _rate_matrix(interpolant, posterior, t, backward=True)
        assert jnp.allclose(jnp.sum(rate_matrix, axis=-1), 0.0, atol=1e-4)
        off_diagonal = rate_matrix - jnp.diag(jnp.diag(rate_matrix))
        assert jnp.all(off_diagonal >= -1e-5)


def test_mask_forward_velocity_from_mask_matches_two_point(mask_interpolant):
    t = jnp.asarray(0.5)
    posterior = jax.nn.softmax(jr.normal(jr.PRNGKey(0), (K,)))
    num_states = mask_interpolant.num_states
    rates_masked = mask_interpolant.rates_from_target_posterior(
        posterior, jax.nn.one_hot(K, num_states), t
    )
    coeff = 1.0 / (1.0 - t)
    assert jnp.allclose(rates_masked[:K], coeff * posterior, atol=1e-4)
    assert jnp.allclose(rates_masked[K], -coeff, atol=1e-4)


def test_mask_forward_velocity_freezes_unmasked_tokens(mask_interpolant):
    """On the mixture path a non-mask z_t is the target, so p_{tgt|t}=δ_{z_t}."""
    t = jnp.asarray(0.5)
    posterior = jax.nn.softmax(jr.normal(jr.PRNGKey(0), (K,)))
    num_states = mask_interpolant.num_states
    z_t = jax.nn.one_hot(0, num_states)
    rates = mask_interpolant.rates_from_target_posterior(posterior, z_t, t)
    assert jnp.allclose(rates, 0.0, atol=1e-5)


# ── Corrector ──


@pytest.mark.parametrize("t_val", [0.25, 0.5, 0.75])
def test_corrector_is_a_valid_rate_matrix(mask_interpolant, uniform_interpolant, t_val):
    t = jnp.asarray(t_val)
    posterior = jax.nn.softmax(jr.normal(jr.PRNGKey(7), (K,)))
    lam = jnp.asarray(0.5)
    for interpolant in (mask_interpolant, uniform_interpolant):
        rate_matrix = _corrector_rate_matrix(interpolant, posterior, t, lam)
        assert jnp.allclose(jnp.sum(rate_matrix, axis=-1), 0.0, atol=1e-5)
        off_diagonal = rate_matrix - jnp.diag(jnp.diag(rate_matrix))
        assert jnp.all(off_diagonal >= -1e-5)


@pytest.mark.parametrize("t_val", [0.25, 0.5, 0.75])
def test_divergence_free_mix_is_stationary(
    mask_interpolant, uniform_interpolant, t_val
):
    """:math:`\\hat u + \\check u` is the corrector alone (no net time drift)."""
    t = jnp.asarray(t_val)
    # Uniform: free learned posterior. Mask: a revealed token pins p_{tgt|t}=δ_{z_t},
    # so the path-correct target atom is a point mass (use that same atom).
    cases = (
        (uniform_interpolant, jax.nn.softmax(jr.normal(jr.PRNGKey(7), (K,)))),
        (mask_interpolant, jax.nn.one_hot(0, K).astype(jnp.float32)),
    )
    for interpolant, posterior in cases:
        z_ref = jax.nn.one_hot(0, interpolant.num_states)
        posteriors = interpolant.mixture_distributions_from_target_posterior(
            posterior, z_ref
        )
        kappa = interpolant.kappa_fn(t)[0]
        # Two-point mixture marginal: κ p_tgt + (1-κ) p_src.
        marginal = kappa * posteriors[..., 0, :] + (1.0 - kappa) * posteriors[..., 1, :]
        rate_matrix = _rate_matrix(
            interpolant, posterior, t, backward=False
        ) + _rate_matrix(interpolant, posterior, t, backward=True)
        assert jnp.allclose(jnp.sum(rate_matrix, axis=-1), 0.0, atol=1e-5)
        assert jnp.allclose(marginal @ rate_matrix, 0.0, atol=1e-4)
