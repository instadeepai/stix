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

"""Unit tests for per-generator-type step / drift functions."""

import jax
import jax.numpy as jnp
import jax.random as jr

from stix.core.generator import TransitionRates, VelocityAndScore
from stix.sampling.steps import (
    ctmc_euler_step,
    velocity_and_score_drift,
    velocity_and_score_euler_maruyama_step,
)
from stix.sampling.utils import Direction

# Three-state CTMC sitting on state 0, with both sets of transition rates leaving
# it at rate 2: the forward one jumps to state 1, the reversed one to state 2.
# Distinct targets, so a step that weighs them wrongly lands somewhere the
# assertions see.
# Off-diagonal rate 2 collapses the λ=0 step to a delta after clip when
# |dt| = 0.5, making the categorical draw deterministic.
_FORWARD_RATE = jnp.array([-2.0, 2.0, 0.0])
_BACKWARD_RATE = jnp.array([-2.0, 0.0, 2.0])
_STAY = jnp.array([1.0, 0.0, 0.0])
_TO_FORWARD = jnp.array([0.0, 1.0, 0.0])
_TO_BACKWARD = jnp.array([0.0, 0.0, 1.0])
_DT = 0.5


def _ctmc_euler_step(
    *,
    dt,
    direction,
    forward=_FORWARD_RATE,
    backward=_BACKWARD_RATE,
    stochasticity_scale=None,
    key=jr.PRNGKey(0),
):
    """One step from state 0 with the given rates, corrector scale, and direction."""
    return ctmc_euler_step(
        t=jnp.array(0.5),
        z_t_k=_STAY,
        generator_k=TransitionRates(forward_rates=forward, backward_rates=backward),
        dt=dt,
        key_k=key,
        stochasticity_scale_k=stochasticity_scale,
        direction=direction,
    )


def _ctmc_landing_frequencies(*, direction, stochasticity_scale, num_keys=8000):
    """Empirical distribution of the state one step lands on, over ``num_keys`` draws."""
    keys = jr.split(jr.PRNGKey(0), num_keys)
    out = jax.vmap(
        lambda key: _ctmc_euler_step(
            dt=_DT,
            direction=direction,
            stochasticity_scale=stochasticity_scale,
            key=key,
        )
    )(keys)
    return jnp.mean(out, axis=0)


def test_ctmc_forward_default_follows_forward_rates():
    """Default λ=0 forward step with dt>0 applies û alone and jumps."""
    out = _ctmc_euler_step(dt=_DT, direction=Direction.FORWARD)
    assert jnp.array_equal(out, _TO_FORWARD)


def test_ctmc_reverse_default_follows_backward_rates():
    """Default λ=0 reverse swaps the weights, so the step applies č alone."""
    out = _ctmc_euler_step(dt=-_DT, direction=Direction.REVERSE)
    assert jnp.array_equal(out, _TO_BACKWARD)


def test_ctmc_reverse_swap_is_constant_folded_under_jit():
    """Direction is closed over by the solver scan; the swap must survive ``jax.jit``."""
    out = jax.jit(
        lambda key: _ctmc_euler_step(dt=-_DT, direction=Direction.REVERSE, key=key)
    )(jr.PRNGKey(0))
    assert jnp.array_equal(out, _TO_BACKWARD)


def test_ctmc_step_ignores_the_sign_of_dt():
    """Only ``|dt|`` enters the step: direction comes from the weight swap alone.

    A signed ``dt`` would flip the chain a second time and clip it onto itself,
    making every reverse step a no-op.
    """
    for direction in (Direction.FORWARD, Direction.REVERSE):
        assert jnp.array_equal(
            _ctmc_euler_step(dt=_DT, direction=direction),
            _ctmc_euler_step(dt=-_DT, direction=direction),
        )


def test_ctmc_none_scale_matches_zero_lambda():
    """``stochasticity_scale_k is None`` is exactly :math:`\\lambda \\equiv 0`, both directions."""
    for direction in (Direction.FORWARD, Direction.REVERSE):
        assert jnp.array_equal(
            _ctmc_euler_step(dt=_DT, direction=direction),
            _ctmc_euler_step(
                dt=_DT, direction=direction, stochasticity_scale=lambda t: 0.0
            ),
        )


def test_ctmc_corrector_weighs_the_backward_rates_by_lambda():
    """λ=0.5 mixes (1+λ) û + λ č, i.e. rates 3 forward against 1 backward."""
    frequencies = _ctmc_landing_frequencies(
        direction=Direction.FORWARD, stochasticity_scale=lambda t: jnp.array(0.5)
    )
    assert jnp.allclose(frequencies, jnp.array([0.0, 0.75, 0.25]), atol=0.02)


def test_ctmc_reverse_swaps_the_corrector_weights():
    """On reverse the same λ=0.5 mixes λ û + (1+λ) č, mirroring the forward split."""
    frequencies = _ctmc_landing_frequencies(
        direction=Direction.REVERSE, stochasticity_scale=lambda t: jnp.array(0.5)
    )
    assert jnp.allclose(frequencies, jnp.array([0.0, 0.25, 0.75]), atol=0.02)


def test_velocity_and_score_drift_flips_score_term_on_reverse():
    """Reverse SDE drift is ``b - λ s``; forward is ``b + λ s``."""
    generator = VelocityAndScore(
        velocity=jnp.array([1.0, 0.0]),
        score=jnp.array([0.0, 1.0]),
    )

    def stochasticity_scale(_t):
        return jnp.array(2.0)

    t = jnp.array(0.5)
    forward = velocity_and_score_drift(
        t, generator, stochasticity_scale, Direction.FORWARD
    )
    reverse = velocity_and_score_drift(
        t, generator, stochasticity_scale, Direction.REVERSE
    )
    assert jnp.allclose(forward, jnp.array([1.0, 2.0]))
    assert jnp.allclose(reverse, jnp.array([1.0, -2.0]))


def test_velocity_and_score_euler_maruyama_step_flips_score_term_on_reverse():
    """Same key ⇒ same noise; reverse and forward steps differ by ``-2 λ s dt``."""
    generator = VelocityAndScore(
        velocity=jnp.array([1.0, 0.0]),
        score=jnp.array([0.0, 1.0]),
    )

    def stochasticity_scale(_t):
        return jnp.array(2.0)

    dt = 0.1
    t = jnp.array(0.5)
    z_t_k = jnp.zeros(2)
    key_k = jr.PRNGKey(0)
    forward = velocity_and_score_euler_maruyama_step(
        t,
        z_t_k,
        generator,
        dt,
        key_k,
        stochasticity_scale,
        Direction.FORWARD,
    )
    reverse = velocity_and_score_euler_maruyama_step(
        t,
        z_t_k,
        generator,
        dt,
        key_k,
        stochasticity_scale,
        Direction.REVERSE,
    )
    assert jnp.allclose(forward - reverse, jnp.array([0.0, 0.4]))
