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

"""End-to-end discrete DFM CTMC sampling tests.

Drives :class:`PosteriorMixtureGenerativeModel` through the fixed-step
:class:`ManualSolver` (the only solver that steps ``TransitionRates``): masked and
uniform diffusion, terminal mask clean-up, the diffrax ``Solver`` rejection of
``TransitionRates``, and the logit-space discrete CFG recipe.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest
from conftest import (
    DISCRETE_SHAPE,
    NUM_CATEGORIES,
    SHAPE,
    toy_multimodal_gen_model,
    toy_posterior_mixture_gen_model,
)

from stix.sampling.guidance import (
    get_discrete_classifier_free_guidance_generator,
)
from stix.sampling.solver import Solver, SolverConfig
from stix.sampling.solver_manual import ManualSolver, ManualSolverConfig
from stix.sampling.utils import Direction


def _sample(gen_model, solver, num_samples, key, **call_kwargs):
    """Sample ``num_samples`` from ``gen_model`` by vmapping ``solver`` over the batch."""
    init_key, sample_key = jr.split(key)
    z_init = gen_model.modality_registry.sample_initial_state(
        init_key, num_samples=num_samples
    )
    sample_keys = jr.split(sample_key, num_samples)

    def _one(x_init_k, key_k):
        return solver(gen_model, x_init_k, key_k, **call_kwargs)

    return jax.vmap(_one)(z_init, sample_keys)


def test_masked_diffusion_resolves_to_favoured_category():
    """A network favouring one category unmasks every position to it, no masks left."""
    gen_model = toy_posterior_mixture_gen_model(source="mask", favoured_category=0)
    solver = ManualSolver(ManualSolverConfig(num_steps=100))
    out = _sample(gen_model, solver, num_samples=16, key=jr.PRNGKey(0))["tokens"]

    assert out.shape == (16, *DISCRETE_SHAPE)
    assert jnp.all(out != NUM_CATEGORIES)  # the mask symbol is index NUM_CATEGORIES
    assert jnp.all(out == 0)


def test_masked_diffusion_initial_state_is_all_mask():
    """The DFM initial state is the source draw: every position masked."""
    gen_model = toy_posterior_mixture_gen_model(source="mask")
    z_init = gen_model.modality_registry.sample_initial_state(
        jr.PRNGKey(0), num_samples=8
    )
    # z_0 is a one-hot state; every position is the mask symbol (index NUM_CATEGORIES).
    assert z_init["tokens"].shape == (8, *DISCRETE_SHAPE, NUM_CATEGORIES + 1)
    assert jnp.all(jnp.argmax(z_init["tokens"], axis=-1) == NUM_CATEGORIES)


def test_terminal_mask_cleanup_resolves_leftover_masks():
    """Stopping before t=1 leaves ~(1-kappa) masked; the clean-up resolves them all."""
    gen_model = toy_posterior_mixture_gen_model(source="mask", favoured_category=1)
    out_off = _sample(
        gen_model,
        ManualSolver(
            ManualSolverConfig(
                num_steps=50, t_max_tolerance=0.3, resolve_terminal_mask=False
            )
        ),
        num_samples=64,
        key=jr.PRNGKey(1),
    )["tokens"]
    out_on = _sample(
        gen_model,
        ManualSolver(
            ManualSolverConfig(
                num_steps=50, t_max_tolerance=0.3, resolve_terminal_mask=True
            )
        ),
        num_samples=64,
        key=jr.PRNGKey(1),
    )["tokens"]

    # Without clean-up, some positions are still on the mask symbol.
    assert jnp.any(out_off == NUM_CATEGORIES)
    # With clean-up, none remain; leftovers resolve to the favoured category.
    assert jnp.all(out_on != NUM_CATEGORIES)
    resolved = out_on[out_off == NUM_CATEGORIES]
    assert jnp.all(resolved == 1)


def test_uniform_diffusion_runs_and_converges():
    """Uniform diffusion stays in range and concentrates on the favoured category."""
    gen_model = toy_posterior_mixture_gen_model(source="uniform", favoured_category=2)
    solver = ManualSolver(ManualSolverConfig(num_steps=100))
    out = _sample(gen_model, solver, num_samples=32, key=jr.PRNGKey(2))["tokens"]

    assert jnp.all((out >= 0) & (out < NUM_CATEGORIES))
    assert jnp.mean(out == 2) > 0.8


def test_corrector_sampling_runs_and_stays_valid():
    """A non-trivial DFM corrector (λ>0) keeps samples valid and mask-free."""
    gen_model = toy_posterior_mixture_gen_model(
        source="mask",
        favoured_category=0,
    )
    solver = ManualSolver(
        ManualSolverConfig(
            num_steps=200,
            stochasticity_scale={"tokens": lambda t: jnp.asarray(0.5)},
        )
    )
    out = _sample(gen_model, solver, num_samples=16, key=jr.PRNGKey(3))["tokens"]

    assert jnp.all((out >= 0) & (out <= NUM_CATEGORIES))
    assert jnp.all(out != NUM_CATEGORIES)


def test_reverse_direction_remasks_revealed_tokens():
    """Direction.REVERSE runs masked diffusion backwards: revealed tokens re-mask.

    Started from fully-revealed data at t~1, the reverse CTMC must walk the
    tokens back onto the mask symbol by t~0. Asserting that (rather than only
    that the call returns) is what catches a reverse step that is a silent
    no-op: a doubly-flipped step clips every off-diagonal to zero and returns
    the input unchanged.
    """
    gen_model = toy_posterior_mixture_gen_model(source="mask", favoured_category=0)
    num_states = NUM_CATEGORIES + 1
    mask_index = NUM_CATEGORIES
    z_init = {
        "tokens": jax.nn.one_hot(jnp.zeros(DISCRETE_SHAPE, dtype=jnp.int32), num_states)
    }
    solver = ManualSolver(
        ManualSolverConfig(
            direction=Direction.REVERSE,
            num_steps=50,
            resolve_terminal_mask=False,
        )
    )
    out = solver(gen_model, z_init, key=jr.PRNGKey(0))["tokens"]

    assert out.shape == DISCRETE_SHAPE
    assert jnp.all((out >= 0) & (out <= NUM_CATEGORIES))
    assert jnp.all(out == mask_index), (
        "reverse masked diffusion left tokens revealed: "
        f"{float(jnp.mean(out == mask_index)):.0%} masked"
    )


def test_reverse_direction_skips_terminal_mask_resolution():
    """``resolve_terminal_mask`` is a forward-only finishing step.

    On reverse the terminal time is t~0, where every position is meant to be on
    the mask symbol; resolving there would argmax them straight back to data and
    undo the whole run. Leaving the default on must therefore change nothing.
    """
    gen_model = toy_posterior_mixture_gen_model(source="mask", favoured_category=0)
    num_states = NUM_CATEGORIES + 1
    z_init = {
        "tokens": jax.nn.one_hot(jnp.zeros(DISCRETE_SHAPE, dtype=jnp.int32), num_states)
    }

    def _run(resolve_terminal_mask):
        solver = ManualSolver(
            ManualSolverConfig(
                direction=Direction.REVERSE,
                num_steps=50,
                resolve_terminal_mask=resolve_terminal_mask,
            )
        )
        return solver(gen_model, z_init, key=jr.PRNGKey(0))["tokens"]

    assert jnp.array_equal(_run(True), _run(False))
    assert jnp.all(_run(True) == NUM_CATEGORIES)


def test_diffrax_solver_rejects_transition_rates():
    """The adaptive diffrax ``Solver`` cannot step a CTMC; it must say so clearly."""
    gen_model = toy_posterior_mixture_gen_model(source="mask")
    z_init = gen_model.modality_registry.sample_initial_state(
        jr.PRNGKey(0), num_samples=1
    )
    single = jax.tree.map(lambda a: a[0], z_init)
    solver = Solver(SolverConfig())
    with pytest.raises(TypeError, match="ManualSolver"):
        solver(gen_model, single, key=jr.PRNGKey(0))


def test_discrete_cfg_recipe_runs():
    """The logit-space discrete CFG recipe drives masked diffusion end-to-end."""
    gen_model = toy_posterior_mixture_gen_model(source="mask", favoured_category=0)
    solver = ManualSolver(
        ManualSolverConfig(
            num_steps=100,
            guidance_fn=get_discrete_classifier_free_guidance_generator,
            guidance_scale={"tokens": lambda t: jnp.array(2.0)},
        )
    )
    context_data = {"context": jnp.ones(())}
    out = _sample(
        gen_model,
        solver,
        num_samples=8,
        key=jr.PRNGKey(4),
        context_data=context_data,
    )["tokens"]

    assert jnp.all(out != NUM_CATEGORIES)
    assert jnp.all(out == 0)


def test_multimodal_joint_sampling():
    """One pass advances a continuous SDE and a discrete CTMC on the same grid."""
    gen_model = toy_multimodal_gen_model(velocity_value=1.0, favoured_category=0)
    solver = ManualSolver(
        ManualSolverConfig(
            num_steps=100,
            # SDE on the continuous leg, plain CTMC (λ=0) on the discrete leg.
            stochasticity_scale={"cont": lambda t: jnp.array(0.01), "tokens": None},
        )
    )
    out = _sample(gen_model, solver, num_samples=8, key=jr.PRNGKey(5))

    assert out["cont"].shape == (8, *SHAPE)
    assert jnp.all(jnp.isfinite(out["cont"]))
    assert out["tokens"].shape == (8, *DISCRETE_SHAPE)
    assert jnp.all(out["tokens"] != NUM_CATEGORIES)
    assert jnp.all(out["tokens"] == 0)


def test_negative_stochasticity_scale_is_rejected():
    """A negative λ would put negative rates in the CTMC mix, which the step's
    probability clip would then absorb without a word. Refuse it up front."""
    gen_model = toy_posterior_mixture_gen_model(source="mask")
    z_init = gen_model.modality_registry.sample_initial_state(
        jr.PRNGKey(0), num_samples=1
    )
    single = jax.tree.map(lambda a: a[0], z_init)
    solver = ManualSolver(
        ManualSolverConfig(stochasticity_scale={"tokens": lambda t: jnp.asarray(-0.5)})
    )
    with pytest.raises(ValueError, match="must be non-negative"):
        solver(gen_model, single, key=jr.PRNGKey(0))


def test_traced_negative_scale_is_clamped_to_zero():
    """A λ closing over a traced value cannot be checked up front, so the step
    clamps it: the run stays a valid CTMC and matches ``λ = 0`` exactly."""
    gen_model = toy_posterior_mixture_gen_model(source="mask", favoured_category=0)
    num_states = NUM_CATEGORIES + 1
    z_init = {
        "tokens": jax.nn.one_hot(
            jnp.full(DISCRETE_SHAPE, NUM_CATEGORIES, dtype=jnp.int32), num_states
        )
    }

    @jax.jit
    def _run(scale_value, key):
        """Sample with a λ that is a tracer, hence invisible to the validation."""
        solver = ManualSolver(
            ManualSolverConfig(
                num_steps=50,
                resolve_terminal_mask=False,
                stochasticity_scale={"tokens": lambda t: scale_value},
            )
        )
        return solver(gen_model, z_init, key)["tokens"]

    key = jr.PRNGKey(0)
    clamped = _run(jnp.asarray(-1.0), key)
    at_zero = _run(jnp.asarray(0.0), key)

    # Unclamped, λ=-1 zeroes the forward weight and negates the backward rates,
    # which from an all-mask start gives identically zero rates: every position would
    # stay on the mask symbol for the whole run. The clean-up is off so that
    # freezing stays visible in the decoded tokens.
    assert jnp.any(clamped != NUM_CATEGORIES)
    assert jnp.array_equal(clamped, at_zero)


def test_stochasticity_scale_drives_both_legs():
    """One ``stochasticity_scale`` tree: SDE noise on ``cont``, corrector on ``tokens``."""
    gen_model = toy_multimodal_gen_model(velocity_value=1.0, favoured_category=0)
    solver = ManualSolver(
        ManualSolverConfig(
            num_steps=100,
            stochasticity_scale={
                "cont": lambda t: jnp.array(0.01),
                "tokens": lambda t: jnp.array(0.5),
            },
        )
    )
    out = _sample(gen_model, solver, num_samples=8, key=jr.PRNGKey(6))

    assert jnp.all(jnp.isfinite(out["cont"]))
    assert jnp.all(out["tokens"] != NUM_CATEGORIES)
    assert jnp.all(out["tokens"] == 0)
