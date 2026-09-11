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

"""Fixed-step solver for sampling from the generative model.

A simple fixed-step integrator that makes every computation explicit and
debuggable, unlike the diffrax-based adaptive Solver. **It defines no update rule
of its own**: the rules live in :data:`~stix.sampling.steps.STEP_FNS`, keyed by
``generator_type`` — Euler for a :class:`~stix.core.generator.Velocity`,
Euler-Maruyama for a :class:`~stix.core.generator.VelocityAndScore`, Euler on the
transition probabilities for a :class:`~stix.core.generator.TransitionRates`.
What this solver owns is everything around them: the time grid, the per-modality
PRNG split, and the stochasticity callables it threads in. It is also the only
solver that handles continuous SDEs and discrete CTMCs simultaneously.

NOTE: those three rules are the *only* schemes expressible here. The network is
evaluated once per step, before dispatch, so no step function can query the model
at an intermediate stage; Heun, midpoint and higher-order stochastic Runge-Kutta
would need that evaluation moved into the step function itself. Use
:class:`~stix.sampling.solver.Solver` for higher-order integration of purely
continuous modalities.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from jaxtyping import PyTree

from stix.core.gen_model.gen_model import GenerativeModel
from stix.core.generator import TransitionRates
from stix.core.interpolant.standard_interpolants.discrete_diffusion import (
    MaskDiscreteInterpolant,
)
from stix.sampling.guidance import (
    GuidanceFn,
    get_context_conditional_generator,
)
from stix.sampling.steps import STEP_FNS
from stix.sampling.utils import (
    Direction,
    padded_time_endpoints,
    validate_generator_types,
)
from stix.typing import (
    Mask,
    PRNGKeyArray,
    RawVar,
    Scalar,
    Time,
    Var,
)


@dataclass
class ManualSolverConfig:
    r"""Configuration for the fixed-step solver.

    Args:
        stochasticity_scale: Per-modality time-dependent stochasticity scale.
            Each callable maps :math:`t` to :math:`\lambda_t\in\mathbb{R}_{\geq 0}`,
            and ``None`` (default, or a ``None`` leaf) is :math:`\lambda\equiv 0`.
            For a :class:`~stix.core.generator.VelocityAndScore` modality it is
            the SDE scale, so ``None`` means ODE sampling. For a
            :class:`~stix.core.generator.TransitionRates` modality it is the CTMC
            corrector scale of
            :math:`\bar u_t = (1 + \lambda_t)\,\hat u_t + \lambda_t\,\check u_t`,
            so ``None`` means the plain generative chain (see
            :doc:`/introduction` for the notations and more details). Setting it
            for a modality whose ``generator_type`` carries no score
            (:class:`~stix.core.generator.Velocity`), or setting a callable that
            is negative on ``[0, 1]``, raises a ``ValueError``.
        direction: Integration direction ("forward" or "reverse"). Defaults to
            "forward". Forward integrates ``t=0`` -> ``t=1``, reverse ``t=1`` -> ``t=0``.
            For :class:`~stix.core.generator.TransitionRates` modalities, reverse
            swaps the two corrector weights, namely :math:`(1 + \lambda_t)\,\hat u_t + \lambda_t\,\check u_t``
            becomes :math:`\lambda_t\,\hat u_t + (1 + \lambda_t)\,\check u_t`,
            so the same :math:`\lambda_t` time-reverses the CTMC.
        num_steps: Number of fixed steps. Defaults to 300.
        t_min_tolerance: Pad above ``t=0`` (integration starts/ends at
            ``0 + t_min_tolerance``). Defaults to 1e-4.
        t_max_tolerance: Pad below ``t=1`` (integration starts/ends at
            ``1 - t_max_tolerance``). Defaults to 1e-4.
        guidance_fn: A :class:`~stix.sampling.guidance.GuidanceFn` recipe
            returning the per-modality :class:`~stix.core.generator.Generator`.
            ``None`` (default) means *no guidance*: the solver falls back to
            :func:`~stix.sampling.guidance.get_context_conditional_generator`, a
            single network forward that is unconditional (with ``context_data``).
        guidance_scale: Per-modality time-dependent guidance scale. Defaults to
            an empty dict (sufficient for the default ``guidance_fn``, which
            ignores it).
        resolve_terminal_mask: If True (default), a mask-source discrete modality
            (masked diffusion) whose CTMC left some positions on the mask symbol
            has them resolved after the final step, by ``argmax`` of the terminal
            denoising posterior. This parameter is a no-op for modalities without
            a :class:`~stix.core.interpolant.MaskDiscreteInterpolant`, so it is
            safe to leave on. It is also a no-op on
            :attr:`~stix.sampling.utils.Direction.REVERSE`: a mask interpolant has
            data at :math:`t=1`.
    """

    stochasticity_scale: PyTree[Callable[[Time], Scalar] | None] | None = None
    direction: Direction = Direction.FORWARD
    num_steps: int = 300
    t_min_tolerance: float = 1e-4
    t_max_tolerance: float = 1e-4
    guidance_fn: GuidanceFn | None = None
    guidance_scale: PyTree[Callable[[Time], Scalar]] = field(default_factory=dict)
    resolve_terminal_mask: bool = True


class _TimeGrid(NamedTuple):
    """The padded, signed time grid a fixed-step run walks.

    Derived once per run by :func:`_time_grid` and threaded from there, so no
    part of the solver re-derives a time or a step size for itself.

    Attributes:
        t_start: Time of the initial state, as a ``float32`` scalar ready to
            carry through the scan.
        t_end: Time of the final state, ``num_steps`` steps of ``dt`` away, in
            the same dtype.
        dt: Signed step size, negative on
            :attr:`~stix.sampling.utils.Direction.REVERSE`.
        num_steps: Number of steps taken between ``t_start`` and ``t_end``.
    """

    t_start: Time
    t_end: Time
    dt: float
    num_steps: int


def _time_grid(solver_config: ManualSolverConfig) -> _TimeGrid:
    """Return the :class:`_TimeGrid` for the configured direction and step count.

    The endpoints are padded by the configured tolerances, then ordered by the
    direction, so ``dt`` carries the sign and every downstream consumer is
    direction-agnostic.

    Args:
        solver_config: Fixed-step solver configuration.

    Returns:
        The grid this configuration describes.
    """
    t_start, t_end = padded_time_endpoints(
        solver_config.t_min_tolerance,
        solver_config.t_max_tolerance,
        solver_config.direction,
    )
    num_steps = solver_config.num_steps
    return _TimeGrid(
        t_start=jnp.float32(t_start),
        t_end=jnp.float32(t_end),
        dt=(t_end - t_start) / num_steps,
        num_steps=num_steps,
    )


def _run_fixed_step(
    grid: _TimeGrid,
    step_fn: Callable,
    z_init: PyTree[Var],
    gen_model_state: Any,
    key: PRNGKeyArray,
    store_trajectory: bool = False,
):
    """Run the fixed-step scan loop and return the final state (or trajectory).

    Args:
        grid: The time grid to walk, from :func:`_time_grid`.
        step_fn: ``(t, z_t, gen_model_state, key) -> PyTree[Var]``, one full step
            for every modality (dispatching per ``generator_type`` internally).
        z_init: Initial samples in embedding space, one array per modality.
        gen_model_state: nnx state pytree, threaded as a dynamic arg.
        key: PRNG key; split into one key per step. Deterministic (ODE) steps
            ignore their key, so their numbers are unchanged.
        store_trajectory: If True, also return per-step states and times.

    Returns:
        The final state ``z_final`` as a ``PyTree[Var]`` when ``store_trajectory``
        is False, otherwise the triple ``(z_final, trajectory, trajectory_t)``.
        The trajectory holds the state *entering* each step, so it runs
        ``grid.t_start`` to ``grid.t_end - grid.dt`` and excludes ``z_final``.
    """
    dt, num_steps = grid.dt, grid.num_steps

    @jax.jit
    def _integrate(gen_model_state: Any, z_init: PyTree[Var], key: PRNGKeyArray):
        """JIT body that walks the fixed grid (optionally storing a trajectory)."""

        def _scan_step(carry, step_key):
            """Advance one fixed step inside ``lax.scan``."""
            z_t, t = carry
            z_next = step_fn(t, z_t, gen_model_state, step_key)
            output = (z_t, t) if store_trajectory else None
            return (z_next, t + dt), output

        step_keys = jr.split(key, num_steps)
        (z_final, _), scan_output = jax.lax.scan(
            _scan_step,
            init=(z_init, grid.t_start),
            xs=step_keys,
            length=num_steps,
        )
        if store_trajectory and scan_output is not None:
            trajectory, trajectory_t = scan_output
            return z_final, trajectory, trajectory_t
        return z_final

    return _integrate(gen_model_state, z_init, key)


class ManualSolver:
    """Unconditional or guided fixed-step SDE/ODE/CTMC solver for the generative model.

    A fixed-step integrator that is easy to inspect and debug, and compatible with
    ``jax.vmap`` over samples. Each step calls the configured ``guidance_fn``
    recipe for the per-modality :class:`~stix.core.generator.Generator`, then
    applies the update rule :data:`~stix.sampling.steps.STEP_FNS` holds for that
    modality's ``generator_type``:

    - ODE path (``stochasticity_scale is None``): Euler on the velocity.
    - SDE path: Euler-Maruyama, drift = velocity + score term. Only
      :class:`~stix.core.generator.VelocityAndScore` modalities may
      carry a ``stochasticity_scale``.
    - CTMC path: Euler on the transition probabilities of
      :class:`~stix.core.generator.TransitionRates` modalities, where
      ``stochasticity_scale`` weighs the backward rates into the corrector.

    This is the only solver that handles continuous SDEs and discrete CTMCs
    simultaneously — :class:`~stix.sampling.solver.Solver` rejects a
    :class:`~stix.core.generator.TransitionRates` modality outright — so it is
    the solver for discrete and mixed-modality sampling.

    **Guidance is opt-in**: with the default ``guidance_fn=None`` the recipe is a
    single plain network forward, so an unconfigured solver performs
    **unconditional sampling** (context-conditional if you pass ``context_data``).
    Set ``guidance_fn`` to a recipe from :mod:`stix.sampling.guidance` for e.g.
    classifier-free or intrinsic guidance.

    The gen_model is split internally: the graph definition is captured
    in closures (static) and the state flows as a dynamic pytree.

    Note:
        **These three rules are the only schemes this class can express.** The
        network is evaluated once per step, before dispatch, so no step function
        can re-evaluate it mid-step: Heun, midpoint and higher-order stochastic
        Runge-Kutta are out of reach by construction. See
        :class:`~stix.sampling.solver.Solver` for higher-order integration of
        purely continuous modalities.
    """

    def __init__(self, solver_config: ManualSolverConfig):
        """Initialise the manual solver with the given configuration.

        Args:
            solver_config: Fixed-step solver configuration.
        """
        self.solver_config = solver_config

    def __call__(
        self,
        gen_model: GenerativeModel,
        z_init: PyTree[Var],
        key: PRNGKeyArray,
        *,
        context_data: PyTree[Var] | None = None,
        context_mask: PyTree[Mask | None] | None = None,
        intrinsic_data: PyTree[Var] | None = None,
        intrinsic_mask: PyTree[Mask | None] | None = None,
        attention_mask: PyTree[Mask | None] | None = None,
    ) -> PyTree[RawVar]:
        """Integrate the SDE/ODE/CTMC and return decoded samples.

        Safe to use under jax.vmap — the gen_model state is traced,
        the graph definition is captured in the closure.

        Args:
            gen_model: The generative model (owns the network and the modality registry).
            z_init: Initial samples in embedding space, one array per modality.
            key: PRNG key for the Brownian increments and CTMC draws; split per
                step and per modality (deterministic ODE steps ignore theirs).
            context_data: Context conditioning consumed by the configured
                ``guidance_fn`` recipe, forwarded to the network. See
                :mod:`stix.sampling.guidance` for the context contract.
            context_mask: Context masks paired with ``context_data``.
            intrinsic_data: Per-modality intrinsic-guidance targets consumed by
                the configured ``guidance_fn`` recipe.
            intrinsic_mask: Per-modality mask paired with ``intrinsic_data``.
            attention_mask: Per-modality attention masks (sequence padding etc.);
                independent of conditioning.

        Returns:
            Decoded samples in raw data space.

        Raises:
            TypeError: If a modality declares a ``generator_type`` this solver
                cannot step.
            ValueError: If ``stochasticity_scale`` is set for a modality whose
                generator carries no score or is negative on ``[0, 1]``, or if
                ``guidance_scale`` is not structurally compatible with the
                modality registry.
        """
        z_final = self.integrate(
            gen_model,
            z_init,
            key,
            context_data=context_data,
            context_mask=context_mask,
            intrinsic_data=intrinsic_data,
            intrinsic_mask=intrinsic_mask,
            attention_mask=attention_mask,
        )
        return gen_model.modality_registry.map(
            lambda modality, z_final_k: modality.embedder.from_embeddings_to_raw(
                z_final_k
            ),
            z_final,
        )

    def integrate(
        self,
        gen_model: GenerativeModel,
        z_init: PyTree[Var],
        key: PRNGKeyArray,
        *,
        context_data: PyTree[Var] | None = None,
        context_mask: PyTree[Mask | None] | None = None,
        intrinsic_data: PyTree[Var] | None = None,
        intrinsic_mask: PyTree[Mask | None] | None = None,
        attention_mask: PyTree[Mask | None] | None = None,
    ) -> PyTree[Var]:
        """Integrate the SDE/ODE/CTMC and return embedding-space ``z_final``.

        Use this (instead of :meth:`__call__`) when you need the final state
        in embedding space — for example to compose two solver passes for a
        round-trip, or to hand the output to another component that operates
        in embedding space — without paying the cost of trajectory storage
        from :meth:`integrate_with_trajectory`.

        Args:
            gen_model: The generative model (owns the network and the modality registry).
            z_init: Initial samples in embedding space, one array per modality.
            key: PRNG key for the Brownian increments and CTMC draws; split per
                step and per modality (deterministic ODE steps ignore theirs).
            context_data: Context conditioning consumed by the configured
                ``guidance_fn`` recipe, forwarded to the network. See
                :mod:`stix.sampling.guidance` for the context contract.
            context_mask: Context masks paired with ``context_data``.
            intrinsic_data: Per-modality intrinsic-guidance targets consumed by
                the configured ``guidance_fn`` recipe.
            intrinsic_mask: Per-modality mask paired with ``intrinsic_data``.
            attention_mask: Per-modality attention masks (sequence padding etc.);
                independent of conditioning.

        Returns:
            Final state in embedding space, ``PyTree[Var]`` (no per-modality
            decode applied).

        Raises:
            TypeError: If a modality declares a ``generator_type`` this solver
                cannot use.
            ValueError: If ``stochasticity_scale`` is set for a modality whose
                generator carries no score or is negative on ``[0, 1]``, or if
                ``guidance_scale`` is not structurally compatible with the
                modality registry.
        """
        step_fn, grid, gen_model_state = self._prepare(
            gen_model,
            context_data,
            context_mask,
            intrinsic_data,
            intrinsic_mask,
            attention_mask,
        )
        z_final = _run_fixed_step(grid, step_fn, z_init, gen_model_state, key)
        return self._maybe_resolve_terminal_mask(
            gen_model,
            z_final,
            grid.t_end,
            context_data,
            context_mask,
            intrinsic_data,
            intrinsic_mask,
            attention_mask,
        )

    def integrate_with_trajectory(
        self,
        gen_model: GenerativeModel,
        z_init: PyTree[Var],
        key: PRNGKeyArray,
        *,
        context_data: PyTree[Var] | None = None,
        context_mask: PyTree[Mask | None] | None = None,
        intrinsic_data: PyTree[Var] | None = None,
        intrinsic_mask: PyTree[Mask | None] | None = None,
        attention_mask: PyTree[Mask | None] | None = None,
    ) -> tuple[PyTree[Var], PyTree[Var], Var]:
        """Integrate and return the full trajectory in embedding space.

        Args:
            gen_model: The generative model (owns the network and the modality registry).
            z_init: Initial samples in embedding space, one array per modality.
            key: PRNG key for the Brownian increments and CTMC draws; split per
                step and per modality (deterministic ODE steps ignore theirs).
            context_data: Context conditioning consumed by the configured
                ``guidance_fn`` recipe, forwarded to the network. See
                :mod:`stix.sampling.guidance` for the context contract.
            context_mask: Context masks paired with ``context_data``.
            intrinsic_data: Per-modality intrinsic-guidance targets consumed by
                the configured ``guidance_fn`` recipe.
            intrinsic_mask: Per-modality mask paired with ``intrinsic_data``.
            attention_mask: Per-modality attention masks (sequence padding etc.);
                independent of conditioning.

        Returns:
            A triple ``(z_final, trajectory, trajectory_t)``: the final state as a
            ``PyTree[Var]``, the per-step states as a ``PyTree[Var]`` with a leading
            ``(num_steps,)`` axis, and the time values at each step with shape
            ``(num_steps,)``. ``trajectory`` holds the state *entering* each step,
            so it starts at ``z_init`` and stops one step short of ``z_final``.
            ``resolve_terminal_mask`` acts on ``z_final`` alone, never on the
            states stored in ``trajectory``.

        Raises:
            TypeError: If a modality declares a ``generator_type`` this solver
                cannot use.
            ValueError: If ``stochasticity_scale`` is set for a modality whose
                generator carries no score or is negative on ``[0, 1]``, or if
                ``guidance_scale`` is not structurally compatible with the
                modality registry.
        """
        step_fn, grid, gen_model_state = self._prepare(
            gen_model,
            context_data,
            context_mask,
            intrinsic_data,
            intrinsic_mask,
            attention_mask,
        )
        z_final, trajectory, trajectory_t = _run_fixed_step(
            grid,
            step_fn,
            z_init,
            gen_model_state,
            key,
            store_trajectory=True,
        )
        z_final = self._maybe_resolve_terminal_mask(
            gen_model,
            z_final,
            grid.t_end,
            context_data,
            context_mask,
            intrinsic_data,
            intrinsic_mask,
            attention_mask,
        )
        return z_final, trajectory, trajectory_t

    def _prepare(
        self,
        gen_model: GenerativeModel,
        context_data: PyTree[Var] | None,
        context_mask: PyTree[Mask | None] | None,
        intrinsic_data: PyTree[Var] | None,
        intrinsic_mask: PyTree[Mask | None] | None,
        attention_mask: PyTree[Mask | None] | None,
    ) -> tuple[Callable, _TimeGrid, Any]:
        """Validate, split the model, and build the per-step function.

        Args:
            gen_model: Generative model to split and step.
            context_data: Context conditioning for the guidance recipe.
            context_mask: Context masks paired with ``context_data``.
            intrinsic_data: Intrinsic-guidance targets for the guidance recipe.
            intrinsic_mask: Intrinsic masks paired with ``intrinsic_data``.
            attention_mask: Per-modality attention masks.

        Returns:
            Triple ``(step_fn, grid, gen_model_state)``.
        """
        validate_generator_types(
            gen_model.modality_registry,
            supported=STEP_FNS,
            stochasticity_scale=self.solver_config.stochasticity_scale,
        )
        if self.solver_config.guidance_scale:
            gen_model.modality_registry.assert_compatible(
                self.solver_config.guidance_scale
            )
        stochasticity_scale_tree = gen_model.modality_registry.broadcast(
            self.solver_config.stochasticity_scale,
            is_leaf=lambda leaf: leaf is None or callable(leaf),
        )
        graphdef, gen_model_state = nnx.split(gen_model)
        grid = _time_grid(self.solver_config)
        step_fn = self._build_step_fn(
            graphdef,
            stochasticity_scale_tree,
            grid.dt,
            context_data,
            context_mask,
            intrinsic_data,
            intrinsic_mask,
            attention_mask,
        )
        return step_fn, grid, gen_model_state

    def _maybe_resolve_terminal_mask(
        self,
        gen_model: GenerativeModel,
        z_final: PyTree[Var],
        t_end: Time,
        context_data: PyTree[Var] | None,
        context_mask: PyTree[Mask | None] | None,
        intrinsic_data: PyTree[Var] | None,
        intrinsic_mask: PyTree[Mask | None] | None,
        attention_mask: PyTree[Mask | None] | None,
    ) -> PyTree[Var]:
        """Resolve any leftover mask tokens on mask-source discrete modalities.

        Masked diffusion can end a finite-step CTMC with a few positions still on
        the mask symbol (and mask interpolant always runs forward to t=1 as one-sided).
        One extra forward at the terminal time gives the denoising posterior; each
        still-masked position is set to its ``argmax`` data category (read off the
        forward rates' data-category entries, which are ``coeff * posterior``).
        Modalities without a :class:`~stix.core.interpolant.MaskDiscreteInterpolant`
        are untouched, and if the model has none the extra forward is skipped
        entirely.

        Args:
            gen_model: Generative model used for the terminal forward.
            z_final: Final embedding-space state after integration.
            t_end: Time of the last state on the grid (``grid.t_end``).
            context_data: Context conditioning for the guidance recipe.
            context_mask: Context masks paired with ``context_data``.
            intrinsic_data: Intrinsic-guidance targets for the guidance recipe.
            intrinsic_mask: Intrinsic masks paired with ``intrinsic_data``.
            attention_mask: Per-modality attention masks.

        Returns:
            ``z_final`` with residual mask tokens resolved where applicable.
        """
        if not self.solver_config.resolve_terminal_mask:
            return z_final
        # A MaskDiscreteInterpolant is a OneSidedInterpolant in which data is always t=1.
        if self.solver_config.direction == Direction.REVERSE:
            return z_final
        registry = gen_model.modality_registry
        if not any(
            isinstance(modality.interpolant, MaskDiscreteInterpolant)
            for modality in registry._modality_leaves()
        ):
            return z_final
        guidance_fn = (
            self.solver_config.guidance_fn or get_context_conditional_generator
        )
        generators = guidance_fn(
            gen_model,
            z_final,
            t_end,
            context_data,
            context_mask,
            intrinsic_data,
            intrinsic_mask,
            attention_mask=attention_mask,
            guidance_scale=self.solver_config.guidance_scale,
        )

        def _resolve(modality, z_final_k, generator_k):
            """Resolve residual mask tokens for one modality leaf."""
            interpolant = modality.interpolant
            if not isinstance(interpolant, MaskDiscreteInterpolant):
                return z_final_k
            # A MaskDiscreteInterpolant fixes generator_type to TransitionRates
            # (DiscreteInterpolant.generator_type), so the rates below are there.
            assert isinstance(generator_k, TransitionRates)
            # ``z_final_k`` is a one-hot vector over the ``K + 1`` states. Resolve
            # each still-masked position to the argmax data category (read off
            # the forward rate's data-category entries, which are
            # ``coeff * posterior``),
            # keeping the state one-hot so the downstream decode (argmax) is exact.
            num_categories = interpolant.num_categories
            num_states = interpolant.num_states
            current_index = jnp.argmax(z_final_k, axis=-1)
            data_argmax = jnp.argmax(
                generator_k.forward_rates[..., :num_categories], axis=-1
            )
            is_masked = current_index == interpolant.mask_index
            resolved_index = jnp.where(is_masked, data_argmax, current_index)
            return jax.nn.one_hot(resolved_index, num_states, dtype=z_final_k.dtype)

        return registry.map(_resolve, z_final, generators)

    def _build_step_fn(
        self,
        graphdef: nnx.GraphDef,
        stochasticity_scale_tree: PyTree,
        dt: float,
        context_data: PyTree[Var] | None,
        context_mask: PyTree[Mask | None] | None,
        intrinsic_data: PyTree[Var] | None,
        intrinsic_mask: PyTree[Mask | None] | None,
        attention_mask: PyTree[Mask | None] | None,
    ) -> Callable:
        r"""Build the single per-step closure consumed by ``_run_fixed_step``.

        On each call the closure asks ``guidance_fn`` for the generators, then
        gives each modality to the step function that matches its
        ``generator_type`` (:data:`~stix.sampling.steps.STEP_FNS`). Each modality
        gets its own PRNG key, so a CTMC modality can draw its categorical jump.

        Args:
            graphdef: Static nnx graph definition of the generative model.
            stochasticity_scale_tree: Per-modality stochasticity scale callables or ``None``.
            dt: Signed time step on the fixed grid.
            context_data: Context conditioning for the guidance recipe.
            context_mask: Context masks paired with ``context_data``.
            intrinsic_data: Intrinsic-guidance targets for the guidance recipe.
            intrinsic_mask: Intrinsic masks paired with ``intrinsic_data``.
            attention_mask: Per-modality attention masks.

        Returns:
            A ``step_fn(t, z_t, gen_model_state, key) -> z_{t+dt}`` closure.
        """
        guidance_fn = (
            self.solver_config.guidance_fn or get_context_conditional_generator
        )
        guidance_scale = self.solver_config.guidance_scale
        direction = self.solver_config.direction

        def _step_fn(
            t: Time, z_t: PyTree[Var], gen_model_state: Any, key: PRNGKeyArray
        ) -> PyTree[Var]:
            """One fixed-step update over all modalities."""
            gen_model = nnx.merge(graphdef, gen_model_state)
            # TODO: evaluating the generators here, rather than inside the
            # step function, is what pins this solver to single-stage schemes.
            generators = guidance_fn(
                gen_model,
                z_t,
                t,
                context_data,
                context_mask,
                intrinsic_data,
                intrinsic_mask,
                attention_mask=attention_mask,
                guidance_scale=guidance_scale,
            )
            key_tree = gen_model.modality_registry.split_and_project_key(key)
            return gen_model.modality_registry.map(
                lambda modality, z_t_k, generator_k, stochasticity_scale_k, key_k: (
                    STEP_FNS[modality.generator_type](
                        t,
                        z_t_k,
                        generator_k,
                        dt,
                        key_k,
                        stochasticity_scale_k,
                        direction,
                    )
                ),
                z_t,
                generators,
                stochasticity_scale_tree,
                key_tree,
            )

        return _step_fn
