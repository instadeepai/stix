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

from dataclasses import dataclass, field
from typing import Callable

import diffrax
import jax
import lineax as lx
from flax import nnx
from jaxtyping import PyTree

from stix.core.gen_model.gen_model import GenerativeModel
from stix.sampling.guidance import (
    GuidanceFn,
    get_context_conditional_generator,
)
from stix.sampling.steps import (
    DIFFUSION_FNS,
    DRIFT_FNS,
)
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
class SolverConfig:
    r"""Configuration for the diffrax SDE/ODE solver.

    Args:
        stochasticity_scale: Per-modality time-dependent stochasticity scale.
            Each callable maps :math:`t` to :math:`\lambda_t\in\mathbb{R}_{\geq 0}`;
            a callable found negative on ``[0, 1]`` raises a ``ValueError``.
            If ``None`` (default), ODE sampling is used, otherwise SDE sampling is used.
            Setting it for a modality whose ``generator_type`` carries no score
            (:class:`~stix.core.generator.Velocity`) raises a ``ValueError``. A
            :class:`~stix.core.generator.TransitionRates` modality is rejected by
            this solver outright, with a ``TypeError``, scale or no scale: use
            :class:`~stix.sampling.solver_manual.ManualSolver` for CTMCs.
        direction: Integration direction ("forward" or "reverse"). Defaults to
            "forward". Forward integrates ``t=0`` -> ``t=1``, reverse ``t=1`` -> ``t=0``.
        rtol: Relative tolerance for adaptive step-size control. Defaults to 1e-3.
        atol: Absolute tolerance for adaptive step-size control. Defaults to 1e-3.
        max_steps: Maximum number of solver steps. Defaults to 4096.
        t_min_tolerance: Pad above ``t=0`` (integration starts/ends at
            ``0 + t_min_tolerance``). Defaults to 1e-4.
        t_max_tolerance: Pad below ``t=1`` (integration starts/ends at
            ``1 - t_max_tolerance``). Defaults to 1e-4.
        guidance_fn: A :class:`~stix.sampling.guidance.GuidanceFn` recipe
            returning the per-modality :class:`~stix.core.generator.Generator`.
            ``None`` (default) means *no guidance*: the solver falls back to
            :func:`~stix.sampling.guidance.get_context_conditional_generator`, a
            single network forward that is unconditional (with ``context_data``).
        guidance_scale: Per-modality time-dependent guidance scale. Defaults to an empty dict (sufficient for the default ``guidance_fn``, which ignores it).

    Note:
        The single ``guidance_scale`` field is sufficient for every recipe in
        :mod:`stix.sampling.guidance`. A recipe that genuinely needs multiple
        independent scales should bake them into a custom guidance function.
    """

    stochasticity_scale: PyTree[Callable[[Time], Scalar] | None] | None = None
    direction: Direction = Direction.FORWARD
    rtol: float = 1e-3
    atol: float = 1e-3
    max_steps: int = 4096
    t_min_tolerance: float = 1e-4
    t_max_tolerance: float = 1e-4
    guidance_fn: GuidanceFn | None = None
    guidance_scale: PyTree[Callable[[Time], Scalar]] = field(default_factory=dict)


def _run(
    solver_config: SolverConfig,
    drift_fn: Callable,
    diffusion_fn: Callable | None,
    z_init: PyTree[Var],
    gen_model_state,
    key: PRNGKeyArray,
) -> PyTree[Var]:
    """Run diffrax integration and return the embedded final state.

    Pure mechanics: builds terms, picks the integrator, runs ``diffeqsolve``.
    Knows nothing about the model, conditioning, or guidance: those concerns
    live in the drift/diffusion closures supplied by the caller.

    Args:
        solver_config: Solver configuration controlling tolerances, direction, etc.
        drift_fn: Diffrax drift function ``(t, z_t, args) -> PyTree[Var]``.
        diffusion_fn: Diffrax diffusion function (SDE only); ``None`` for ODE.
            For per-modality independent (diagonal) noise it must return a
            single ``lx.DiagonalLinearOperator`` wrapping the per-modality
            pytree of diagonals.
        z_init: Initial samples in embedding space, one array per modality.
        gen_model_state: nnx state pytree, threaded through diffrax ``args``.
        key: PRNG key for Brownian motion (consumed only on the SDE path).

    Returns:
        Embedded final state at the integration endpoint.
    """
    drift_term = diffrax.ODETerm(drift_fn)

    is_ode = diffusion_fn is None

    if is_ode:
        terms: diffrax.AbstractTerm = drift_term
        solver: diffrax.AbstractSolver = diffrax.Tsit5()
    else:
        brownian_shape = jax.tree.map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), z_init
        )
        brownian_tree = diffrax.VirtualBrownianTree(
            t0=0.0,
            t1=1.0,
            tol=1e-5,
            shape=brownian_shape,
            key=key,
            levy_area=diffrax.SpaceTimeLevyArea,
        )
        diffusion_term = diffrax.ControlTerm(diffusion_fn, brownian_tree)
        terms = diffrax.MultiTerm(drift_term, diffusion_term)
        solver = diffrax.ShARK()

    t0, t1 = padded_time_endpoints(
        solver_config.t_min_tolerance,
        solver_config.t_max_tolerance,
        solver_config.direction,
    )

    solution = diffrax.diffeqsolve(
        terms=terms,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=None,
        y0=z_init,
        args=gen_model_state,
        stepsize_controller=diffrax.PIDController(
            rtol=solver_config.rtol, atol=solver_config.atol
        ),
        max_steps=solver_config.max_steps,
        throw=True,
    )

    # ``solution.ys`` carries a leading axis over saved timesteps. The default
    # ``SaveAt`` saves the endpoint only, so take the last saved state: callers
    # then get a state shaped exactly like ``z_init``, matching ``ManualSolver``.
    return jax.tree.map(lambda saved_states: saved_states[-1], solution.ys)


class Solver:
    """Unconditional or guided SDE/ODE solver for sampling from the generative model.

    A diffrax-based solver which uses adaptive step-sizes. Each integration step
    calls the configured ``guidance_fn`` recipe to obtain the per-modality
    :class:`~stix.core.generator.Generator`, then steps it via the
    drift/diffusion tables in :mod:`stix.sampling.steps`.

    **Guidance is opt-in**: with the default ``guidance_fn=None`` the recipe is a
    single plain network forward, so an unconfigured solver performs
    **unconditional sampling** (conditional, if you pass ``context_data``). Set
    ``guidance_fn`` to a recipe from :mod:`stix.sampling.guidance` for
    classifier-free or intrinsic guidance; the solver class does not change.

    Two-path design. This solver supplies only the terms — the drift and
    diffusion formulas come from :data:`~stix.sampling.steps.DRIFT_FNS` and
    :data:`~stix.sampling.steps.DIFFUSION_FNS`, keyed by ``generator_type``,
    and the integration scheme is diffrax's:

    - ODE path (``stochasticity_scale is None``): drift = velocity only,
      integrated with ``Tsit5``.
    - SDE path: drift = velocity + score term, with Brownian noise from a
      ``VirtualBrownianTree``, integrated with ``ShARK``. Only
      :class:`~stix.core.generator.VelocityAndScore` modalities may
      carry a ``stochasticity_scale``.

    The gen_model is split internally: the graph definition is captured
    in closures (static) and the state flows as a dynamic pytree.

    Note:
        A :class:`~stix.core.generator.TransitionRates` (CTMC) modality cannot
        be integrated by diffrax; :meth:`integrate` raises a ``TypeError``
        naming the solver. Use
        :class:`~stix.sampling.solver_manual.ManualSolver` instead.
    """

    def __init__(self, solver_config: SolverConfig):
        """Initialise the solver with the given configuration.

        Args:
            solver_config: Diffrax solver configuration.
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
        """Solve the ODE/SDE and return decoded samples.

        The gen_model is split internally: the graph definition is captured
        in closures (static) and the state flows through diffrax ``args``
        (dynamic, traced). Safe to use under ``jax.vmap``.

        Args:
            gen_model: The generative model (owns the network and the modality registry).
            z_init: Initial samples in embedding space, one array per modality.
            key: PRNG key for the Brownian motion (required; ignored for ODE).
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
                cannot step (only ``Velocity`` / ``VelocityAndScore``).
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
        """Solve the ODE/SDE and return embedding-space ``z_final``.

        Use this (instead of :meth:`__call__`) when you need the final state
        in embedding space, for example to compose two solver passes for a
        round-trip, or to hand the output to another component that operates
        in embedding space.

        Args:
            gen_model: The generative model (owns the network and the modality registry).
            z_init: Initial samples in embedding space, one array per modality.
            key: PRNG key for the Brownian motion (required; ignored for ODE).
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
                cannot use (only ``Velocity`` / ``VelocityAndScore``).
            ValueError: If ``stochasticity_scale`` is set for a modality whose
                generator carries no score or is negative on ``[0, 1]``, or if
                ``guidance_scale`` is not structurally compatible with the
                modality registry.
        """
        validate_generator_types(
            gen_model.modality_registry,
            supported=DRIFT_FNS,
            stochasticity_scale=self.solver_config.stochasticity_scale,
        )
        if self.solver_config.guidance_scale:
            gen_model.modality_registry.assert_compatible(
                self.solver_config.guidance_scale
            )
        graphdef, gen_model_state = nnx.split(gen_model)
        stochasticity_scale_tree = gen_model.modality_registry.broadcast(
            self.solver_config.stochasticity_scale,
            is_leaf=lambda leaf: leaf is None or callable(leaf),
        )
        drift_fn, diffusion_fn = self._build_drift_and_diffusion(
            graphdef,
            stochasticity_scale_tree,
            context_data,
            context_mask,
            intrinsic_data,
            intrinsic_mask,
            attention_mask,
        )
        return _run(
            self.solver_config, drift_fn, diffusion_fn, z_init, gen_model_state, key
        )

    def _build_drift_and_diffusion(
        self,
        graphdef: nnx.GraphDef,
        stochasticity_scale_tree: PyTree,
        context_data: PyTree[Var] | None,
        context_mask: PyTree[Mask | None] | None,
        intrinsic_data: PyTree[Var] | None,
        intrinsic_mask: PyTree[Mask | None] | None,
        attention_mask: PyTree[Mask | None] | None,
    ) -> tuple[Callable, Callable | None]:
        """Build the drift and diffusion closures consumed by ``_run``.

        Both closures take ``(t, z_t, gen_model_state)``: diffrax hands the model
        state back in as ``args``, so it stays a traced value instead of being
        frozen into the closure. On each call the drift asks ``guidance_fn`` for
        the generators, then gives each modality to the drift formula that matches
        its ``generator_type`` (:data:`~stix.sampling.steps.DRIFT_FNS`); the
        diffusion does the same with :data:`~stix.sampling.steps.DIFFUSION_FNS`.
        The ODE path has no diffusion at all, hence ``None`` in its place.

        Args:
            graphdef: Static nnx graph definition of the generative model.
            stochasticity_scale_tree: Per-modality stochasticity scale callables or ``None``.
            context_data: Context conditioning for the guidance recipe.
            context_mask: Context masks paired with ``context_data``.
            intrinsic_data: Intrinsic-guidance targets for the guidance recipe.
            intrinsic_mask: Intrinsic masks paired with ``intrinsic_data``.
            attention_mask: Per-modality attention masks.

        Returns:
            ``(drift_fn, diffusion_fn)``, or ``(drift_fn, None)`` on the ODE
            path.
        """
        is_sde = self.solver_config.stochasticity_scale is not None
        guidance_fn = (
            self.solver_config.guidance_fn or get_context_conditional_generator
        )
        guidance_scale = self.solver_config.guidance_scale
        direction = self.solver_config.direction

        def _drift_fn(t, z_t, gen_model_state):
            """Diffrax drift closure over the guided generators."""
            gen_model = nnx.merge(graphdef, gen_model_state)
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
            return gen_model.modality_registry.map(
                lambda modality, generator_k, stochasticity_scale_k: DRIFT_FNS[
                    modality.generator_type
                ](t, generator_k, stochasticity_scale_k, direction),
                generators,
                stochasticity_scale_tree,
            )

        if not is_sde:
            return _drift_fn, None

        def _diffusion_fn(t, z_t, gen_model_state):
            """Diffrax diffusion closure (diagonal operator)."""
            gen_model = nnx.merge(graphdef, gen_model_state)
            diagonals = gen_model.modality_registry.map(
                lambda modality, z_t_k, stochasticity_scale_k: DIFFUSION_FNS[
                    modality.generator_type
                ](t, z_t_k, stochasticity_scale_k),
                z_t,
                stochasticity_scale_tree,
            )
            # Diagonal operator over the whole state pytree.
            return lx.DiagonalLinearOperator(diagonals)

        return _drift_fn, _diffusion_fn
