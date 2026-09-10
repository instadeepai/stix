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

"""Shared primitives and helpers for the sampling solvers."""

from enum import Enum
from typing import Callable, Mapping

import jax
import jax.numpy as jnp

from stix.core.generator import Generator, TransitionRates, Velocity
from stix.typing import Scalar, Time


class Direction(Enum):
    """Integration direction for the ODE/SDE solver."""

    FORWARD = "forward"
    REVERSE = "reverse"


def padded_time_endpoints(
    t_min_tolerance: float, t_max_tolerance: float, direction: Direction
) -> tuple[float, float]:
    """Return the ``(t_start, t_end)`` both solvers integrate between.

    The interpolation domain is ``[0, 1]``; the endpoints are padded inwards by
    the two tolerances, then ordered by ``direction``.

    Args:
        t_min_tolerance: Pad above ``t=0``.
        t_max_tolerance: Pad below ``t=1``.
        direction: Integration direction; reverse swaps the pair.

    Returns:
        Pair ``(t_start, t_end)``, ordered so that integration runs from the
        first to the second.
    """
    t_min = t_min_tolerance
    t_max = 1.0 - t_max_tolerance
    if direction == Direction.FORWARD:
        return t_min, t_max
    return t_max, t_min


def check_scale_is_non_negative(scale_fn: Callable[[Time], Scalar]) -> bool:
    r"""Check that ``scale_fn`` is non-negative on the interior of ``[0, 1]``.

    Evaluated on a 1000-point grid excluding the endpoints, so a ``True`` result
    is necessary but not sufficient. ``True`` also comes back when the values are
    not concrete, which is the case for a callable closing over a traced value;
    :func:`~stix.sampling.steps.ctmc_euler_step` clamps those at run time.
    """
    tol = 1e-6
    n_grid_points = 1000
    times = jnp.linspace(0 + tol, 1 - tol, n_grid_points)
    try:
        return bool(jnp.all(jnp.asarray(scale_fn(times)) >= 0))
    except jax.errors.ConcretizationTypeError:
        return True


def validate_generator_types(
    modality_registry,
    *,
    supported: Mapping[type[Generator], object],
    stochasticity_scale=None,
) -> None:
    r"""Validate each modality's ``generator_type`` against a solver before tracing.

    Args:
        modality_registry: Registry of modalities to validate.
        supported: Mapping of generator types the solver can step.
        stochasticity_scale: Optional per-modality stochasticity scale tree.

    Raises:
        TypeError: If a modality declares a ``generator_type`` the solver cannot
            step (i.e. absent from ``supported``).
        ValueError: If ``stochasticity_scale`` is set for a modality whose
            generator carries no score (:class:`~stix.core.generator.Velocity`),
            or if one of its callables is negative on ``[0, 1]``.
    """

    def _check(modality, stochasticity_scale_k):
        """Validate one modality against ``supported`` and the scale tree."""
        generator_type = modality.generator_type
        if generator_type not in supported:
            supported_names = ", ".join(
                sorted(supported_type.__name__ for supported_type in supported)
            )
            # Only the diffrax Solver excludes TransitionRates, so the CTMC
            # pointer is exactly the case where it is missing from `supported`.
            hint = (
                " Use ManualSolver for TransitionRates (CTMC) modalities."
                if TransitionRates not in supported
                else ""
            )
            raise TypeError(
                "This solver cannot simulate generator_type="
                f"{getattr(generator_type, '__name__', generator_type)!r}; it "
                f"supports only {supported_names}.{hint}"
            )
        if stochasticity_scale_k is not None and generator_type is Velocity:
            raise ValueError(
                "stochasticity_scale is set for a Velocity modality, which carries "
                "no score and therefore admits no SDE term. Use a stochastic "
                "continuous interpolant (VelocityAndScore) to sample its SDE, or "
                "drop the scale for ODE sampling."
            )
        if stochasticity_scale_k is not None and not check_scale_is_non_negative(
            stochasticity_scale_k
        ):
            raise ValueError(
                "stochasticity_scale must be non-negative, but it is negative "
                "somewhere on [0, 1] for a "
                f"{getattr(generator_type, '__name__', generator_type)!r} "
                "modality. It scales the SDE diffusion as sqrt(2 * lambda_t) "
                "and weighs the CTMC backward rates, neither of which admits a "
                "negative value."
            )
        return None

    def _per_modality_or_none(tree):
        """Align ``tree`` with the registry, or a ``None`` at every leaf."""
        if tree is None:
            return modality_registry.map(lambda modality: None)
        modality_registry.assert_compatible(tree)
        return tree

    stochasticity_scale_tree = _per_modality_or_none(stochasticity_scale)
    modality_registry.map(_check, stochasticity_scale_tree)
