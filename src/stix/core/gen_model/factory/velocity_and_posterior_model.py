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

from typing import Any

import jax
from jaxtyping import PyTree

from stix.core.gen_model.factory.utils import (
    collapse_mask,
    drop_mask_symbol_from_one_hot,
    mask_out_revealed_states,
)
from stix.core.gen_model.gen_model import GenerativeModel
from stix.core.generator import Generator, TransitionRates, VelocityAndScore
from stix.core.interpolant.discrete_interpolant import DiscreteInterpolant
from stix.core.interpolant.interpolant import OneSidedInterpolant
from stix.core.interpolant.linear_interpolant import (
    OneSidedLinearStochasticInterpolant,
)
from stix.core.interpolant.standard_interpolants.discrete_diffusion import (
    MaskDiscreteInterpolant,
)
from stix.core.loss.criterion import CrossEntropyCriterion, MSECriterion
from stix.core.loss.utils import reduce_modality_losses
from stix.core.modality import ModalityRegistry
from stix.nn.network import Network
from stix.typing import (
    EmbeddedSourceTargetPair,
    Mask,
    NoiseVar,
    RawSourceTargetPair,
    Scalar,
    Time,
    Var,
)


class VelocityAndPosteriorGenerativeModel[NetworkType: Network](
    GenerativeModel[NetworkType]
):
    r"""Joint model over continuous one-sided-linear and discrete DFM modalities.

    Continuous modalities behave like
    :class:`~stix.core.gen_model.factory.VelocityOneSidedGenerativeModel`
    (velocity MSE, ``VelocityAndScore`` generator); discrete modalities behave
    like
    :class:`~stix.core.gen_model.factory.PosteriorMixtureGenerativeModel`
    (denoising cross-entropy, ``TransitionRates`` generator). Dispatch is based on the
    modality's interpolant type, resolved per modality.

    Continuous modality's must have interpolant of type :class:`~stix.core.interpolant.OneSidedLinearStochasticInterpolant`.
    Discrete modalities must have a one-sided :class:`~stix.core.interpolant.DiscreteInterpolant`
    implementing `rates_from_target_posterior`, and set ``num_categories``.
    """

    def __init__(
        self,
        network: NetworkType,
        modality_registry: ModalityRegistry,
    ):
        r"""Initialise and validate each modality's interpolant type.

        Args:
            network: The neural network module.
            modality_registry: The registry. Each modality's interpolant must be a
                one-sided linear SI (continuous) or a discrete interpolant;
                discrete modalities need ``num_categories``.

        Raises:
            TypeError: If a modality's interpolant is neither supported type.
            ValueError: If a discrete modality lacks ``num_categories``.
        """
        super().__init__(network, modality_registry)
        for modality in modality_registry._modality_leaves():
            interpolant = modality.interpolant
            if isinstance(interpolant, DiscreteInterpolant):
                if not isinstance(interpolant, OneSidedInterpolant):
                    raise TypeError(
                        f"{self.__class__.__name__} requires discrete modalities "
                        "to use a one-sided DiscreteInterpolant, but got "
                        f"{type(interpolant).__name__} for modality {modality}."
                    )
                if not hasattr(interpolant, "rates_from_target_posterior"):
                    raise TypeError(
                        f"{self.__class__.__name__} requires discrete modalities "
                        "to use an interpolant implementing "
                        f"`rates_from_target_posterior`, but got "
                        f"{type(interpolant).__name__} for modality {modality}."
                    )
                if modality.num_categories is None:
                    raise ValueError(
                        f"{self.__class__.__name__}: a discrete modality "
                        f"must set num_categories, but it is None for modality {modality}."
                    )
            elif not isinstance(interpolant, OneSidedLinearStochasticInterpolant):
                raise TypeError(
                    f"{self.__class__.__name__} requires continuous modalities "
                    "to use a OneSidedLinearStochasticInterpolant, but got "
                    f"{type(interpolant).__name__} for modality {modality}."
                )

    def get_loss(
        self,
        net_out: Any,
        t: Time,
        z_t: PyTree[Var],
        epsilon: PyTree[NoiseVar],
        embedded_pairs: PyTree[EmbeddedSourceTargetPair],
        raw_pairs: PyTree[RawSourceTargetPair],
        loss_mask: PyTree[Mask],
    ) -> Scalar:
        """Per-modality velocity MSE (continuous) or weighted CE (discrete).

        Args:
            net_out: The network's raw output for this sample.
            t: Interpolation time, shared across modalities.
            z_t: Per-modality interpolated state in embedded space.
            epsilon: Per-modality noise sample drawn for the interpolation path.
            embedded_pairs: Per-modality embedded ``(source, target)`` pairs.
            raw_pairs: Per-modality raw-space ``(source, target)`` pairs.
            loss_mask: Per-modality loss mask (all-ones when unmasked).

        Returns:
            A scalar loss for this sample.
        """
        mse = MSECriterion()
        ce = CrossEntropyCriterion()

        def _per_modality(
            modality,
            net_out_k,
            z_t_k,
            epsilon_k,
            embedded_pair_k,
            mask_k,
        ):
            """Per-modality helper mapped over the registry."""
            interpolant = modality.interpolant
            if isinstance(interpolant, DiscreteInterpolant):
                one_hot_target = embedded_pair_k.target
                active = collapse_mask(mask_k, z_t_k)
                if isinstance(interpolant, MaskDiscreteInterpolant):
                    one_hot_target = drop_mask_symbol_from_one_hot(
                        one_hot_target, interpolant.mask_index
                    )
                    active = mask_out_revealed_states(
                        active, z_t_k, interpolant.mask_index
                    )
                return ce(net_out_k, one_hot_target, t, active[..., None])
            conditional_velocity = modality.interpolant.get_conditional_velocity(
                embedded_pair_k, t, epsilon_k
            )
            return mse(net_out_k, conditional_velocity, t, mask_k)

        per_modality_losses = self.modality_registry.map(
            _per_modality,
            net_out,
            z_t,
            epsilon,
            embedded_pairs,
            loss_mask,
        )
        return reduce_modality_losses(per_modality_losses)

    def get_generator(
        self, net_out: Any, z_t: PyTree[Var], t: Time
    ) -> PyTree[Generator]:
        r"""Per-modality ``VelocityAndScore`` (continuous) or ``TransitionRates`` (discrete).

        Args:
            net_out: The network's raw output.
            z_t: Per-modality noisy state in embedded space.
            t: Current time, :math:`t \in [0, 1]`.

        Returns:
            Per-modality sampling-time generators.
        """

        def _per_modality(modality, net_out_k, z_t_k):
            """Per-modality helper mapped over the registry."""
            if isinstance(modality.interpolant, DiscreteInterpolant):
                posterior = jax.nn.softmax(net_out_k, axis=-1)
                # __init__ already checks every discrete modality implements `rates_from_target_posterior`.
                return TransitionRates(
                    forward_rates=modality.interpolant.rates_from_target_posterior(  # type: ignore[attr-defined]
                        posterior, z_t_k, t, backward=False
                    ),
                    backward_rates=modality.interpolant.rates_from_target_posterior(  # type: ignore[attr-defined]
                        posterior, z_t_k, t, backward=True
                    ),
                )
            score_k = modality.interpolant.score_from_velocity(net_out_k, z_t_k, t)
            return VelocityAndScore(velocity=net_out_k, score=score_k)

        return self.modality_registry.map(_per_modality, net_out, z_t)
