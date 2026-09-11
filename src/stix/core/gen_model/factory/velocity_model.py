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

from jaxtyping import PyTree

from stix.core.gen_model.gen_model import GenerativeModel
from stix.core.generator import Generator, Velocity, VelocityAndScore
from stix.core.interpolant.linear_interpolant import (
    LinearDeterministicInterpolant,
    OneSidedLinearStochasticInterpolant,
)
from stix.core.loss.criterion import MSECriterion
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


class VelocityOneSidedGenerativeModel[NetworkType: Network](
    GenerativeModel[NetworkType]
):
    r"""One-sided model whose network output *is* the (embedded) velocity field.

    A ready-made, predefined :class:`~stix.core.gen_model.GenerativeModel`: the
    single choice to *train the network on the velocity* fixes both methods
    the rest of the library relies on.

    - :meth:`get_loss` — a per-modality MSE between the network output and the
      interpolant's conditional velocity, reduced across modalities.
    - :meth:`get_generator` — the network output already *is* the velocity; the
      score is derived from it via the interpolant and both are returned as a
      :class:`~stix.core.generator.VelocityAndScore`.

    :meth:`~stix.core.gen_model.GenerativeModel.get_guidance_loss` is
    **not** implemented here. To use this model with
    :func:`~stix.sampling.guidance.get_intrinsic_guidance_generator`, subclass it and
    override that hook. See the `conditioning and guidance tutorial
    <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/4.conditioning_and_guidance.ipynb>`_
    for a worked implementation.

    The loss is a velocity MSE (:class:`~stix.core.loss.MSECriterion`) — that is
    what makes this a *velocity* model, so it is not configurable here.
    """

    def __init__(
        self,
        network: NetworkType,
        modality_registry: ModalityRegistry,
    ):
        r"""Initialise the model and check every interpolant is one-sided linear.

        Args:
            network: The neural network module.
            modality_registry: The registry of modalities. Every modality's
                interpolant must be an
                :class:`~stix.core.interpolant.OneSidedLinearStochasticInterpolant`,
                since the velocity/score conversions rely on the one-sided linear
                primitives.

        Raises:
            TypeError: If any modality's interpolant is not an
                ``OneSidedLinearStochasticInterpolant``.
        """
        super().__init__(network, modality_registry)

        for modality in modality_registry._modality_leaves():
            if not isinstance(
                modality.interpolant, OneSidedLinearStochasticInterpolant
            ):
                raise TypeError(
                    f"{self.__class__.__name__} requires every modality to use an "
                    "OneSidedLinearStochasticInterpolant (its velocity/score "
                    "conversions rely on the one-sided linear primitives), but got "
                    f"{type(modality.interpolant).__name__}."
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
        """Velocity MSE: compare the network output to the conditional velocity.

        One scalar loss is computed per modality (the interpolant provides the
        ground-truth conditional velocity), then reduced across modalities with
        :func:`~stix.core.loss.reduce_modality_losses`.

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

        def _per_modality_network_output_to_loss(
            modality, net_out_k, embedded_pair_k, epsilon_k, mask_k
        ):
            """Per-modality scalar loss for one registry leaf."""
            conditional_velocity = modality.interpolant.get_conditional_velocity(
                embedded_pair_k, t, epsilon_k
            )
            return mse(net_out_k, conditional_velocity, t, mask_k)

        per_modality_losses = self.modality_registry.map(
            _per_modality_network_output_to_loss,
            net_out,
            embedded_pairs,
            epsilon,
            loss_mask,
        )
        return reduce_modality_losses(per_modality_losses)

    def get_generator(
        self, net_out: Any, z_t: PyTree[Var], t: Time
    ) -> PyTree[Generator]:
        r"""Wrap the predicted velocity as a :class:`~stix.core.generator.VelocityAndScore`.

        The network output already *is* the velocity; the score is derived from
        it via the interpolant. ODE sampling ignores the score when
        ``stochasticity_scale`` is ``None``.

        Args:
            net_out: The network's raw output.
            z_t: Per-modality noisy state in embedded space.
            t: Current time, :math:`t \in [0, 1]`.

        Returns:
            Per-modality sampling-time generators.
        """

        def _per_modality_generator(modality, net_out_k, z_t_k):
            """Per-modality generator built from the network output."""
            score_k = modality.interpolant.score_from_velocity(net_out_k, z_t_k, t)
            return VelocityAndScore(velocity=net_out_k, score=score_k)

        return self.modality_registry.map(_per_modality_generator, net_out, z_t)


class VelocityTwoSidedGenerativeModel[NetworkType: Network](
    GenerativeModel[NetworkType]
):
    r"""Two-sided model whose network output *is* the (embedded) velocity field.

    The two-sided counterpart of :class:`VelocityOneSidedGenerativeModel`: source
    and target are both *data* distributions, transported into one another, rather
    than noise into data.

    - :meth:`get_loss` — a per-modality MSE between the network output and the
      interpolant's conditional velocity, reduced across modalities.
    - :meth:`get_generator` — wraps the network output as a
      :class:`~stix.core.generator.Velocity`. A deterministic interpolant
      carries no noise term, so no score exists and sampling is ODE-only
      (``stochasticity_scale=None``); that generator type is inferred from the
      interpolant.

    The loss is a velocity MSE (:class:`~stix.core.loss.MSECriterion`) — that is
    what makes this a *velocity* model, so it is not configurable here.
    """

    def __init__(
        self,
        network: NetworkType,
        modality_registry: ModalityRegistry,
    ):
        r"""Initialise the model and check every interpolant is linear deterministic.

        Args:
            network: The neural network module.
            modality_registry: The registry of modalities. Every modality's
                interpolant must be a
                :class:`~stix.core.interpolant.LinearDeterministicInterpolant`,
                since the velocity conversion relies on the linear primitives and
                the absence of a noise term.

        Raises:
            TypeError: If any modality's interpolant is not a
                ``LinearDeterministicInterpolant``.
        """
        super().__init__(network, modality_registry)

        for modality in modality_registry._modality_leaves():
            if not isinstance(modality.interpolant, LinearDeterministicInterpolant):
                raise TypeError(
                    f"{self.__class__.__name__} requires every modality to use a "
                    "LinearDeterministicInterpolant (its velocity conversion relies "
                    "on the linear primitives, and a deterministic interpolant is "
                    "what makes the two-sided transport noise-free), but got "
                    f"{type(modality.interpolant).__name__}."
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
        """Velocity MSE: compare the network output to the conditional velocity.

        One scalar loss is computed per modality (the interpolant provides the
        ground-truth conditional velocity), then reduced across modalities with
        :func:`~stix.core.loss.reduce_modality_losses`.

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

        def _per_modality_network_output_to_loss(
            modality, net_out_k, embedded_pair_k, epsilon_k, mask_k
        ):
            """Per-modality scalar loss for one registry leaf."""
            conditional_velocity = modality.interpolant.get_conditional_velocity(
                embedded_pair_k, t, epsilon_k
            )
            return mse(net_out_k, conditional_velocity, t, mask_k)

        per_modality_losses = self.modality_registry.map(
            _per_modality_network_output_to_loss,
            net_out,
            embedded_pairs,
            epsilon,
            loss_mask,
        )
        return reduce_modality_losses(per_modality_losses)

    def get_generator(
        self, net_out: Any, z_t: PyTree[Var], t: Time
    ) -> PyTree[Generator]:
        r"""Wrap the predicted velocity as a :class:`~stix.core.generator.Velocity` generator.

        A deterministic interpolant carries no noise term and therefore defines
        no score, so the inferred generator type is ``Velocity``; SDE sampling is
        unavailable and the process is an ODE.

        Args:
            net_out: The network's raw output.
            z_t: Per-modality noisy state in embedded space.
            t: Current time, :math:`t \in [0, 1]`.

        Returns:
            Per-modality sampling-time generators.
        """

        def _per_modality_generator(modality, net_out_k, z_t_k):
            """Per-modality generator built from the network output."""
            del modality, z_t_k
            return Velocity(velocity=net_out_k)

        return self.modality_registry.map(_per_modality_generator, net_out, z_t)
