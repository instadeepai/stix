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

from abc import ABC, abstractmethod
from typing import Any

from flax import nnx
from jaxtyping import PyTree

from stix.core.generator import Generator
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


class GenerativeModel[NetworkType: Network](nnx.Module, ABC):
    """Generative model base class.

    GenerativeModel objects contain everything needed to perform sampling.
    It is responsible for:

    - Embedding the data
    - Computing the network output
    - Computing the sampling-time generator (velocity, velocity+score, or rates)
    - Computing the loss

    To support classifier-style intrinsic guidance, the model must override the method :meth:`get_guidance_loss`
    (consumed by :func:`stix.sampling.guidance.get_intrinsic_guidance_generator`).
    """

    def __init__(
        self,
        network: NetworkType,
        modality_registry: ModalityRegistry,
    ):
        """Initialise the generative model.

        Args:
            network: The neural network module that is used to compute the model's predictions.
            modality_registry: A registry containing all modalities along with their embedders,
            interpolants, generator types and other metadata.
        """
        self.network = network
        self.modality_registry = modality_registry

        # All generative models require each modality to have these fields filled.
        modality_registry.assert_fields_set("interpolant", "embedder", "is_discrete")
        # All discrete modalities must have 'num_categories' set.
        modality_registry.assert_num_categories_set_on_discrete_modalities()

    def get_network_output(
        self,
        z_t: PyTree[Var],
        t: Time,
        context_data: PyTree[Var] | None,
        context_mask: PyTree[Mask | None] | None,
        attention_mask: PyTree[Mask | None] | None,
    ) -> Any:
        r"""Get the output from the neural network.

        Calls ``self.network`` positionally; the accepted call is the
        :class:`~stix.nn.Network` contract, which every network must satisfy.

        Args:
            z_t: Per-modality interpolated state in embedded space, at time ``t``.
            t: Interpolation time.
            context_data: Context conditioning (a pytree) consumed by the
                network's context encoder.
            context_mask: Context masks gating ``context_data``.
            attention_mask: Per-modality attention masks.

        Returns:
            The network's raw output (structure defined by the concrete network).
        """
        return self.network(z_t, t, context_data, context_mask, attention_mask)

    def get_embeddings(
        self, raw_batch: PyTree[RawSourceTargetPair]
    ) -> PyTree[EmbeddedSourceTargetPair]:
        r"""Embed raw data :math:`(x_{\mathrm{src}}, x_{\mathrm{tgt}})` to produce the embedded pairs :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.

        Args:
            raw_batch: Per-modality raw ``(source, target)`` pairs
                :math:`(x_{\mathrm{src}}, x_{\mathrm{tgt}})`.

        Returns:
            Per-modality embedded pairs :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.
            ``None`` raw source variables are mapped to ``None`` embedded source variables.
        """
        embedded_target = self.modality_registry.map(
            lambda modality, raw_pair: modality.embedder.from_raw_to_embeddings(
                raw_pair.target
            ),
            raw_batch,
        )

        embedded_source = self.modality_registry.map(
            lambda modality, raw_pair: (
                None
                if raw_pair.source is None
                else modality.embedder.from_raw_to_embeddings(raw_pair.source)
            ),
            raw_batch,
        )

        return self.modality_registry.map(
            lambda modality, target, source: EmbeddedSourceTargetPair(
                target=target, source=source
            ),
            embedded_target,
            embedded_source,
        )

    @abstractmethod
    def get_generator(
        self, net_out: Any, z_t: PyTree[Var], t: Time
    ) -> PyTree[Generator]:
        r"""Convert network output to per-modality sampling generators at :math:`(z_t, t)`.

        The returned pytree must have one
        :class:`~stix.core.generator.Generator` per modality, and each
        modality's generator **must** be an instance of its declared
        ``modality.generator_type`` (inferred from the interpolant).

        Args:
            net_out: The network's raw output.
            z_t: Per-modality noisy state in embedded space.
            t: Current time, :math:`t \in [0, 1]`.

        Returns:
            Per-modality sampling-time generators.
        """

    def get_guidance_loss(
        self,
        net_out: Any,
        z_t: PyTree[Var],
        t: Time,
        intrinsic_data: PyTree[Var],
        intrinsic_mask: PyTree[Mask | None],
    ) -> Scalar:
        r"""An optional scalar loss used for conditional intrinsic guidance.

        It is consumed by :func:`stix.sampling.guidance.get_intrinsic_guidance_generator`
        at sampling time to guide the sampling process towards the conditional data.

        More precisely, the gradient of this loss is used as an estimate of
        the conditional score
        :math:`\nabla_{z_t}\log p(c | z_t) \simeq \nabla_{z_t} \ell (c , z_t)`.

        This method is not implemented by default, as the exact form of the guidance loss
        typically depends on the interpretation of the network output and the embedded variables,
        which are specified by the generative model. See the `conditioning and guidance
        tutorial
        <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/4.conditioning_and_guidance.ipynb>`_
        for a worked implementation.

        Args:
            net_out: Per-modality network output for this sample.
            z_t: Per-modality state in embedded space.
            t: Current time, :math:`t \in [0, 1]`.
            intrinsic_data: Per-modality conditioning targets, one per registry
                modality.
            intrinsic_mask: Per-modality 0/1 masks; ``None`` (whole or per-entry)
                is treated as all-ones (no masking).

        Returns:
            A scalar guidance loss for this sample.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_guidance_loss. It is "
            "required only for classifier-style (intrinsic) guidance; override it "
            "to turn the network output into a scalar loss on a per-modality "
            "condition (see get_intrinsic_guidance_generator)."
        )

    @abstractmethod
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
        r"""Compute the scalar training loss for one interpolated sample.

        **The loss defines what the network learns to predict.** It is therefore
        inextricably linked to :meth:`get_generator`.

        For example, if the network output is trained against the conditional velocity,
        then ``get_generator`` (typically) wraps the network output directly as the
        velocity, deriving the score from it when a
        :class:`~stix.core.generator.VelocityAndScore` is required; if it
        is trained against some other target, that conversion changes accordingly.
        For this reason there is no sensible default — each concrete model must
        implement ``get_loss`` alongside its prediction-to-generator conversion.
        For more details regarding the implementation of a tailored
        GenerativeModel, see the `generative model tutorial
        <https://github.com/instadeepai/stix/blob/main/tutorials/notebooks/3.generative_model.ipynb>`_.

        A common pattern is to compute one scalar loss per modality and reduce
        with :func:`~stix.core.loss.reduce_modality_losses`::

            def get_loss(self, net_out, t, z_t, epsilon,
                         embedded_pairs, raw_pairs, loss_mask):
                criterion = MSECriterion()
                losses = self.modality_registry.map(
                    lambda modality, net_out_k, pair_k, eps_k, mask_k: criterion(
                        net_out_k,
                        modality.interpolant.get_conditional_velocity(
                            pair_k, t, eps_k
                        ),
                        t,
                        mask_k,
                    ),
                    net_out, embedded_pairs, epsilon, loss_mask,
                )
                return reduce_modality_losses(losses)

        Losses that cannot be decomposed per modality (e.g. cross-modality /
        coupled losses) are equally valid — simply implement them directly here
        rather than mapping over the registry.

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
