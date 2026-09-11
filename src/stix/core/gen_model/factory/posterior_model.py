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
from stix.core.generator import TransitionRates
from stix.core.interpolant.discrete_interpolant import DiscreteInterpolant
from stix.core.interpolant.standard_interpolants.discrete_diffusion import (
    MaskDiscreteInterpolant,
)
from stix.core.loss.criterion import CrossEntropyCriterion
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


class PosteriorMixtureGenerativeModel[NetworkType: Network](
    GenerativeModel[NetworkType]
):
    r"""An all-discrete model over `DiscreteInterpolant` modalities with `rates_from_target_posterior`.

    The network head predicts target-posterior logits over the ``K`` data
    categories (for a mask interpolant the mask symbol is *excluded* from the logits).

    This model fixes:

    - :meth:`get_loss` — cross-entropy of the posterior logits against the embedded target one-hot.
    - :meth:`get_generator` — per-modality :class:`~stix.core.generator.TransitionRates` generators computed
      using the learned posterior logits and the interpolant's `rates_from_target_posterior` method.
    """

    def __init__(
        self,
        network: NetworkType,
        modality_registry: ModalityRegistry,
    ):
        r"""Initialise and validate every modality's interpolant implements `rates_from_target_posterior`.

        Args:
            network: The neural network module (its per-modality head must emit ``K`` posterior logits).
            modality_registry: The registry. Every modality's interpolant must be
                a `DiscreteInterpolant` implementing `rates_from_target_posterior`
                and its ``num_categories`` must be set.

        Raises:
            TypeError: If any modality's interpolant does not implement
                `rates_from_target_posterior`.
            ValueError: If any modality has `num_categories` unset.
        """
        super().__init__(network, modality_registry)
        modality_registry.assert_fields_set("num_categories")
        for modality in modality_registry._modality_leaves():
            interpolant = modality.interpolant
            if not isinstance(interpolant, DiscreteInterpolant) or not hasattr(
                interpolant, "rates_from_target_posterior"
            ):
                raise TypeError(
                    f"{self.__class__.__name__} requires every modality to use a "
                    f"`DiscreteInterpolant` implementing `rates_from_target_posterior`, "
                    f"but got {type(interpolant).__name__}."
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
        r"""Cross-entropy of the posterior logits against the embedded target one-hot.

        Args:
            net_out: The network's output, interpreted as posterior logits.
            t: Interpolation time.
            z_t: Per-modality interpolated state in embedded space.
            epsilon: Per-modality noise sample drawn for the interpolation path.
            embedded_pairs: Per-modality embedded ``(source, target)`` pairs.
            raw_pairs: Per-modality raw-space ``(source, target)`` pairs.
            loss_mask: Per-modality loss mask (all-ones when unmasked).

        Returns:
            A scalar loss for this sample.
        """
        del epsilon, raw_pairs
        ce = CrossEntropyCriterion()

        def _per_modality(modality, net_out_k, z_t_k, embedded_pair_k, mask_k):
            """Per-modality helper mapped over the registry."""
            interpolant = modality.interpolant
            one_hot_target = embedded_pair_k.target
            active = collapse_mask(mask_k, z_t_k)
            if isinstance(interpolant, MaskDiscreteInterpolant):
                one_hot_target = drop_mask_symbol_from_one_hot(
                    one_hot_target, interpolant.mask_index
                )
                active = mask_out_revealed_states(active, z_t_k, interpolant.mask_index)
            return ce(net_out_k, one_hot_target, t, active[..., None])

        per_modality_losses = self.modality_registry.map(
            _per_modality, net_out, z_t, embedded_pairs, loss_mask
        )
        return reduce_modality_losses(per_modality_losses)

    def get_generator(
        self, net_out: Any, z_t: PyTree[Var], t: Time
    ) -> PyTree[TransitionRates]:
        r"""Turn target-posterior logits into per-modality :class:`~stix.core.generator.TransitionRates`.

        Uses the interpolant's `rates_from_target_posterior` method.

        Args:
            net_out: The network's raw output.
            z_t: Per-modality noisy state in embedded space.
            t: Current time, :math:`t \in [0, 1]`.

        Returns:
            Per-modality sampling-time generators.
        """

        def _per_modality(modality, net_out_k, z_t_k):
            """Per-modality helper mapped over the registry."""
            posterior = jax.nn.softmax(net_out_k, axis=-1)
            return TransitionRates(
                forward_rates=modality.interpolant.rates_from_target_posterior(
                    posterior, z_t_k, t, backward=False
                ),
                backward_rates=modality.interpolant.rates_from_target_posterior(
                    posterior, z_t_k, t, backward=True
                ),
            )

        return self.modality_registry.map(_per_modality, net_out, z_t)
