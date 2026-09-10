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

import jax.numpy as jnp
import optax

from stix.typing import (
    Mask,
    Scalar,
    Time,
    Var,
)


class Criterion(ABC):
    """Abstract per-modality criterion: a bare prediction-vs-ground-truth comparison."""

    @abstractmethod
    def __call__(
        self,
        prediction: Any,
        ground_truth: Any,
        t: Time,
        mask: Mask,
    ) -> Scalar:
        """Compute the per-modality loss for a single prediction/ground-truth pair."""
        pass


class MSECriterion(Criterion):
    """Mean squared error criterion for continuous modalities."""

    def __call__(
        self,
        prediction: Var,
        ground_truth: Var,
        t: Time,
        mask: Mask,
    ) -> Scalar:
        """Compute masked mean squared error between prediction and ground-truth.

        Args:
            prediction: Predicted variable, of shape ``(*dims, dim)``.
            ground_truth: Ground-truth variable, of shape ``(*dims, dim)``.
            t: Time step.
            mask: Mask, of shape ``(*dims, dim)``.

        Returns:
            The masked MSE loss averaged over the unmasked entries. Note
            :class:`~stix.training.loss_pipeline.LossPipeline` ``vmap`` s over the
            batch, so this sees one sample, not a batch.
        """
        residual = prediction - ground_truth
        # Clamp denominator to 1.0 so fully-masked examples yield 0.0 loss, not NaN.
        return jnp.sum(residual**2 * mask) / jnp.maximum(mask.sum(), 1.0)


class CrossEntropyCriterion(Criterion):
    """Cross entropy for discrete modalities, computed on logits prediction vs one-hot ground-truth labels."""

    def __call__(
        self,
        prediction: Var,
        ground_truth: Var,
        t: Time,
        mask: Mask,
    ) -> Scalar:
        r"""Compute masked softmax cross entropy between predicted logits and one-hot ground-truth labels.

        Use ``optax.softmax_cross_entropy`` to compute the standard per-example cross
        entropy

        .. math::
            -\sum_k \mathrm{label}_k \cdot \mathrm{log-softmax}(\mathrm{logits})_k

        and returns its average over the unmasked entries.

        Args:
            prediction: Predicted logits, of shape ``(*dims, num_categories)``.
            ground_truth: One-hot ground-truth labels, of shape ``(*dims, num_categories)``.
            t: Time step.
            mask: Mask, of shape ``(*dims, num_categories)``.

        Returns:
            The softmax cross entropy loss averaged over the unmasked entries.
            Note :class:`~stix.training.loss_pipeline.LossPipeline` ``vmap`` s over
            the batch, so this sees one sample, not a batch.
        """
        per_example = optax.softmax_cross_entropy(prediction, ground_truth)
        # softmax_cross_entropy already reduces over the category axis, so collapse
        # mask's trailing axis too to align shapes for the element-wise multiply.
        example_mask = jnp.max(mask, axis=-1)
        return jnp.sum(per_example * example_mask) / jnp.maximum(
            example_mask.sum(), 1.0
        )
