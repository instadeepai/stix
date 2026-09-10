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


import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PyTree

from stix.core.coupling import Coupling
from stix.core.gen_model.gen_model import GenerativeModel
from stix.core.gen_model.utils import fill_embedded_source_from_prior
from stix.core.interpolant.continuous_interpolant import ContinuousInterpolant
from stix.core.modality import ModalityRegistry
from stix.training.time_sampler import TimeSampler, UniformTimeSampler
from stix.typing import (
    EmbeddedSourceTargetPair,
    EmbeddedVar,
    Mask,
    NoiseVar,
    PRNGKeyArray,
    RawSourceTargetPair,
    Scalar,
    Var,
    get_batch_size,
)
from stix.typing.data import Batch


class LossPipeline:
    """Base class for loss pipelines.

    The loss pipeline orchestrates the computation of the Monte Carlo estimator of the training loss for a batch of data.

    It performs the following steps:
    - Embeds the raw data.
    - Samples the embedded source for two-sided modalities that have an embedded source prior.
    - Applies the coupling strategy to the raw and embedded pairs.
    - Generates the noise for each modality.
    - Interpolates the embedded pairs.
    - Computes the network output for each modality.
    - Computes the loss for each modality.
    - Returns the mean loss and the metrics.
    """

    def __init__(
        self,
        coupling: Coupling | None = None,
        time_sampler: TimeSampler | None = None,
        antithetical: bool = False,
    ):
        """Initialise the loss with a coupling strategy.

        Args:
            coupling: Optional coupling applied to raw and embedded pairs.
                ``None`` (the default) leaves the batch pairing unchanged.
            time_sampler: Strategy for sampling time values. Defaults to uniform.
            antithetical: Whether to perform antithetical sampling. Averages the
                loss over ``(epsilon, -epsilon)`` pairs for variance reduction, at
                the cost of doubling the forward passes per step.
        """
        self.coupling = coupling
        self.time_sampler = time_sampler or UniformTimeSampler()
        self.antithetical = antithetical

    def _generate_noise(
        self,
        key: PRNGKeyArray,
        gen_model: GenerativeModel,
        embedded_batch: PyTree[EmbeddedSourceTargetPair],
    ) -> PyTree[NoiseVar]:
        r"""Generate the interpolant noise :math:`\epsilon` for each modality.

        The noise distribution is delegated to each modality's
        ``interpolant.sample_noise`` method.

        Args:
            key: PRNG key; split per modality.
            gen_model: Model whose registry supplies per-modality interpolants.
            embedded_batch: Embedded ``(source, target)`` pairs.

        Returns:
            Per-modality noise tree.
        """
        key_tree = gen_model.modality_registry.split_and_project_key(key)
        return gen_model.modality_registry.map(
            lambda modality, pair, key_k: modality.interpolant.sample_noise(
                key_k, pair.target.shape
            ),
            embedded_batch,
            key_tree,
        )

    def _fill_loss_mask(
        self,
        modality_registry: ModalityRegistry,
        z_t: PyTree[EmbeddedVar],
        loss_mask: PyTree[Mask | None] | None,
    ) -> PyTree[Mask]:
        """Fill default (all-ones) masks against the interpolated *state* structure.

        A whole-tree ``None`` means no masking; a per-modality ``None`` leaf
        defaults that modality to ones. The default mask shape is inferred
        from the interpolated embedded state :math:`z_t`.

        Args:
            modality_registry: Registry defining the modality tree.
            z_t: Interpolated state tree used as the mask shape reference.
            loss_mask: Optional loss-mask tree; ``None`` means all ones.

        Returns:
            A fully populated per-modality mask tree.
        """

        def _ones_like_state(z_t_k: EmbeddedVar) -> Mask:
            """All-ones float mask matching one state leaf."""
            # Float mask regardless of the (possibly integer) state dtype.
            return jnp.ones(jnp.shape(z_t_k))

        if loss_mask is None:
            return modality_registry.map(
                lambda modality, z_t_k: _ones_like_state(z_t_k), z_t
            )
        return modality_registry.map(
            lambda modality, state_k, loss_mask_k: (
                _ones_like_state(state_k) if loss_mask_k is None else loss_mask_k
            ),
            z_t,
            loss_mask,
        )

    def __call__(
        self,
        gen_model: GenerativeModel,
        batch: Batch,
        key: PRNGKeyArray,
    ) -> tuple[Scalar, PyTree[Scalar]]:
        """Compute the training loss for a batch of data.

        Args:
            gen_model: The generative model (``nnx.Module``). Must be passed
                explicitly so that ``nnx.value_and_grad`` can trace its parameters.
            batch: A :class:`~stix.typing.data.Batch` carrying the raw
                training batch, context data/masks, attention masks
                and loss masks.
            key: PRNG key for stochasticity.

        Returns:
            A pair ``(loss, metrics)``. ``metrics`` is a dict that always
            contains ``"loss"`` and may be extended by subclasses with extra
            diagnostics (e.g. per-modality losses, variance estimates).
        """
        raw_batch = batch.raw_batch
        context_data = batch.context_data
        context_mask = batch.context_mask
        attn_mask = batch.attn_mask
        loss_mask = batch.loss_mask

        # Embed raw data using the registry's per-modality embedders (traced by NNX).
        embedded_batch = gen_model.get_embeddings(raw_batch)

        # Sample embedded source z_src for two-sided modalities
        # for which the source is not provided in the batch,
        # using the priors provided in the modality registry.
        # Raises if a two-sided modality still has no source afterwards.
        key, prior_key = jr.split(key)
        embedded_batch = fill_embedded_source_from_prior(
            gen_model.modality_registry, embedded_batch, prior_key
        )

        if self.coupling is not None:
            raw_batch, embedded_batch = self.coupling(raw_batch, embedded_batch)

        key, epsilon_key = jr.split(key)
        epsilon = self._generate_noise(epsilon_key, gen_model, embedded_batch)

        # Per-sample forward pass: interpolate, predict, compute loss.
        def _step(
            embedded_pairs: PyTree[EmbeddedSourceTargetPair],
            epsilon: PyTree[NoiseVar],
            raw_sample: PyTree[RawSourceTargetPair],
            context_data: PyTree[Var] | None,
            context_mask: PyTree[Mask | None] | None,
            attn_mask: PyTree[Mask | None] | None,
            loss_mask: PyTree[Mask | None] | None,
            key: PRNGKeyArray,
        ):
            """Per-sample interpolate / predict / loss for ``vmap``."""
            t = self.time_sampler(key)

            z_t = gen_model.modality_registry.map(
                lambda modality, embedded_pairs_k, epsilon_k: (
                    modality.interpolant.interpolate(embedded_pairs_k, t, epsilon_k)
                ),
                embedded_pairs,
                epsilon,
            )
            net_out = gen_model.get_network_output(
                z_t, t, context_data, context_mask, attn_mask
            )

            # Fill loss mask with default values if not provided
            loss_mask_filled = self._fill_loss_mask(
                gen_model.modality_registry, z_t, loss_mask
            )

            loss = gen_model.get_loss(
                net_out,
                t,
                z_t,
                epsilon,
                embedded_pairs,
                raw_sample,
                loss_mask_filled,
            )

            return loss

        batch_size = get_batch_size(raw_batch)
        keys = jr.split(key, batch_size)  # per-sample time sampling
        losses = jax.vmap(_step)(
            embedded_batch,
            epsilon,
            raw_batch,
            context_data,
            context_mask,
            attn_mask,
            loss_mask,
            keys,
        )

        if self.antithetical:
            # Antithetical sampling negates epsilon, which is only valid for
            # continuous (Gaussian) noise.
            for modality in gen_model.modality_registry._modality_leaves():
                if not isinstance(modality.interpolant, ContinuousInterpolant):
                    raise ValueError(
                        "antithetical=True requires every modality's interpolant to "
                        "be a ContinuousInterpolant (symmetric Gaussian noise), but "
                        f"{type(modality.interpolant).__name__} is not."
                    )
            negative_epsilon = jax.tree.map(lambda e: -e, epsilon)
            losses_negative = jax.vmap(_step)(
                embedded_batch,
                negative_epsilon,
                raw_batch,
                context_data,
                context_mask,
                attn_mask,
                loss_mask,
                keys,
            )
            loss = (losses.mean() + losses_negative.mean()) / 2
        else:
            loss = losses.mean()

        return loss, {"loss": loss}
