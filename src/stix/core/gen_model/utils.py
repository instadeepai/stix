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

from jaxtyping import PyTree

from stix.core.interpolant.interpolant import OneSidedInterpolant
from stix.core.modality import (
    Modality,
    ModalityRegistry,
    assert_embedded_source_prior_valid,
)
from stix.typing import EmbeddedSourceTargetPair, PRNGKeyArray, Shape, Var


def sample_embedded_source_from_prior(
    modality: Modality,
    key: PRNGKeyArray,
    shape: Shape,
) -> Var:
    r"""Sample an embedded source :math:`z_{\mathrm{src}}` from ``modality.embedded_source_prior``.

    Args:
        modality: Modality carrying a non-``None`` ``embedded_source_prior``.
        key: PRNG key for the draw.
        shape: Shape of the embedded source to draw, typically ``(*batch_dims, *embedding_shape)``

    Returns:
        An embedded source sample :math:`z_{\mathrm{src}}` of shape ``shape``.

    Raises:
        ValueError: If ``embedded_source_prior`` is unset, or is set on a
            one-sided interpolant.
    """
    assert_embedded_source_prior_valid(modality)
    prior = modality.embedded_source_prior
    if prior is None:
        raise ValueError(
            "Cannot sample an embedded source from a prior: "
            "embedded_source_prior is None."
        )
    return prior(key, shape)


def assert_two_sided_sources_available(
    modality_registry: ModalityRegistry,
    embedded_batch: PyTree[EmbeddedSourceTargetPair],
) -> None:
    r"""Raise if a two-sided modality has no embedded source.

    Args:
        modality_registry: Registry defining the modality tree.
        embedded_batch: Embedded pairs after prior fill.

    Raises:
        ValueError: If a two-sided modality has ``source is None``.
    """

    def _check(modality: Modality, pair: EmbeddedSourceTargetPair) -> None:
        """Validate one modality leaf has a source when two-sided."""
        if isinstance(modality.interpolant, OneSidedInterpolant):
            return
        if pair.source is None:
            raise ValueError(
                "Two-sided interpolant requires an embedded source: provide "
                "a raw source in the batch, or set embedded_source_prior on "
                f"the modality (interpolant={type(modality.interpolant).__name__})."
            )

    modality_registry.map(_check, embedded_batch)


def fill_embedded_source_from_prior(
    modality_registry: ModalityRegistry,
    embedded_batch: PyTree[EmbeddedSourceTargetPair],
    key: PRNGKeyArray,
) -> PyTree[EmbeddedSourceTargetPair]:
    r"""Fill embedded source :math:`z_{\mathrm{src}}` from ``embedded_source_prior`` when needed.

    Only modalities with a prior and a missing embedded source are filled.
    Existing :math:`z_{\mathrm{src}}` are kept; one-sided
    modalities keep ``source=None``.

    Args:
        modality_registry: Registry supplying per-modality priors.
        embedded_batch: Embedded pairs
            :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`.
        key: PRNG key; split per modality.

    Returns:
        Embedded pairs with prior-drawn :math:`z_{\mathrm{src}}` filled where
        applicable.

    Raises:
        ValueError: If ``embedded_source_prior`` is set on a one-sided interpolant,
            or if a two-sided modality still has no embedded source after fill
            (no raw source in the batch and no prior).
    """
    key_tree = modality_registry.split_and_project_key(key)

    def _per_modality(
        modality: Modality,
        pair: EmbeddedSourceTargetPair,
        key_k: PRNGKeyArray,
    ) -> EmbeddedSourceTargetPair:
        """Fill source from the prior when unset and a prior is configured."""
        assert_embedded_source_prior_valid(modality)
        if pair.source is not None or modality.embedded_source_prior is None:
            return pair
        source = sample_embedded_source_from_prior(modality, key_k, pair.target.shape)
        return EmbeddedSourceTargetPair(target=pair.target, source=source)

    filled = modality_registry.map(_per_modality, embedded_batch, key_tree)
    assert_two_sided_sources_available(modality_registry, filled)
    return filled
