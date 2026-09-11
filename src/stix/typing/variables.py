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

from typing import Annotated, NamedTuple

import jax
from jaxtyping import Array, Float, PyTree

# Raw variables
type RawCtsVar = Annotated[Float[Array, "..."], "continuous float variables"]
type RawDisVar = Annotated[Array, "raw discrete variables, usually integer indices"]
type RawVar = RawCtsVar | RawDisVar


class RawSourceTargetPair(NamedTuple):
    r"""Raw-space source/target pair for one modality.

    Attributes:
        target: Raw target variable :math:`x_{\mathrm{tgt}}`.
        source: Raw source variable :math:`x_{\mathrm{src}}`, or ``None`` when
            the modality has no source distribution (one-sided setups).
    """

    target: RawVar
    source: RawVar | None = None


def get_batch_size(raw_batch: PyTree[RawSourceTargetPair]) -> int:
    """Infer the batch size from a raw batch, validated across modalities.

    Args:
        raw_batch: A pytree whose leaves are :class:`RawSourceTargetPair`.

    Returns:
        The common leading (batch) dimension of each modality's ``target`` array.

    Raises:
        ValueError: If the batch is empty, or if modalities disagree on the
            leading (batch) dimension of their target arrays.
    """
    pairs = jax.tree.leaves(
        raw_batch, is_leaf=lambda x: isinstance(x, RawSourceTargetPair)
    )
    if not pairs:
        raise ValueError("Cannot infer batch_size from empty raw batch.")

    sizes = [pair.target.shape[0] for pair in pairs]
    if len(set(sizes)) > 1:
        raise ValueError(f"Inconsistent batch sizes across modalities: {sizes}")

    return sizes[0]


# Embedded variables
type EmbeddedVar = Array


class EmbeddedSourceTargetPair(NamedTuple):
    r"""Embedded-space source/target pair for one modality.

    Holds :math:`(z_{\mathrm{src}}, z_{\mathrm{tgt}})`, *not* automatically
    :math:`(z_0, z_1)`. See :class:`RawSourceTargetPair` for the
    :math:`z_{\mathrm{src}}` / :math:`z_0` distinction.

    Attributes:
        target: Embedded target :math:`z_{\mathrm{tgt}}`.
        source: Embedded source :math:`z_{\mathrm{src}}`, or ``None`` when absent
            (one-sided; then :math:`z_0 \neq z_{\mathrm{src}}` because there is
            no source variable).
    """

    target: EmbeddedVar
    source: EmbeddedVar | None = None


# Noise variables.
type NoiseVar = Array
